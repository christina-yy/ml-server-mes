from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
import pandas as pd
import sqlalchemy
import os
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
DB_URL = os.environ.get(
    "DB_URL",
    "mysql+pymysql://root:@127.0.0.1:3306/manufacturing_db"
)

engine = sqlalchemy.create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_recycle=280,
    connect_args={"connect_timeout": 10}
)

FEATURES = [
    'defects', 'scrapRate', 'downTimeHours', 'energyConsumption',
    'maintenanceHours', 'reworkHours', 'qualityChecksFailed',
    'averageTemperature', 'averageHumidityPercent'
]

FEATURE_LABELS = {
    'defects':                 'Defects',
    'scrapRate':               'Scrap Rate',
    'downTimeHours':           'Downtime (hrs)',
    'energyConsumption':       'Energy Consumption',
    'maintenanceHours':        'Maintenance (hrs)',
    'reworkHours':             'Rework (hrs)',
    'qualityChecksFailed':     'Quality Checks Failed',
    'averageTemperature':      'Avg Temperature',
    'averageHumidityPercent':  'Avg Humidity (%)',
}

# Features used in scoring rules
SCORING_RULE_FEATURES = {
    'defects', 'scrapRate', 'downTimeHours',
    'reworkHours', 'qualityChecksFailed', 'maintenanceHours'
}

rf_model         = None
le               = LabelEncoder()
_feature_medians = {}


# ─────────────────────────────────────────────
# Auto Label Score (returns numeric score)
# ─────────────────────────────────────────────

def auto_label_score(row):
    score = 0

    if row['defects'] > 15:             score += 4
    elif row['defects'] > 12:           score += 3
    elif row['defects'] > 9:            score += 2
    elif row['defects'] > 7:            score += 1

    if row['scrapRate'] > 0.046:        score += 3
    elif row['scrapRate'] > 0.040:      score += 2
    elif row['scrapRate'] > 0.030:      score += 1

    if row['downTimeHours'] > 2.5:      score += 2
    elif row['downTimeHours'] > 1.5:    score += 1

    if row['reworkHours'] > 1.8:        score += 1
    elif row['reworkHours'] > 1.4:      score += 0.5

    if row['qualityChecksFailed'] == 2: score += 2
    elif row['qualityChecksFailed'] == 1: score += 1

    if row['maintenanceHours'] > 4.5:   score += 2
    elif row['maintenanceHours'] > 4.0: score += 1

    return score


# ─────────────────────────────────────────────
# Auto Label
# ─────────────────────────────────────────────

def auto_label(row):
    score = auto_label_score(row)
    if score >= 4:   return "Critical"
    elif score >= 2: return "At Risk"
    else:            return "Healthy"


# ─────────────────────────────────────────────
# Rule-based Explanation
# ─────────────────────────────────────────────

def explain_label(row):
    reasons = []
    score   = 0

    # --- Defects ---
    if row['defects'] > 15:
        score += 4
        reasons.append(f"Defects critically high ({row['defects']:.0f} units — threshold >15)")
    elif row['defects'] > 12:
        score += 3
        reasons.append(f"Defects very high ({row['defects']:.0f} units — threshold >12)")
    elif row['defects'] > 9:
        score += 2
        reasons.append(f"Defects elevated ({row['defects']:.0f} units — threshold >9)")
    elif row['defects'] > 7:
        score += 1
        reasons.append(f"Defects slightly above normal ({row['defects']:.0f} units — threshold >7)")

    # --- Scrap Rate ---
    if row['scrapRate'] > 0.046:
        score += 3
        reasons.append(f"Scrap rate critically high ({row['scrapRate']:.4f} — threshold >0.046)")
    elif row['scrapRate'] > 0.040:
        score += 2
        reasons.append(f"Scrap rate elevated ({row['scrapRate']:.4f} — threshold >0.040)")
    elif row['scrapRate'] > 0.030:
        score += 1
        reasons.append(f"Scrap rate slightly above normal ({row['scrapRate']:.4f} — threshold >0.030)")

    # --- Downtime ---
    if row['downTimeHours'] > 2.5:
        score += 2
        reasons.append(f"Downtime high ({row['downTimeHours']:.2f}h — threshold >2.5h)")
    elif row['downTimeHours'] > 1.5:
        score += 1
        reasons.append(f"Downtime moderately elevated ({row['downTimeHours']:.2f}h — threshold >1.5h)")

    # --- Rework ---
    if row['reworkHours'] > 1.8:
        score += 1
        reasons.append(f"Rework hours high ({row['reworkHours']:.2f}h — threshold >1.8h)")
    elif row['reworkHours'] > 1.4:
        score += 0.5
        reasons.append(f"Rework hours slightly elevated ({row['reworkHours']:.2f}h — threshold >1.4h)")

    # --- Quality Checks ---
    if row['qualityChecksFailed'] == 2:
        score += 2
        reasons.append("Quality checks failed: both checks failed (2/2)")
    elif row['qualityChecksFailed'] == 1:
        score += 1
        reasons.append("Quality checks failed: 1 out of 2 checks failed")

    # --- Maintenance ---
    if row['maintenanceHours'] > 4.5:
        score += 2
        reasons.append(f"Maintenance hours very high ({row['maintenanceHours']:.2f}h — threshold >4.5h)")
    elif row['maintenanceHours'] > 4.0:
        score += 1
        reasons.append(f"Maintenance hours elevated ({row['maintenanceHours']:.2f}h — threshold >4.0h)")

    # --- Healthy fallback ---
    if not reasons:
        reasons.append("All key metrics are within normal operating ranges.")

    return {"score": score, "reasons": reasons}


# ─────────────────────────────────────────────
# Impute
# ─────────────────────────────────────────────

def impute(df):
    for col in FEATURES:
        if col in df.columns:
            median = _feature_medians.get(col)
            if median is not None:
                df[col] = df[col].fillna(median)
            else:
                df[col] = df[col].fillna(df[col].median())
    return df


# ─────────────────────────────────────────────
# Train Model
# ─────────────────────────────────────────────

def train_model():
    global rf_model, le, _feature_medians
    try:
        today = datetime.today().strftime("%Y-%m-%d")
        with engine.connect() as conn:
            df = pd.read_sql(
                sqlalchemy.text("SELECT * FROM mes WHERE date <= :d ORDER BY date"),
                conn, params={"d": today}
            )
        if df.empty:
            logging.warning("No training data found — model not trained.")
            return
        _train_core(df)
    except Exception as e:
        logging.error(f"train_model() failed: {e}")


def _train_core(df):
    global rf_model, le, _feature_medians

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    for col in FEATURES:
        if col in df.columns:
            _feature_medians[col] = df[col].median()

    df          = impute(df)
    df['label'] = df.apply(auto_label, axis=1)

    label_counts = df['label'].value_counts().to_dict()
    logging.info(f"Label distribution: {label_counts}")

    X = df[FEATURES].values
    y = le.fit_transform(df['label'])

    # ─────────────────────────────────────────────
    # Time-aware Train/Test Split (80/20)
    # ─────────────────────────────────────────────
    split_idx       = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ─────────────────────────────────────────────
    # SMOTE — Oversample Minority Classes
    # ─────────────────────────────────────────────
    try:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        logging.info(
            f"SMOTE applied — "
            f"before: {len(X_train)} rows, "
            f"after: {len(X_train_sm)} rows"
        )
    except Exception as e:
        logging.warning(f"SMOTE failed ({e}), using original training data.")
        X_train_sm, y_train_sm = X_train, y_train

    # ─────────────────────────────────────────────
    # Class Weights
    # ─────────────────────────────────────────────
    cw = {
        i: (5 if cls == "Critical" else 1)
        for i, cls in enumerate(le.classes_)
    }

    # ─────────────────────────────────────────────
    # Train Random Forest
    # FIX: added n_jobs=1 to prevent thread explosion on Render
    # ─────────────────────────────────────────────
    rf_model = RandomForestClassifier(
        n_estimators=500,
        n_jobs=1,
        class_weight=cw,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42
    )
    rf_model.fit(X_train_sm, y_train_sm)

    # ─────────────────────────────────────────────
    # Test Set Evaluation
    # ─────────────────────────────────────────────
    if len(X_test) > 0:
        y_pred = rf_model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0
        )
        logging.info(f"Classification report:\n{report}")

        cm        = confusion_matrix(y_test, y_pred)
        cm_header = f"{'':>12}" + "".join(f"{c:>12}" for c in le.classes_)
        cm_rows   = "\n".join(
            f"{le.classes_[i]:>12}" + "".join(f"{v:>12}" for v in row)
            for i, row in enumerate(cm)
        )
        logging.info(f"Confusion matrix:\n{cm_header}\n{cm_rows}")

    # ─────────────────────────────────────────────
    # Feature Importance Log
    # ─────────────────────────────────────────────
    importances = rf_model.feature_importances_
    fi_sorted   = sorted(
        zip(FEATURES, importances),
        key=lambda x: x[1],
        reverse=True
    )
    fi_log = " | ".join(
        f"{FEATURE_LABELS[f]}: {round(v * 100, 4)}%"
        for f, v in fi_sorted
    )
    logging.info(f"Feature importances: {fi_log}")

    logging.info("Cross validation skipped (disabled for Render deployment).")
    logging.info(f"Model trained on {len(df)} rows. Classes: {list(le.classes_)}")
    return label_counts


train_model()


# ─────────────────────────────────────────────
# Intent / Parse Helpers
# ─────────────────────────────────────────────

INTENT_KEYWORDS = {
    "units":           ["unit", "units produced", "how many units", "production count", "produced"],
    "defects":         ["defect", "defects", "faulty", "faults", "failures"],
    "downtime":        ["downtime", "down time", "down hours", "machine down"],
    "maintenance":     ["maintenance", "maintenance hours", "serviced"],
    "scrap":           ["scrap", "scrap rate", "waste rate"],
    "rework":          ["rework", "rework hours", "redo"],
    "quality":         ["quality", "quality checks", "quality failed", "failed checks"],
    "energy":          ["energy", "energy consumption", "power usage", "electricity"],
    "temperature":     ["temperature", "temp", "heat"],
    "humidity":        ["humidity", "humid", "moisture"],
    "cost":            ["cost summary", "total cost", "price per unit", "cost breakdown"],
    "summary":         ["summary", "overview", "report", "overall"],
    "predict":         ["predict", "status", "health", "condition", "forecast"],
    "compare":         ["compare", "vs", "versus", "difference between"],
    "top":             ["top 3", "top 5", "best machine", "highest production", "ranking"],
    "operators":       ["operator", "operators", "staff", "worker"],
    "volume":          ["volume", "production volume", "cubic"],
    "shift":           ["shift", "day shift", "night shift"],
    "machine_id":      ["machine id", "machine ids", "which machine", "what machine", "list machine"],
    "production_time": ["production time", "how long", "time taken", "duration"],
    "product_type":    ["product type", "product types", "what product", "which product", "type of product"],
    "production_id":   ["production id", "production ids", "record id", "record ids", "list records", "all records"],
    "material_cost":   ["material cost", "material cost per unit", "material price"],
    "labour_cost":     ["labour cost", "labor cost", "labour cost per unit", "labor cost per unit", "labour price", "worker cost"],
    "why":             ["why", "reason", "explain", "cause", "because", "how come"],
}

INTENT_FIELD_MAP = {
    "units":           "unitsProduced",
    "defects":         "defects",
    "downtime":        "downTimeHours",
    "maintenance":     "maintenanceHours",
    "scrap":           "scrapRate",
    "rework":          "reworkHours",
    "quality":         "qualityChecksFailed",
    "energy":          "energyConsumption",
    "temperature":     "averageTemperature",
    "humidity":        "averageHumidityPercent",
    "operators":       "operatorCount",
    "volume":          "productionVolumeCubicMeters",
    "shift":           "shift",
    "machine_id":      "machineID",
    "production_time": "ProductionTime",
    "product_type":    "productType",
    "production_id":   "productionID",
    "material_cost":   "materialCostPerUnit",
    "labour_cost":     "labourCostPerUnit",
    "why":             None,
}


def detect_intent(text):
    text = text.lower()
    for priority in ["why", "predict", "compare", "top", "summary"]:
        if any(k in text for k in INTENT_KEYWORDS[priority]):
            return priority
    scores = {i: sum(1 for kw in kws if kw in text) for i, kws in INTENT_KEYWORDS.items()}
    scores = {i: s for i, s in scores.items() if s > 0}
    return max(scores, key=scores.get) if scores else "unknown"


def extract_machine_id(text):
    text = text.lower()
    for pat in [r'machine\s*no\.?\s*(\d+)', r'machine\s*number\s*(\d+)',
                r'machine\s*(\d+)', r'\bm\s*(\d+)\b']:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def extract_top_n(text):
    m = re.search(r'top\s*(\d+)', text.lower())
    return int(m.group(1)) if m else 3


def extract_compare_machines(text):
    matches = re.findall(r'machine\s*(\d+)', text.lower())
    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])
    return None, None


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": rf_model is not None,
        "timestamp":   datetime.now().isoformat()
    })


# ─────────────────────────────────────────────
# Parse Route
# ─────────────────────────────────────────────

@app.route("/parse", methods=["POST"])
def parse():
    try:
        text       = request.json.get("message", "")
        intent     = detect_intent(text)
        machine_id = extract_machine_id(text)
        m1, m2     = extract_compare_machines(text) if intent == "compare" else (None, None)
        top_n      = extract_top_n(text) if intent == "top" else None
        return jsonify({
            "intent":           intent,
            "machine_id":       machine_id,
            "requested_field":  INTENT_FIELD_MAP.get(intent),
            "compare_machines": [m1, m2] if intent == "compare" else None,
            "top_n":            top_n,
        })
    except Exception as e:
        logging.error(f"/parse error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Predict Route
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        machine_id = request.json.get("machineID")
        if rf_model is None:
            return jsonify({"error": "Model not trained yet. Call /retrain first."})

        today = datetime.today().strftime("%Y-%m-%d")
        with engine.connect() as conn:
            df = pd.read_sql(
                sqlalchemy.text("""
                    SELECT * FROM mes
                    WHERE machineID = :mid AND date <= :d
                    ORDER BY date DESC LIMIT 7
                """),
                conn, params={"mid": machine_id, "d": today}
            )
        if df.empty:
            return jsonify({"error": f"No data found for machine {machine_id}"})

        df         = impute(df)
        row        = df[FEATURES].median()
        row_values = row.values.reshape(1, -1)

        pred  = rf_model.predict(row_values)[0]
        proba = rf_model.predict_proba(row_values)[0]
        label = le.inverse_transform([pred])[0]

        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        rule_exp = explain_label(row)

        return jsonify({
            "machineID":     machine_id,
            "status":        label,
            "confidence":    round(float(max(proba)) * 100, 1),
            "probabilities": class_probs,
            "explanation": {
                "score":   rule_exp["score"],
                "reasons": rule_exp["reasons"],
            }
        })

    except Exception as e:
        logging.error(f"/predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Retrain Route
# ─────────────────────────────────────────────

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        train_model()
        return jsonify({
            "message":     "Model retrained successfully",
            "model_ready": rf_model is not None,
        })
    except Exception as e:
        logging.error(f"/retrain error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Predict Raw
# ─────────────────────────────────────────────

@app.route("/predict-raw", methods=["POST"])
def predict_raw():
    try:
        if rf_model is None:
            return jsonify({"error": "Model not trained yet."}), 400

        records = request.json.get("records", [])
        if not records:
            return jsonify({"error": "No records provided"}), 400

        df = pd.DataFrame(records)
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
        df = impute(df)

        row        = df[FEATURES].median()
        row_values = row.values.reshape(1, -1)

        pred  = rf_model.predict(row_values)[0]
        proba = rf_model.predict_proba(row_values)[0]
        label = le.inverse_transform([pred])[0]

        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        rule_exp = explain_label(row)

        return jsonify({
            "status":        label,
            "confidence":    round(float(max(proba)) * 100, 1),
            "probabilities": class_probs,
            "rows_used":     len(df),
            "explanation": {
                "score":   rule_exp["score"],
                "reasons": rule_exp["reasons"],
            }
        })

    except Exception as e:
        logging.error(f"/predict-raw error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Retrain Raw
# ─────────────────────────────────────────────

@app.route("/retrain-raw", methods=["POST"])
def retrain_raw():
    global rf_model, le
    try:
        records = request.json.get("records", [])
        if not records:
            return jsonify({"error": "No records provided"}), 400

        df = pd.DataFrame(records)
        df['date']   = pd.to_datetime(df['date'], errors='coerce')
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')

        label_counts = _train_core(df)

        return jsonify({
            "message":      "Model retrained successfully",
            "rows_used":    len(df),
            "classes":      list(le.classes_),
            "label_counts": label_counts,
            "model_ready":  True,
        })

    except Exception as e:
        logging.error(f"/retrain-raw error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Feature Importance Route
# ─────────────────────────────────────────────

@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        if rf_model is None:
            return jsonify({"error": "Model not trained yet. Call /retrain first."}), 400

        importances = rf_model.feature_importances_

        result = sorted(
            [
                {
                    "feature":          FEATURE_LABELS[feat],
                    "key":              feat,
                    "importance":       round(float(imp) * 100, 2),
                    "in_scoring_rules": feat in SCORING_RULE_FEATURES
                }
                for feat, imp in zip(FEATURES, importances)
            ],
            key=lambda x: x["importance"],
            reverse=True
        )

        non_rule_total = round(sum(
            item["importance"]
            for item in result
            if not item["in_scoring_rules"]
        ), 2)

        return jsonify({
            "feature_importance":         result,
            "non_scoring_rule_total_pct": non_rule_total,
            "note": (
                f"Features with in_scoring_rules=false (temperature, humidity, energy) "
                f"collectively contribute {non_rule_total}% to model predictions "
                f"despite not being used in label generation. "
                f"This may contribute to misclassifications in borderline cases."
            )
        })

    except Exception as e:
        logging.error(f"/feature-importance error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Misclassified Route
# ─────────────────────────────────────────────

@app.route("/misclassified", methods=["GET"])
def misclassified():
    try:
        if rf_model is None:
            return jsonify({"error": "Model not trained yet. Call /retrain first."}), 400

        today = datetime.today().strftime("%Y-%m-%d")
        with engine.connect() as conn:
            df = pd.read_sql(
                sqlalchemy.text("SELECT * FROM mes WHERE date <= :d ORDER BY date"),
                conn, params={"d": today}
            )

        if df.empty:
            return jsonify({"error": "No data found."}), 400

        df = impute(df)
        df['real_label'] = df.apply(auto_label, axis=1)

        # Use last 20% as test set (same split as training)
        split_idx = int(len(df) * 0.8)
        df_test   = df.iloc[split_idx:].copy()

        X_test      = df_test[FEATURES].values
        y_pred      = rf_model.predict(X_test)
        pred_labels = le.inverse_transform(y_pred)

        df_test = df_test.copy()
        df_test['predicted_label'] = pred_labels
        df_test['score']           = df_test.apply(auto_label_score, axis=1)

        # Filter only misclassified rows
        wrong = df_test[df_test['real_label'] != df_test['predicted_label']].copy()

        # Flag borderline cases (score exactly at boundary)
        wrong['is_borderline'] = wrong['score'].apply(
            lambda s: s in [2.0, 4.0, 2.5, 4.5]
        )

        # Select relevant columns for output
        output_cols = [
            'machineID', 'date', 'real_label', 'predicted_label',
            'score', 'is_borderline',
            'defects', 'scrapRate', 'downTimeHours',
            'reworkHours', 'qualityChecksFailed', 'maintenanceHours',
            'energyConsumption', 'averageTemperature', 'averageHumidityPercent'
        ]

        output_cols = [c for c in output_cols if c in wrong.columns]
        result = wrong[output_cols].copy()

        if 'date' in result.columns:
            result['date'] = result['date'].astype(str)

        borderline_count = int(wrong['is_borderline'].sum())
        total_wrong      = len(wrong)
        total_test       = len(df_test)

        return jsonify({
            "total_test_rows":       total_test,
            "total_misclassified":   total_wrong,
            "borderline_count":      borderline_count,
            "borderline_percentage": round(borderline_count / total_wrong * 100, 1) if total_wrong > 0 else 0,
            "summary": (
                f"{borderline_count} out of {total_wrong} misclassified rows "
                f"({round(borderline_count / total_wrong * 100, 1) if total_wrong > 0 else 0}%) "
                f"had scores exactly at the class boundary (2, 2.5, 4, or 4.5), "
                f"confirming that borderline threshold values are a primary cause of misclassification."
            ),
            "misclassified_rows": result.to_dict(orient='records')
        })

    except Exception as e:
        logging.error(f"/misclassified error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
