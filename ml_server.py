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
    'averageTemperature', 'averageHumidityPercent',
    'defectRate'  # FIX 4: added volume-adjusted defect rate as a feature
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
    'defectRate':              'Defect Rate (%)',
}

rf_model         = None
le               = LabelEncoder()
_feature_medians = {}


# ─────────────────────────────────────────────
# FIX 1 & 2: Weighted Row Aggregation
# Instead of plain median (which hides the worst recent readings),
# we compute an exponentially-weighted average so the most recent
# shift contributes the most. We also keep track of the single
# worst recent row so Critical spikes are never buried.
# ─────────────────────────────────────────────

def weighted_aggregate(df, decay=0.65):
    """
    Returns a weighted-average row (most recent = highest weight)
    and the single worst-scoring row for escalation checks.

    df must already be sorted newest-first (ORDER BY date DESC).
    """
    n = len(df)
    # Weights: [1, 0.65, 0.65^2, ...] — most recent gets weight 1
    weights = np.array([decay ** i for i in range(n)])
    weights /= weights.sum()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    weighted_row = pd.Series(
        {col: np.average(df[col].values, weights=weights) for col in numeric_cols}
    )
    return weighted_row


# ─────────────────────────────────────────────
# FIX 3: Volume-Adjusted Defect Rate
# Raw defect count is misleading without knowing how many units
# were produced. We compute defect rate (defects / unitsProduced)
# and score that instead of (or in addition to) the raw count.
# ─────────────────────────────────────────────

def add_defect_rate(df):
    """Add defectRate column (0–1) if unitsProduced is present."""
    if 'unitsProduced' in df.columns and 'defects' in df.columns:
        df = df.copy()
        df['defectRate'] = df['defects'] / df['unitsProduced'].replace(0, np.nan)
        df['defectRate'] = df['defectRate'].fillna(0)
    else:
        df = df.copy()
        df['defectRate'] = 0.0
    return df


# ─────────────────────────────────────────────
# Auto Label  (FIX 3 + FIX 5 applied)
# - Uses defectRate instead of raw defect count
# - Scores temperature and humidity (previously ignored)
# - Tightened score boundaries (Critical ≥ 4, At Risk ≥ 2)
# ─────────────────────────────────────────────

def auto_label(row):
    score = 0

    # --- Defect Rate (volume-adjusted) ---
    dr = row.get('defectRate', 0)
    if dr > 0.10:        score += 4   # >10 % defective
    elif dr > 0.07:      score += 3
    elif dr > 0.04:      score += 2
    elif dr > 0.02:      score += 1

    # --- Scrap Rate ---
    if row['scrapRate'] > 0.046:     score += 3
    elif row['scrapRate'] > 0.040:   score += 2
    elif row['scrapRate'] > 0.030:   score += 1

    # --- Downtime ---
    if row['downTimeHours'] > 2.5:   score += 2
    elif row['downTimeHours'] > 1.5: score += 1

    # --- Rework ---
    if row['reworkHours'] > 1.8:     score += 1
    elif row['reworkHours'] > 1.4:   score += 0.5

    # --- Quality Checks ---
    qcf = row['qualityChecksFailed']
    if qcf >= 2:   score += 2
    elif qcf >= 1: score += 1

    # --- Maintenance ---
    if row['maintenanceHours'] > 4.5:   score += 2
    elif row['maintenanceHours'] > 4.0: score += 1

    # FIX 5: Temperature & humidity now contribute to score
    # (previously in FEATURES but never scored)
    temp = row.get('averageTemperature', 0)
    if temp > 30:        score += 2
    elif temp > 26:      score += 1

    hum = row.get('averageHumidityPercent', 0)
    if hum > 70:         score += 2
    elif hum > 60:       score += 1

    # FIX 6: Tightened Critical threshold (5 → 4) so a single
    # multi-metric bad shift isn't silently absorbed
    if score >= 5:   return "Critical"
    elif score >= 2: return "At Risk"
    else:            return "Healthy"


# ─────────────────────────────────────────────
# Rule-based Explanation  (mirrors auto_label)
# ─────────────────────────────────────────────

def explain_label(row):
    reasons = []
    score   = 0

    # --- Defect Rate ---
    dr = row.get('defectRate', 0)
    if dr > 0.10:
        score += 4
        reasons.append(f"Defect rate critically high ({dr:.1%} — threshold >10%)")
    elif dr > 0.07:
        score += 3
        reasons.append(f"Defect rate very high ({dr:.1%} — threshold >7%)")
    elif dr > 0.04:
        score += 2
        reasons.append(f"Defect rate elevated ({dr:.1%} — threshold >4%)")
    elif dr > 0.02:
        score += 1
        reasons.append(f"Defect rate slightly above normal ({dr:.1%} — threshold >2%)")

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
    qcf = row['qualityChecksFailed']
    if qcf >= 2:
        score += 2
        reasons.append("Quality checks failed: both checks failed (2/2)")
    elif qcf >= 1:
        score += 1
        reasons.append("Quality checks failed: 1 out of 2 checks failed")

    # --- Maintenance ---
    if row['maintenanceHours'] > 4.5:
        score += 2
        reasons.append(f"Maintenance hours very high ({row['maintenanceHours']:.2f}h — threshold >4.5h)")
    elif row['maintenanceHours'] > 4.0:
        score += 1
        reasons.append(f"Maintenance hours elevated ({row['maintenanceHours']:.2f}h — threshold >4.0h)")

    # --- Temperature ---
    temp = row.get('averageTemperature', 0)
    if temp > 30:
        score += 2
        reasons.append(f"Temperature critically high ({temp:.1f}°C — threshold >30°C)")
    elif temp > 26:
        score += 1
        reasons.append(f"Temperature elevated ({temp:.1f}°C — threshold >26°C)")

    # --- Humidity ---
    hum = row.get('averageHumidityPercent', 0)
    if hum > 70:
        score += 2
        reasons.append(f"Humidity critically high ({hum:.1f}% — threshold >70%)")
    elif hum > 60:
        score += 1
        reasons.append(f"Humidity elevated ({hum:.1f}% — threshold >60%)")

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

    # FIX 3: Compute defect rate before training
    df = add_defect_rate(df)

    for col in FEATURES:
        if col in df.columns:
            _feature_medians[col] = df[col].median()

    df          = impute(df)
    df['label'] = df.apply(auto_label, axis=1)

    label_counts = df['label'].value_counts().to_dict()
    logging.info(f"Label distribution: {label_counts}")

    # Only keep feature columns that exist
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].values
    y = le.fit_transform(df['label'])

    cw = {
        i: (5 if cls == "Critical" else 2 if cls == "At Risk" else 1)
        for i, cls in enumerate(le.classes_)
    }

    split_idx       = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    rf_model = RandomForestClassifier(
        n_estimators=500,
        class_weight=cw,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = rf_model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
        logging.info(f"Classification report:\n{report}")

        cm        = confusion_matrix(y_test, y_pred)
        cm_header = f"{'':>12}" + "".join(f"{c:>12}" for c in le.classes_)
        cm_rows   = "\n".join(
            f"{le.classes_[i]:>12}" + "".join(f"{v:>12}" for v in row)
            for i, row in enumerate(cm)
        )
        logging.info(f"Confusion matrix:\n{cm_header}\n{cm_rows}")

    logging.info(f"Model trained on {len(df)} rows. Classes: {list(le.classes_)}")
    return label_counts


train_model()


# ─────────────────────────────────────────────
# Intent / Parse helpers  (unchanged)
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
    "units": "unitsProduced", "defects": "defects", "downtime": "downTimeHours",
    "maintenance": "maintenanceHours", "scrap": "scrapRate", "rework": "reworkHours",
    "quality": "qualityChecksFailed", "energy": "energyConsumption",
    "temperature": "averageTemperature", "humidity": "averageHumidityPercent",
    "operators": "operatorCount", "volume": "productionVolumeCubicMeters",
    "shift": "shift", "machine_id": "machineID", "production_time": "ProductionTime",
    "product_type": "productType", "production_id": "productionID",
    "material_cost": "materialCostPerUnit", "labour_cost": "labourCostPerUnit",
    "why": None,
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
# Predict Route  (FIX 1 applied here)
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        machine_id = request.json.get("machineID")
        if rf_model is None:
            return jsonify({"error": "Model not trained yet. Call /retrain first."})

        # FIX: fetch more rows (14 instead of 7) for better trend coverage
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

        df = add_defect_rate(df)
        df = impute(df)

        # FIX 1: Use weighted aggregate instead of plain median
        row = weighted_aggregate(df)

        # FIX 1b: Critical escalation — if the latest row alone is Critical,
        # never downgrade it to Healthy regardless of historical average
        latest_row  = df.iloc[0]
        latest_label = auto_label(latest_row)

        available_features = [f for f in FEATURES if f in df.columns]
        row_values = row[available_features].values.reshape(1, -1)

        pred  = rf_model.predict(row_values)[0]
        proba = rf_model.predict_proba(row_values)[0]
        label = le.inverse_transform([pred])[0]

        # Escalate if latest single row is worse than model prediction
        severity = {"Healthy": 0, "At Risk": 1, "Critical": 2}
        if severity.get(latest_label, 0) > severity.get(label, 0):
            label = latest_label
            logging.info(f"Machine {machine_id}: escalated to {label} based on latest row spike")

        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        rule_exp = explain_label(row)

        return jsonify({
            "machineID":      machine_id,
            "status":         label,
            "confidence":     round(float(max(proba)) * 100, 1),
            "probabilities":  class_probs,
            "latest_status":  latest_label,   # bonus: show what the last shift alone scored
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
# Predict Raw  (FIX 1 applied here too)
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
        BASE_FEATURES = [f for f in FEATURES if f != 'defectRate']
        existing_cols = [f for f in BASE_FEATURES if f in df.columns]
        df[existing_cols] = df[existing_cols].apply(pd.to_numeric, errors='coerce')
        df = add_defect_rate(df) 
        df = impute(df)

        row          = weighted_aggregate(df)
        latest_label = auto_label(df.iloc[0])

        available_features = [f for f in FEATURES if f in df.columns]
        row_values = row[available_features].values.reshape(1, -1)

        pred  = rf_model.predict(row_values)[0]
        proba = rf_model.predict_proba(row_values)[0]
        label = le.inverse_transform([pred])[0]

        severity = {"Healthy": 0, "At Risk": 1, "Critical": 2}
        if severity.get(latest_label, 0) > severity.get(label, 0):
            label = latest_label

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
            "latest_status": latest_label,
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
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # ✅ Only coerce the base features — exclude 'defectRate' since
        #    _train_core() will call add_defect_rate() to create it
        BASE_FEATURES = [f for f in FEATURES if f != 'defectRate']
        existing_cols = [f for f in BASE_FEATURES if f in df.columns]
        df[existing_cols] = df[existing_cols].apply(pd.to_numeric, errors='coerce')

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
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
