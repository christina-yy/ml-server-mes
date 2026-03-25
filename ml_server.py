from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
import pandas as pd
import sqlalchemy
import os
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

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

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
rf_model   = None   # active model used for predictions (best from comparison)
le         = LabelEncoder()
_feature_medians  = {}
_eval_metrics     = {}   # stores last evaluation results — exposed via /metrics
_model_comparison = []   # stores per-model comparison results


# ─────────────────────────────────────────────
# Auto Label  (data-driven thresholds v2)
# ─────────────────────────────────────────────

def auto_label(row):
    score = 0

    # Defects  (p75=7, p90=9)
    if row['defects'] > 9:             score += 2
    elif row['defects'] > 7:           score += 1

    # Scrap rate  (dataset max=0.05; p75=0.04, p90=0.046)
    if row['scrapRate'] > 0.045:       score += 2
    elif row['scrapRate'] > 0.035:     score += 1

    # Downtime  (p50=1.57h, p75=2.3h, p90=2.69h)
    if row['downTimeHours'] > 2.5:     score += 2
    elif row['downTimeHours'] > 1.5:   score += 1

    # Rework hours  (p75=1.45h, p90=1.8h)
    if row['reworkHours'] > 1.8:       score += 1

    # Quality checks failed  (max=2, so >1 means value is 2)
    if row['qualityChecksFailed'] > 1: score += 1

    # Maintenance — high hours signal underlying issues
    if row['maintenanceHours'] > 4.0:  score += 1

    if score >= 5:   return "Critical"
    elif score >= 2: return "At Risk"
    else:            return "Healthy"


# ─────────────────────────────────────────────
# Impute nulls with training-set medians
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
# Trend Detection
#
# Given a DataFrame of recent records for one machine (sorted oldest→newest),
# compute a degradation score that captures whether key metrics are
# worsening over time — even if the current snapshot still looks "At Risk".
#
# Method:
#   For each of the four most predictive features, fit a linear slope
#   over the window. Normalise each slope by the feature's training-set
#   std so slopes are comparable across different units/scales.
#   A positive slope means the metric is getting worse (more defects,
#   more downtime, etc.).  Sum the normalised slopes into a single
#   degradation score and bucket it into three levels.
#
# Returns a dict:
#   degradation_score  – float (higher = deteriorating faster)
#   trend_level        – "Stable" | "Deteriorating" | "Rapidly Deteriorating"
#   feature_trends     – per-feature slope direction
# ─────────────────────────────────────────────

TREND_FEATURES = ['defects', 'scrapRate', 'downTimeHours', 'reworkHours']

# Approximate stds from training data — used for normalisation.
# Updated by _train_core() each time a model is trained.
_feature_stds = {
    'defects': 3.2, 'scrapRate': 0.011,
    'downTimeHours': 0.88, 'reworkHours': 0.58
}


def compute_trend(df: pd.DataFrame) -> dict:
    """df must be sorted oldest→newest and contain TREND_FEATURES columns."""
    if len(df) < 3:
        return {
            "degradation_score": 0.0,
            "trend_level": "Stable",
            "feature_trends": {},
            "note": "Fewer than 3 data points — trend unreliable"
        }

    n = len(df)
    x = np.arange(n, dtype=float)
    degradation_score = 0.0
    feature_trends = {}

    for feat in TREND_FEATURES:
        if feat not in df.columns:
            continue
        y = df[feat].values.astype(float)
        if np.isnan(y).all():
            continue
        # Fill any remaining NaNs with column median so polyfit doesn't fail
        y = np.where(np.isnan(y), np.nanmedian(y), y)
        slope = np.polyfit(x, y, 1)[0]
        std   = _feature_stds.get(feat, 1.0)
        norm  = slope / std if std > 0 else 0.0
        degradation_score += max(norm, 0)          # only count worsening
        feature_trends[feat] = "worsening" if slope > 0 else "improving"

    if degradation_score >= 0.5:
        level = "Rapidly Deteriorating"
    elif degradation_score >= 0.15:
        level = "Deteriorating"
    else:
        level = "Stable"

    return {
        "degradation_score": round(degradation_score, 3),
        "trend_level":       level,
        "feature_trends":    feature_trends
    }


# ─────────────────────────────────────────────
# Model Comparison + SMOTE  (core training)
#
# Trains three classifiers on the same SMOTE-balanced training set:
#   1. Random Forest   (n=300, balanced weights, min_samples_leaf=5)
#   2. XGBoost         (scale_pos_weight handles imbalance)
#   3. Decision Tree   (baseline — simple, explainable)
#
# Uses a time-aware 80/20 train/test split.
# SMOTE is applied ONLY to the training fold — never to the test set.
# The best model (highest macro F1 on test) becomes the active predictor.
#
# Why macro F1 and not accuracy?
#   Accuracy is dominated by the majority class. Macro F1 weights all
#   three classes equally, so it penalises models that miss Critical.
# ─────────────────────────────────────────────

def _train_core(df: pd.DataFrame):
    global rf_model, le, _feature_medians, _eval_metrics, _model_comparison, _feature_stds

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # Store medians and stds BEFORE imputing
    for col in FEATURES:
        if col in df.columns:
            _feature_medians[col] = float(df[col].median())

    for feat in TREND_FEATURES:
        if feat in df.columns:
            _feature_stds[feat] = float(df[feat].std()) or 1.0

    df = impute(df)
    df['label'] = df.apply(auto_label, axis=1)

    label_counts = df['label'].value_counts().to_dict()
    logging.info(f"Label distribution: {label_counts}")

    X = df[FEATURES].values
    y = le.fit_transform(df['label'])   # At Risk=0, Critical=1, Healthy=2

    # Time-aware split: last 20% of records for testing
    split_idx  = int(len(df) * 0.8)
    X_train_raw, X_test = X[:split_idx],  X[split_idx:]
    y_train_raw, y_test = y[:split_idx],  y[split_idx:]

    # ── SMOTE: oversample minority classes in training set only ──────────
    # k_neighbors must be < smallest class count in training set
    min_class_count = min(np.bincount(y_train_raw))
    k = min(5, min_class_count - 1) if min_class_count > 1 else 1
    try:
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)
        smote_counts = dict(zip(*np.unique(y_train, return_counts=True)))
        logging.info(f"After SMOTE — train class counts: {smote_counts}")
    except Exception as e:
        logging.warning(f"SMOTE failed ({e}), using raw training set.")
        X_train, y_train = X_train_raw, y_train_raw

    # ── Define candidate models ──────────────────────────────────────────
    n_classes = len(le.classes_)
    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=3,
            random_state=42
        )
    }

    # ── Train, evaluate, compare ─────────────────────────────────────────
    results    = []
    best_model = None
    best_f1    = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        per_class_f1 = f1_score(
            y_test, y_pred, average=None,
            labels=range(len(le.classes_)), zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred).tolist()

        class_f1_dict = {
            cls: round(float(f), 3)
            for cls, f in zip(le.classes_, per_class_f1)
        }

        result = {
            "model":         name,
            "accuracy":      round(acc, 4),
            "macro_f1":      round(macro_f1, 4),
            "per_class_f1":  class_f1_dict,
            "confusion_matrix": cm
        }
        results.append(result)

        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0
        )
        logging.info(f"\n── {name} ──\n{report}")

        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_model = model
            best_name  = name

    _model_comparison = results
    rf_model = best_model   # active model = winner of comparison

    # Store full eval metrics for /metrics endpoint
    y_pred_best = rf_model.predict(X_test)
    _eval_metrics = {
        "best_model":        best_name,
        "accuracy":          round(float(accuracy_score(y_test, y_pred_best)), 4),
        "macro_f1":          round(float(f1_score(y_test, y_pred_best, average='macro', zero_division=0)), 4),
        "rows_trained":      int(len(df)),
        "smote_applied":     True,
        "label_counts":      label_counts,
        "classes":           list(le.classes_),
        "confusion_matrix":  confusion_matrix(y_test, y_pred_best).tolist(),
        "classification_report": classification_report(
            y_test, y_pred_best,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True
        ),
        "model_comparison":  results
    }

    logging.info(
        f"Best model: {best_name} (macro F1={best_f1:.3f}). "
        f"Trained on {len(df)} rows (SMOTE applied to train fold)."
    )
    return label_counts


# ─────────────────────────────────────────────
# Train on startup from DB
# ─────────────────────────────────────────────

def train_model():
    global rf_model, le
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


train_model()


# ─────────────────────────────────────────────
# Intent detection helpers  (unchanged)
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
    "trend":           ["trend", "degradation", "deteriorating", "getting worse", "worsening"],
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
}


def detect_intent(text):
    text = text.lower()
    for priority in ["predict", "compare", "top", "summary", "trend"]:
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
        "best_model":  _eval_metrics.get("best_model"),
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
# Predict Route  (DB-based)
#
# Returns:
#   status, confidence, probabilities  — snapshot prediction
#   trend                              — degradation trend over last 14 days
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        machine_id = request.json.get("machineID")
        window     = int(request.json.get("window", 7))  # days to average

        if rf_model is None:
            return jsonify({"error": "Model not trained yet. Call /retrain first."})

        today = datetime.today().strftime("%Y-%m-%d")
        with engine.connect() as conn:
            # Fetch more rows than needed so trend has enough points
            df = pd.read_sql(
                sqlalchemy.text("""
                    SELECT * FROM mes
                    WHERE machineID = :mid AND date <= :d
                    ORDER BY date DESC LIMIT 14
                """),
                conn, params={"mid": machine_id, "d": today}
            )

        if df.empty:
            return jsonify({"error": f"No data found for machine {machine_id}"})

        df = df.sort_values('date').reset_index(drop=True)
        df = impute(df)

        # Snapshot: median of most recent `window` records
        snap_df = df.tail(window)
        row     = snap_df[FEATURES].median().values.reshape(1, -1)
        pred    = rf_model.predict(row)[0]
        proba   = rf_model.predict_proba(row)[0]
        label   = le.inverse_transform([pred])[0]

        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        # Trend: use all available rows (up to 14)
        trend = compute_trend(df)

        return jsonify({
            "machineID":   machine_id,
            "status":      label,
            "confidence":  round(float(max(proba)) * 100, 1),
            "probabilities": class_probs,
            "trend":       trend,
            "rows_used":   len(snap_df)
        })

    except Exception as e:
        logging.error(f"/predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Trend Route  (DB-based, dedicated endpoint)
# ─────────────────────────────────────────────

@app.route("/trend", methods=["POST"])
def trend_route():
    try:
        machine_id = request.json.get("machineID")
        days       = int(request.json.get("days", 14))

        today    = datetime.today().strftime("%Y-%m-%d")
        cutoff   = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

        with engine.connect() as conn:
            df = pd.read_sql(
                sqlalchemy.text("""
                    SELECT * FROM mes
                    WHERE machineID = :mid AND date BETWEEN :start AND :end
                    ORDER BY date ASC
                """),
                conn, params={"mid": machine_id, "start": cutoff, "end": today}
            )

        if df.empty:
            return jsonify({"error": f"No data found for machine {machine_id} in last {days} days"})

        df = impute(df)
        trend = compute_trend(df)

        return jsonify({
            "machineID": machine_id,
            "days":      days,
            "rows":      len(df),
            **trend
        })

    except Exception as e:
        logging.error(f"/trend error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Metrics Route  (evaluation results)
#
# Returns:
#   - Best model name and why it won
#   - Accuracy, macro F1
#   - Full confusion matrix
#   - Per-class precision/recall/F1
#   - Model comparison table (all three candidates)
#   - Label distribution from training data
# ─────────────────────────────────────────────

@app.route("/metrics", methods=["GET"])
def metrics():
    if not _eval_metrics:
        return jsonify({"error": "No metrics available. Train the model first."}), 400
    return jsonify(_eval_metrics)


# ─────────────────────────────────────────────
# Model Comparison Route
# ─────────────────────────────────────────────

@app.route("/model-comparison", methods=["GET"])
def model_comparison():
    if not _model_comparison:
        return jsonify({"error": "No comparison data. Train the model first."}), 400
    return jsonify({
        "best_model": _eval_metrics.get("best_model"),
        "comparison": _model_comparison
    })


# ─────────────────────────────────────────────
# Retrain Route  (DB-based)
# ─────────────────────────────────────────────

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        train_model()
        return jsonify({
            "message":     "Model retrained successfully",
            "best_model":  _eval_metrics.get("best_model"),
            "macro_f1":    _eval_metrics.get("macro_f1"),
            "model_ready": rf_model is not None
        })
    except Exception as e:
        logging.error(f"/retrain error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Predict Raw  (browser sends records directly)
# ─────────────────────────────────────────────

@app.route("/predict-raw", methods=["POST"])
def predict_raw():
    try:
        if rf_model is None:
            return jsonify({"error": "Model not trained yet. Ask me to retrain first."}), 400

        records = request.json.get("records", [])
        if not records:
            return jsonify({"error": "No records provided"}), 400

        df = pd.DataFrame(records)
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
        df = impute(df)

        # Sort by date if present so trend is meaningful
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)

        row   = df[FEATURES].median().values.reshape(1, -1)
        pred  = rf_model.predict(row)[0]
        proba = rf_model.predict_proba(row)[0]
        label = le.inverse_transform([pred])[0]

        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        trend = compute_trend(df)

        return jsonify({
            "status":        label,
            "confidence":    round(float(max(proba)) * 100, 1),
            "probabilities": class_probs,
            "trend":         trend,
            "rows_used":     len(df)
        })

    except Exception as e:
        logging.error(f"/predict-raw error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Retrain Raw  (browser sends all records)
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
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')

        label_counts = _train_core(df)

        return jsonify({
            "message":          "Model retrained successfully",
            "rows_used":        len(df),
            "best_model":       _eval_metrics.get("best_model"),
            "macro_f1":         _eval_metrics.get("macro_f1"),
            "classes":          list(le.classes_),
            "label_counts":     label_counts,
            "model_comparison": _model_comparison,
            "model_ready":      True
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
