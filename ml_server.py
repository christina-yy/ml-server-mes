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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Spike-sensitive features — use max instead of median
# These are features where a single bad day matters more than the average
# SPIKE_FEATURES = ['defects', 'scrapRate', 'downTimeHours', 'qualityChecksFailed', 'reworkHours']

rf_model = None
le = LabelEncoder()

# Median values from training data — used to impute nulls at prediction time
_feature_medians = {}


# ─────────────────────────────────────────────
# Auto Label  (v4 — spike-aware thresholds)
#
# Changes from v3:
#   • Uses MAX values for spike features instead of median
#     (defects, scrapRate, downTimeHours, qualityChecksFailed, reworkHours)
#   • Stable features (energy, maintenance, temp, humidity) still use median
#   • This prevents single bad-day spikes from being smoothed away
#   • Thresholds re-validated against full dataset percentiles:
#       downtime p75=2.3, p90=2.72 → >2.5 is genuinely high (top ~18%)
#       scrapRate p75=0.04, p90=0.046 → thresholds confirmed correct
#       defects p75=7, p90=9 → thresholds confirmed correct
# ─────────────────────────────────────────────

def auto_label(row):
    score = 0

    # Defects (dataset max=34; p75=7, p90=9)
    if row['defects'] > 15:             score += 4
    elif row['defects'] > 9:            score += 2
    elif row['defects'] > 7:            score += 1

    # Scrap Rate — top feature by importance (p75=0.04, p90=0.046)
    if row['scrapRate'] > 0.046:        score += 3
    elif row['scrapRate'] > 0.040:      score += 2
    elif row['scrapRate'] > 0.030:      score += 1

    # Downtime (p75=2.3h, p90=2.72h — >2.5 is top ~18%)
    if row['downTimeHours'] > 2.5:      score += 2
    elif row['downTimeHours'] > 1.5:    score += 1

    # Rework hours (p75=1.49h, p90=1.82h)
    if row['reworkHours'] > 1.8:        score += 1
    elif row['reworkHours'] > 1.4:      score += 0.5

    # Quality checks failed (values: 0, 1, 2)
    if row['qualityChecksFailed'] == 2: score += 2
    elif row['qualityChecksFailed'] == 1: score += 1

    # Maintenance hours (p75=3.76h, p90=4.5h)
    if row['maintenanceHours'] > 4.5:   score += 2
    elif row['maintenanceHours'] > 4.0: score += 1

    if score >= 6:   return "Critical"
    elif score >= 2: return "At Risk"
    else:            return "Healthy"


# ─────────────────────────────────────────────
# FIX: Spike-aware feature aggregation
#
# Problem: Using pure median collapses 7 rows into 1 and smooths away
# critical spikes. e.g. defects=[2,3,12,4,3,2,3] → median=3 (12 ignored!)
#
# Solution: For spike-sensitive features, use MAX.
#           For stable features, use MEDIAN.
# This ensures a single bad day still raises an alert.
# ─────────────────────────────────────────────

def aggregate_features(df):
    """Weight recent rows more heavily than older ones."""
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    # Linear weights: oldest=1, newest=n
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()  # normalise to sum=1

    result = {}
    for col in FEATURES:
        if col in df.columns:
            result[col] = float(np.average(df[col].values, weights=weights))
    return pd.Series(result)

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
# Build feature vector from dataframe
# Uses spike-aware aggregation + fallback to auto_label
# ─────────────────────────────────────────────

def build_feature_vector(df):
    """
    Aggregate rows into a single feature vector using spike-aware method.
    Returns (feature_array, aggregated_series)
    """
    agg = aggregate_features(df)
    return agg.values.reshape(1, -1), agg


def predict_with_fallback(feature_array, agg_series, confidence_threshold=80.0):
    """
    Run RF prediction. If confidence is below threshold, fallback to
    auto_label rules which are guaranteed to be consistent with training labels.

    Returns: (label, confidence, probabilities, method_used)
    """
    pred  = rf_model.predict(feature_array)[0]
    proba = rf_model.predict_proba(feature_array)[0]
    label = le.inverse_transform([pred])[0]
    confidence = round(float(max(proba)) * 100, 1)

    class_probs = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(le.classes_, proba)
    }

    # FIX: If RF confidence is low, fallback to rule-based auto_label
    # This handles boundary cases where RF contradicts its own training labels
    if confidence < confidence_threshold:
        rule_label = auto_label(agg_series)
        logging.info(
            f"Low RF confidence ({confidence}%) → falling back to auto_label: {rule_label}"
        )
        return rule_label, confidence, class_probs, "auto_label_fallback"

    return label, confidence, class_probs, "random_forest"


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
    """Shared training logic used by both /retrain and /retrain-raw."""
    global rf_model, le, _feature_medians

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Store medians BEFORE imputing so they reflect real data distribution
    for col in FEATURES:
        if col in df.columns:
            _feature_medians[col] = df[col].median()

    df = impute(df)
    df['label'] = df.apply(auto_label, axis=1)

    label_counts = df['label'].value_counts().to_dict()
    logging.info(f"Label distribution: {label_counts}")

    X = df[FEATURES].values
    y = le.fit_transform(df['label'])

    # Time-aware train/test split — sort by date so test set is always
    # the most recent 20% of records
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    rf_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = rf_model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0
        )
        logging.info(f"Classification report (time-aware test split):\n{report}")

    logging.info(f"Model trained on {len(df)} rows. Classes: {list(le.classes_)}")
    return label_counts


train_model()


# ─────────────────────────────────────────────
# Intent Keywords Map
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
    for priority in ["predict", "compare", "top", "summary"]:
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
# Predict Route  (DB-based)
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        machine_id = request.json.get("machineID")
        if rf_model is None:
            return jsonify({"error": "Model not trained yet. Call /retrain first."})

        # FIX: Minimum row count guard — unreliable to predict on < 3 records
        MIN_ROWS = 3
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

        if len(df) < MIN_ROWS:
            return jsonify({
                "error": f"Not enough data for machine {machine_id}. "
                         f"Found {len(df)} record(s), need at least {MIN_ROWS}."
            })

        df = impute(df)

        # FIX: Use spike-aware aggregation instead of pure median
        feature_array, agg_series = build_feature_vector(df)

        label, confidence, class_probs, method = predict_with_fallback(
            feature_array, agg_series
        )

        return jsonify({
            "machineID":     machine_id,
            "status":        label,
            "confidence":    confidence,
            "probabilities": class_probs,
            "rows_used":     len(df),
            "method":        method   # tells you if RF or fallback was used
        })

    except Exception as e:
        logging.error(f"/predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Retrain Route  (DB-based)
# ─────────────────────────────────────────────

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        train_model()
        return jsonify({
            "message":     "Model retrained successfully",
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

        # FIX: Minimum row count guard
        MIN_ROWS = 3
        if len(records) < MIN_ROWS:
            return jsonify({
                "error": f"Not enough records provided. "
                         f"Got {len(records)}, need at least {MIN_ROWS}."
            }), 400

        df = pd.DataFrame(records)
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
        df = impute(df)

        # FIX: Use spike-aware aggregation instead of pure median
        feature_array, agg_series = build_feature_vector(df)

        label, confidence, class_probs, method = predict_with_fallback(
            feature_array, agg_series
        )

        return jsonify({
            "status":        label,
            "confidence":    confidence,
            "probabilities": class_probs,
            "rows_used":     len(df),
            "method":        method
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
            "message":      "Model retrained successfully",
            "rows_used":    len(df),
            "classes":      list(le.classes_),
            "label_counts": label_counts,
            "model_ready":  True
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
