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

rf_model = None
le = LabelEncoder()

# Median values from training data — used to impute nulls at prediction time
# Populated during train_model(); avoids filling unknown rows with 0
_feature_medians = {}


# ─────────────────────────────────────────────
# Auto Label  (v2 — data-driven thresholds)
#
# Changes from v1:
#   • ScrapRate thresholds lowered to 0.045 / 0.035
#     (old 0.07 was above the dataset max of 0.05 — never fired)
#   • Defects Critical threshold lowered to >9 (p90) from >15
#   • Downtime At Risk raised to >1.5h (p50) from >1h
#   • qualityChecksFailed threshold lowered to >1 from >2
#     (max value in dataset is 2, so >2 never fired)
#   • Added maintenanceHours >4.0 as a contributing signal
#   • Critical score threshold raised to >=5 so the label is
#     genuinely harder to reach and less noisy
# ─────────────────────────────────────────────

def auto_label(row):
    score = 0

    # Defects  (p75 = 7, p90 = 9)
    if row['defects'] > 9:           score += 2
    elif row['defects'] > 7:         score += 1

    # Scrap rate  (dataset max = 0.05; p75 = 0.04, p90 = 0.046)
    if row['scrapRate'] > 0.045:     score += 2
    elif row['scrapRate'] > 0.035:   score += 1

    # Downtime  (p50 = 1.57h, p75 = 2.3h, p90 = 2.69h)
    if row['downTimeHours'] > 2.5:   score += 2
    elif row['downTimeHours'] > 1.5: score += 1

    # Rework hours  (p75 = 1.45h, p90 = 1.8h)
    if row['reworkHours'] > 1.8:     score += 1

    # Quality checks failed  (max = 2, so >1 means value is 2)
    if row['qualityChecksFailed'] > 1: score += 1

    # Maintenance hours — high maintenance indicates underlying issues
    if row['maintenanceHours'] > 4.0: score += 1

    if score >= 5:   return "Critical"
    elif score >= 2: return "At Risk"
    else:            return "Healthy"


# ─────────────────────────────────────────────
# Impute nulls with training-set medians
# (filling with 0 incorrectly makes missing rows look Healthy)
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
    # the most recent 20% of records. A random split leaks future data
    # into training, which inflates accuracy on sequential data.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    rf_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=5,   # prevents overfitting on the tiny Critical class
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Log a proper per-class evaluation report
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
# Intent Keywords Map  (unchanged)
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

        df = impute(df)

        # Use median instead of mean — more robust to outlier days
        row = df[FEATURES].median().values.reshape(1, -1)
        pred  = rf_model.predict(row)[0]
        proba = rf_model.predict_proba(row)[0]
        label = le.inverse_transform([pred])[0]

        # Return all class probabilities so the frontend can show a breakdown
        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        return jsonify({
            "machineID":   machine_id,
            "status":      label,
            "confidence":  round(float(max(proba)) * 100, 1),
            "probabilities": class_probs
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

        df = pd.DataFrame(records)
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
        df = impute(df)

        # Use median of the window — more robust to outlier days than mean
        row   = df[FEATURES].median().values.reshape(1, -1)
        pred  = rf_model.predict(row)[0]
        proba = rf_model.predict_proba(row)[0]
        label = le.inverse_transform([pred])[0]

        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }

        return jsonify({
            "status":        label,
            "confidence":    round(float(max(proba)) * 100, 1),
            "probabilities": class_probs,
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
