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

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# Database — reads DB_URL from Render env var
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


# ─────────────────────────────────────────────
# Auto Label
# ─────────────────────────────────────────────

def auto_label(row):
    score = 0
    if row['defects'] > 15:            score += 2
    elif row['defects'] > 7:           score += 1
    if row['scrapRate'] > 0.07:        score += 2
    elif row['scrapRate'] > 0.035:     score += 1
    if row['downTimeHours'] > 2.5:     score += 2
    elif row['downTimeHours'] > 1:     score += 1
    if row['reworkHours'] > 1.5:       score += 1
    if row['qualityChecksFailed'] > 2: score += 1
    if score >= 4:   return "Critical"
    elif score >= 1: return "At Risk"
    else:            return "Healthy"


# ─────────────────────────────────────────────
# Train Model (from DB — used on startup)
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

        df['date']   = pd.to_datetime(df['date'])
        df[FEATURES] = df[FEATURES].fillna(0)
        df['label']  = df.apply(auto_label, axis=1)

        X = df[FEATURES].values
        y = le.fit_transform(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        rf_model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        )
        rf_model.fit(X_train, y_train)
        logging.info(f"Model trained on {len(df)} rows. Classes: {list(le.classes_)}")

    except Exception as e:
        logging.error(f"train_model() failed: {e}")


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
# Predict Route (original — uses DB directly)
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

        df[FEATURES] = df[FEATURES].fillna(0)
        row   = df[FEATURES].mean().values.reshape(1, -1)
        pred  = rf_model.predict(row)[0]
        prob  = rf_model.predict_proba(row)[0]
        label = le.inverse_transform([pred])[0]

        return jsonify({
            "machineID":  machine_id,
            "status":     label,
            "confidence": round(float(max(prob)) * 100, 1)
        })

    except Exception as e:
        logging.error(f"/predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Retrain Route (original — uses DB directly)
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
# Predict Raw — browser sends records directly
# No DB call needed, works with InfinityFree
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
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)

        row   = df[FEATURES].mean().values.reshape(1, -1)
        pred  = rf_model.predict(row)[0]
        prob  = rf_model.predict_proba(row)[0]
        label = le.inverse_transform([pred])[0]

        return jsonify({
            "status":     label,
            "confidence": round(float(max(prob)) * 100, 1),
            "rows_used":  len(df)
        })

    except Exception as e:
        logging.error(f"/predict-raw error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Retrain Raw — browser sends all records
# No DB call needed, works with InfinityFree
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
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        df['label']  = df.apply(auto_label, axis=1)

        X = df[FEATURES].values
        y = le.fit_transform(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        rf_model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        )
        rf_model.fit(X_train, y_train)

        label_counts = df['label'].value_counts().to_dict()

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
