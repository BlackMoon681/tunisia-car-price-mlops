# prediction_flask.py
# Fixed version: Properly handle Evidently import, use correct modules, add fallbacks.

import logging
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import json
from pathlib import Path
import sys

# Configure logging FIRST (use module logger, avoid root)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensure output to console
)
logger = logging.getLogger(__name__)  # Use named logger

app = Flask(__name__)
CORS(app)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
MODEL_PATH = "car_price_model.joblib"
REFERENCE_PATH = "reference_data.csv"
BUFFER_PATH = "prediction_buffer.jsonl"
BUFFER_SIZE = 50
CURRENT_YEAR = datetime.now().year

# ────────────────────────────────────────────────
# Evidently Import Handling
# ────────────────────────────────────────────────
EVIDENTLY_AVAILABLE = False
ColumnMapping = None
Report = None
DataDriftPreset = None

try:
    # Modern Evidently (v0.4+): imports from evidently.ui or direct
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping  # This should work if installed correctly
    EVIDENTLY_AVAILABLE = True
    logger.info("Evidently imported successfully. Monitoring enabled.")
except ImportError as e:
    logger.warning(f"Evidently import failed: {e}. "
                   "Install with: pip install evidently")
    logger.warning("Monitoring and drift detection disabled.")
    # Optionally add fallback or dummy classes if needed
    class DummyReport:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass
        def save_html(self, path): pass
        def json(self): return '{}'
    Report = DummyReport
    ColumnMapping = lambda **kwargs: None
    DataDriftPreset = lambda **kwargs: None

# Only setup if available
if EVIDENTLY_AVAILABLE:
    column_mapping = ColumnMapping(
        numerical_features=["year", "mileage_km", "age", "mileage_per_year"],
        categorical_features=[
            "brand_name", "model_name", "energy",
            "gearbox", "transmission", "carrosserie"
        ],
        prediction="prediction"
    )

# ────────────────────────────────────────────────
# Global model
# ────────────────────────────────────────────────
model = None

def load_model():
    global model
    if model is not None:
        return model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded: {MODEL_PATH}")
        return model
    except FileNotFoundError:
        logger.critical(f"Model file missing: {MODEL_PATH}")
        raise
    except Exception as e:
        logger.error(f"Model load failed: {e}", exc_info=True)
        raise

# Load at startup
try:
    load_model()
except Exception:
    logger.critical("Startup failed – exiting.")
    sys.exit(1)

def save_buffer(buffer: list):
    if not buffer:
        return
    try:
        with open(BUFFER_PATH, "a", encoding="utf-8") as f:
            for row in buffer:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        logger.debug(f"Saved {len(buffer)} rows to buffer.")
    except Exception as e:
        logger.error(f"Buffer save failed: {e}")

def load_buffer() -> list:
    if not os.path.exists(BUFFER_PATH):
        return []
    try:
        with open(BUFFER_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Buffer load failed: {e}")
        return []

# In-memory buffer (load persistent on first use)
CURRENT_BUFFER = []

@app.route('/predict', methods=['POST'])
def predict():
    global CURRENT_BUFFER
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON"}), 400

        required = {
            "brand_name", "model_name", "energy", "gearbox",
            "transmission", "carrosserie", "year", "mileage_km"
        }
        missing = required - data.keys()
        if missing:
            return jsonify({"error": f"Missing: {', '.join(missing)}"}), 400

        # Preprocess
        df = pd.DataFrame([data])
        cat_cols = ["brand_name", "model_name", "energy", "gearbox", "transmission", "carrosserie"]
        for col in cat_cols:
            df[col] = df[col].astype(str).str.lower().str.strip()

        df["year"] = pd.to_numeric(df["year"])
        df["mileage_km"] = pd.to_numeric(df["mileage_km"])

        y = df["year"].iloc[0]
        if not (1980 <= y <= CURRENT_YEAR + 1):
            raise ValueError(f"Invalid year: {y} (must be 1980–{CURRENT_YEAR+1})")

        if df["mileage_km"].iloc[0] < 0:
            raise ValueError("Mileage cannot be negative")

        df["age"] = CURRENT_YEAR - df["year"]
        df["mileage_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)

        features = cat_cols + ["year", "mileage_km", "age", "mileage_per_year"]
        X = df[features]

        # Predict
        log_price = model.predict(X)[0]
        import numpy as np

        # Ensure log_price is converted to safe NumPy float
        log_price = np.float64(log_price)
        price = float(np.expm1(log_price))

        # Monitoring
        if EVIDENTLY_AVAILABLE:
            row = data.copy()
            row["prediction"] = price
            row["timestamp"] = datetime.now().isoformat()

            if not CURRENT_BUFFER:  # Lazy load
                CURRENT_BUFFER = load_buffer()

            CURRENT_BUFFER.append(row)

            if len(CURRENT_BUFFER) >= BUFFER_SIZE:
                logger.info(f"Buffer full ({len(CURRENT_BUFFER)}) – drift check")
                check_drift()
                save_buffer(CURRENT_BUFFER)
                CURRENT_BUFFER.clear()

        return jsonify({"predicted_price": round(price, 2)})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error("Prediction error", exc_info=True)
        return jsonify({"error": "Server error"}), 500

def check_drift():
    if not EVIDENTLY_AVAILABLE or not os.path.exists(REFERENCE_PATH):
        return

    try:
        ref = pd.read_csv(REFERENCE_PATH)
        curr = pd.DataFrame(CURRENT_BUFFER)
        common_cols = ref.columns.intersection(curr.columns)
        ref = ref[common_cols]
        curr = curr[common_cols]

        # Type safety
        num_feats = [f for f in column_mapping.numerical_features if f in curr.columns]
        for col in num_feats:
            curr[col] = pd.to_numeric(curr[col], errors='coerce')

        report = Report(metrics=[DataDriftPreset(stattest_threshold=0.05)])

        report.run(reference_data=ref, current_data=curr, column_mapping=column_mapping)

        result = json.loads(report.json())
        drift_data = result.get("metrics", [{}])[0].get("result", {})
        drifted = drift_data.get("dataset_drift", False)
        share = drift_data.get("share_of_drifted_features", 0.0)

        logger.info(f"Drift: detected={drifted}, share={share:.1%}")

        if drifted or share > 0.3:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report.save_html(f"drift_{ts}.html")
            logger.warning(f"Drift detected! Report saved. Share: {share:.1%}")

    except Exception as e:
        logger.error("Drift check error", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting Flask API...")
    # Use waitress or gunicorn in prod, but for dev:
    app.run(host="0.0.0.0", port=5000, debug=False)