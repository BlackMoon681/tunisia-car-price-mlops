import pandas as pd
import numpy as np
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def load_model():
    """Load the trained LightGBM model from disk."""
    try:
        model = joblib.load("car_price_model.joblib")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_input(data):
    """Preprocess input data to match training pipeline."""
    try:
        # Create DataFrame from input JSON
        df = pd.DataFrame([data])

        # Standardize categorical values
        cat_features = ["brand_name", "model_name", "energy", "gearbox", "transmission", "carrosserie"]
        for col in cat_features:
            df[col] = df[col].astype(str).str.lower().str.strip()

        # Convert year and mileage_km to numeric
        try:
            df["year"] = pd.to_numeric(df["year"], errors='raise')
            df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors='raise')
        except Exception as e:
            raise ValueError(f"Invalid numeric value for year or mileage_km: {e}")

        # Validate ranges
        if not (1980 <= df["year"].iloc[0] <= 2025):
            raise ValueError("Year must be between 1980 and 2025")
        if df["mileage_km"].iloc[0] < 0:
            raise ValueError("Mileage_km must be non-negative")

        # Feature engineering
        df["age"] = 2025 - df["year"]
        df["mileage_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)

        # Select features in the correct order
        features = cat_features + ["year", "mileage_km", "age", "mileage_per_year"]
        return df[features]
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict car price."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["brand_name", "model_name", "energy", "gearbox",
                           "transmission", "carrosserie", "year", "mileage_km"]
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Missing field: {field}"}), 400
        # Preprocess input
        input_df = preprocess_input(data)
        # Load model
        model = load_model()
        # Predict and transform back from log_price
        prediction = np.expm1(model.predict(input_df)[0])

        return jsonify({"predicted_price": round(prediction, 2)})
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)