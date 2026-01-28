import os
import logging
from datetime import datetime

import mysql.connector
import pandas as pd
import numpy as np
import joblib
import shap
from math import sqrt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving plots

# ────────────────────────────────────────────────
# MLflow
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature

# ────────────────────────────────────────────────
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            port=3307,
            user="root",
            password="",
            database="car_data"
        )
        logger.info("Database connection established")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed: {err}")
        raise


def fetch_data(conn):
    query = """
    SELECT
        f.price,
        b.brand_name,
        m.model_name,
        f.year,
        f.mileage_km,
        e.type AS energy,
        g.type AS gearbox,
        t.type AS transmission,
        c.type AS carrosserie
    FROM FactVehicleSale f
    LEFT JOIN DimBrand b ON f.brand_id = b.brand_id
    LEFT JOIN DimModel m ON f.model_id = m.model_id
    LEFT JOIN DimEnergy e ON f.energy_id = e.energy_id
    LEFT JOIN DimGearbox g ON f.gearbox_id = g.gearbox_id
    LEFT JOIN DimTransmission t ON f.transmission_id = t.transmission_id
    LEFT JOIN DimCarrosserie c ON f.car_body_id = c.car_body_id
    WHERE f.price IS NOT NULL AND f.price > 0
    """
    return pd.read_sql(query, conn)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price", "brand_name", "model_name", "year", "mileage_km"])

    cat_features = ["brand_name", "model_name", "energy", "gearbox", "transmission", "carrosserie"]
    for col in cat_features:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Outlier & range filters
    df = df[
        (df['mileage_km'] >= 0) &
        (df['price'] > 5000) & (df['price'] <= 200000) &
        (df['year'] >= 1980) & (df['year'] <= 2025) &
        (df['mileage_km'] <= 500000)
        ]
    df = df[~((df['mileage_km'] < 30000) & (df['price'] < 80000))]
    df = df[~((df['mileage_km'] > 100000) & (df['price'] > 90000))]

    # IQR-based outlier removal on price
    Q1, Q3 = df['price'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = 2025 - df["year"]
    df["mileage_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)
    df["log_price"] = np.log1p(df["price"])
    return df


def train_model_with_gridsearch(X, y, cat_features, num_features):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMRegressor(random_state=42, verbosity=-1))
    ])

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.01, 0.05],
        "model__num_leaves": [31, 64],
        "model__reg_alpha": [0.1, 10],
        "model__reg_lambda": [0.1, 10]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    logger.info(f"Best hyperparameters: {grid.best_params_}")

    # Evaluate (on training data – note: in production use hold-out/test set)
    y_pred_log = best_model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y)

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    logger.info(f"RMSE: {rmse:,.0f} TND")
    logger.info(f"R²:   {r2:.4f}")
    logger.info(f"MAE:  {mae:,.0f} TND")

    return best_model, rmse, r2, mae, grid.best_params_


def explain_model(model, X, feature_names, enable_shap: bool = True):
    if not enable_shap:
        logger.info("SHAP explanation skipped (disabled)")
        return False

    try:
        explainer = shap.Explainer(
            model.named_steps["model"],
            model.named_steps["preprocessor"].transform(X)
        )
        shap_values = explainer(model.named_steps["preprocessor"].transform(X))

        shap.summary_plot(
            shap_values,
            features=model.named_steps["preprocessor"].transform(X),
            feature_names=feature_names,
            show=False
        )

        import matplotlib.pyplot as plt
        plt.savefig("shap_summary.png", bbox_inches='tight', dpi=150)
        plt.close()
        logger.info("SHAP summary plot saved → shap_summary.png")
        return True
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {str(e)}")
        return False


def main():
    conn = connect_to_database()
    try:
        df = fetch_data(conn)
        df = clean_data(df)
        df = feature_engineering(df)

        cat_features = ["brand_name", "model_name", "energy", "gearbox", "transmission", "carrosserie"]
        num_features = ["year", "mileage_km", "age", "mileage_per_year"]
        df_reference = df[cat_features + num_features + ["log_price"]].copy()  # features + target
        df_reference.to_csv("reference_data.csv", index=False)
        X = df[cat_features + num_features]
        y = df["log_price"]

        # ────────────────────────────────────────────────
        # MLflow configuration – local file storage
        tracking_uri = "file:///mlruns"
        mlflow.set_tracking_uri(tracking_uri)

        # Show and ensure storage directory exists
        storage_path = tracking_uri.replace("file://", "").lstrip("/")
        abs_storage_path = os.path.abspath(storage_path)
        logger.info(f"MLflow tracking & registry path: {abs_storage_path}")

        os.makedirs(abs_storage_path, exist_ok=True)
        if not os.path.exists(abs_storage_path):
            raise RuntimeError(f"Failed to create MLflow storage directory: {abs_storage_path}")

        # Set experiment
        mlflow.set_experiment("Tunisia-Car-Price-Prediction")

        run_name = f"LightGBM-GridSearch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Started MLflow run: {run.info.run_id} | name: {run_name}")

            mlflow.log_param("num_samples", len(df))
            mlflow.log_param("num_features", len(cat_features) + len(num_features))
            mlflow.log_param("python_version", f"{sys.version_info.major}.{sys.version_info.minor}")

            # Train & evaluate
            best_model, rmse, r2, mae, best_params = train_model_with_gridsearch(
                X, y, cat_features, num_features
            )

            # ─── Logging to MLflow ───────────────────────────────────────
            mlflow.log_params(best_params)
            mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

            mlflow.set_tag("model_type", "LightGBM + Pipeline")
            mlflow.set_tag("dataset_source", "automobile.tn")
            mlflow.set_tag("target", "log_price")
            mlflow.set_tag("run_source", "train_model.py")

            # Model signature & example
            X_sample = X.iloc[:10]
            y_sample_pred = best_model.predict(X_sample)
            signature = infer_signature(X_sample, y_sample_pred)

            # Log model + register
            mlflow.lightgbm.log_model(
                lgb_model=best_model.named_steps["model"],
                artifact_path="lightgbm-model",
                registered_model_name="TunisiaCarPriceModel",
                signature=signature,
                input_example=X_sample.to_dict(orient="records")[0]
            )

            # SHAP explanation
            feature_names = cat_features + num_features
            if explain_model(best_model, X, feature_names, enable_shap=True):
                mlflow.log_artifact("shap_summary.png", artifact_path="shap")

            logger.info(f"Model registered → TunisiaCarPriceModel (version created)")
            logger.info(f"Run ID: {run.info.run_id}")

            # Local backup
            joblib.dump(best_model, "car_price_model.joblib")
            logger.info("Local model saved → car_price_model.joblib")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    import sys

    main()