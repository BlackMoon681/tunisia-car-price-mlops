import mysql.connector
import pandas as pd
import numpy as np
import logging
import joblib
import shap

from math import sqrt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logger.info("Connected to database")
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


def clean_data(df):
    df = df.dropna(subset=["price", "brand_name", "model_name", "year", "mileage_km"])
    cat_features = ["brand_name", "model_name", "energy", "gearbox", "transmission", "carrosserie"]
    for col in cat_features:
        df[col] = df[col].astype(str).str.lower().str.strip()

    df = df[(df['mileage_km'] >= 0) & (df['price'] > 5000) & (df['price'] <= 200000)]
    df = df[(df['year'] >= 1980) & (df['year'] <= 2025) & (df['mileage_km'] <= 500000)]
    df = df[~((df['mileage_km'] < 30000) & (df['price'] < 80000))]
    df = df[~((df['mileage_km'] > 100000) & (df['price'] > 90000))]

    Q1, Q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

    return df


def feature_engineering(df):
    df["age"] = 2025 - df["year"]
    df["mileage_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)
    df["log_price"] = np.log1p(df["price"])
    return df


def train_model(df, cat_features, num_features):
    X = df[cat_features + num_features]
    y = df["log_price"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMRegressor(random_state=42))
    ])
    param_grid = {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.01, 0.05],
        "model__num_leaves": [31, 64],
        "model__reg_alpha": [0.1, 10],
        "model__reg_lambda": [0.1, 10]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    logger.info(f"Best params: {grid.best_params_}")
    # Evaluate with actual prices
    y_pred = np.expm1(best_model.predict(X))
    y_true = df["price"]
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    logger.info(f"RMSE: {rmse:.2f} TND")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAE: {mae:.2f} TND")
    return best_model


def explain_model(model, df, cat_features, num_features):
    X = df[cat_features + num_features]
    explainer = shap.Explainer(model.named_steps["model"], model.named_steps["preprocessor"].transform(X))
    shap_values = explainer(model.named_steps["preprocessor"].transform(X))
    shap.summary_plot(shap_values, features=model.named_steps["preprocessor"].transform(X), feature_names=X.columns)


def save_model(model):
    joblib.dump(model, "car_price_model.joblib")
    logger.info("Model saved to car_price_model.joblib")


def main():
    conn = connect_to_database()
    try:
        df = fetch_data(conn)
        df = clean_data(df)
        df = feature_engineering(df)

        cat_features = ["brand_name", "model_name", "energy", "gearbox", "transmission", "carrosserie"]
        num_features = ["year", "mileage_km", "age", "mileage_per_year"]

        model = train_model(df, cat_features, num_features)
        save_model(model)
        explain_model(model, df, cat_features, num_features)

    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
