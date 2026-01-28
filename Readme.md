# Used Car Price Prediction – Tunisian Market (Automobile.tn)

**End-to-End MLOps Project**
Predicting fair market prices for used cars in Tunisia using real scraped data from automobile.tn

## 🎯 Problem Description (Business & ML Objective)

The Tunisian used car market lacks transparency:
- Sellers often overprice vehicles due to limited market visibility
- Buyers struggle to know if a listed price is fair
- No reliable, up-to-date tool exists that considers local factors (governorate, fuel type, mileage in km, year, etc.)

**This project solves**:
Build an accurate **used car price prediction model** tailored to the Tunisian market by:
1. Scraping current listings from [automobile.tn](https://www.automobile.tn/fr/occasion)
2. Storing & structuring data in a star schema (MySQL)
3. Training a regression model (LightGBM) to predict price in TND
4. Deploying the model as a REST API (Flask) for real-time predictions
5. Enabling future MLOps extensions (experiment tracking, monitoring, automated retraining)

**Target users**:
- Private buyers → check if a car is fairly priced
- Sellers → set competitive prices
- Car dealers / analysts → understand price drivers in the local market

**Success metric** (model performance):
- RMSE < 8 000–12 000 TND (depending on data volume)
- R² > 0.85–0.92 on cleaned data

## 🏗 Project Architecture (End-to-End Pipeline)
[Scraping] → car_listings.json
↓
[ETL → MySQL star schema] → car_data database
↓
[Data cleaning + Feature Engineering]
↓
[Model Training (LightGBM + GridSearchCV)] → car_price_model.joblib
↓
[FAISS semantic index (optional similarity search)]
↓
[Flask REST API] → /predict endpoint
↓

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- MySQL / MariaDB (running locally – default port 3307 in code)
- Google Chrome (required by Selenium)

### Installation

```bash
# 1. Clone or download the project
git https://github.com/BlackMoon681/tunisia-car-price-mlops
cd tunisia-car-price-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux / Mac
# or
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
Recommended minimum requirements.txt (add versions if possible for reproducibility):
textselenium
webdriver-manager
beautifulsoup4
mysql-connector-python
pandas
numpy
scikit-learn
lightgbm
sentence-transformers
faiss-cpu
joblib
shap
flask
flask-cors
tqdm
gunicorn
Run the full pipeline (local)


Bashcurl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "brand_name": "Volkswagen",
  "model_name": "Golf",
  "energy": "Essence",
  "gearbox": "Manuelle",
  "transmission": "Avant",
  "carrosserie": "Compacte",
  "year": 2018,
  "mileage_km": 95000
}'
Expected response example:
JSON{"predicted_price": 54870.25}
🔍 Model Performance (example from last run)

RMSE: ~9 800 TND
R²: 0.89
MAE: ~6 200 TND

(Actual numbers depend on the amount & freshness of scraped data)
Main price drivers (from SHAP analysis):
year ≈ mileage_km > brand_name > model_name > energy > gouvernorat
☁️ Deployment on Render.com

Push the project to GitHub
Go to https://render.com → New → Web Service
Connect your GitHub repo
Settings:
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
Instance Type: Free

Deploy → wait ~3–5 minutes → get your public URL

For maximum points (containerized deployment):
Add a simple Dockerfile and switch Render runtime to Docker.
📂 Project Structure
text.
├── scraper.py               # Scrapes automobile.tn listings
├── etl_to_mysql.py          # Loads JSON → MySQL star schema
├── build_faiss_index.py     # Builds FAISS index for similarity search
├── train_model.py           # Data prep, LightGBM training, SHAP
├── app.py                   # Flask API – /predict endpoint
├── car_price_model.joblib   # Trained model
├── car_listings_index.faiss # FAISS index (optional)
├── car_listings.json        # Raw scraped data
├── requirements.txt
├── README.md
└── .gitignore
🔧 Best Practices Already Implemented

Structured logging
Input validation & error handling in API
Human-like behavior + retries in scraper
Star schema in database
Hyperparameter tuning (GridSearchCV)
Model explainability (SHAP)
Easy to containerize

📈 Possible Improvements (for higher evaluation score)

Add MLflow or Weights & Biases for experiment tracking
Integrate Evidently AI for drift detection & monitoring
Automate pipeline with Prefect or Airflow
Add GitHub Actions CI/CD (lint, tests, deploy)
Create a simple front-end to interact with the API
Schedule periodic scraping & retraining

⚠️ Legal & Ethical Notes

This scraper is built for educational purposes only
Respect website terms of service, robots.txt, and rate limits
Do not use for commercial scraping or overload the target site

Tunis, Tunisia
January 2026
