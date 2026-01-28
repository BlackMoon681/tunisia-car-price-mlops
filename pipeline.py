# pipeline.py
from prefect import flow, task
import subprocess
import os
from datetime import datetime

# Paths – adjust if files are in subfolders
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SCRAPING_SCRIPT      = os.path.join(PROJECT_ROOT, "Scrapping.py")
DATAWAREHOUSE_SCRIPT = os.path.join(PROJECT_ROOT, "datawarehouse.py")
TRAIN_SCRIPT         = os.path.join(PROJECT_ROOT, "train_model.py")
INDEXER_SCRIPT       = os.path.join(PROJECT_ROOT, "car_indexer.py")
FLASK_CHATBOT        = os.path.join(PROJECT_ROOT, "chatbot.py")           # adjust name if different
ANNOUNCES            = os.path.join(PROJECT_ROOT, "announces.py")
PREDICTION_FLASK     = os.path.join(PROJECT_ROOT, "prediction_flask.py")

@task(name="Run Scraping")
def run_scraping():
    subprocess.run(["python", SCRAPING_SCRIPT], check=True)
    print("Scraping completed → car_listings.json updated")

@task(name="Run Data Warehouse / ETL")
def run_etl():
    subprocess.run(["python", DATAWAREHOUSE_SCRIPT], check=True)
    print("ETL to MySQL completed")

@task(name="Train Model")
def run_training():
    subprocess.run(["python", TRAIN_SCRIPT], check=True)
    print("Model training & MLflow registration completed")

@task(name="Build FAISS Index")
def run_indexer():
    subprocess.run(["python", INDEXER_SCRIPT], check=True)
    print("FAISS index built")

@task(name="Start Flask Services", tags=["deployment"])
def start_services():
    print("\nServices ready – start them manually in separate terminals:")
    print(f"   cd {PROJECT_ROOT}")
    print("   python prediction_flask.py")
    print("   python announces.py")
    print("   python chatbot.py   # or your main chatbot/flask file")
    # Note: We don't auto-start servers here to avoid blocking the flow
    # If you want background start, use subprocess.Popen (but handle termination carefully)

@flow(name="Tunisia Car Price Full Pipeline", log_prints=True)
def full_car_price_pipeline():
    """
    End-to-end pipeline:
    1. Scrape data
    2. ETL to MySQL
    3. Train model + register in MLflow
    4. Build semantic index
    5. Prepare Flask services
    """
    print(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_scraping()
    run_etl()
    run_training()
    run_indexer()
    start_services()

    print("\nPipeline completed successfully!")
    print("→ Check MLflow: run 'mlflow ui' in project folder")
    print("→ Check Prefect dashboard: http://127.0.0.1:4200 (if server running)")

if __name__ == "__main__":
    full_car_price_pipeline()