import json
import mysql.connector
from datetime import datetime

# Connect to MySQL
conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3307,
    user="root",
    password="",
)
cursor = conn.cursor()

# Create schema if it doesn't exist
cursor.execute("CREATE DATABASE IF NOT EXISTS car_data")
cursor.execute("USE car_data")

# Create tables if not exist
table_statements = [
    """
    CREATE TABLE IF NOT EXISTS DimBrand (
        brand_id INT AUTO_INCREMENT PRIMARY KEY,
        brand_name VARCHAR(100) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimModel (
        model_id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(100),
        brand_id INT,
        UNIQUE(model_name, brand_id),
        FOREIGN KEY (brand_id) REFERENCES DimBrand(brand_id)
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimLocation (
        location_id INT AUTO_INCREMENT PRIMARY KEY,
        governorate_name VARCHAR(100) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimDate (
        date_id INT AUTO_INCREMENT PRIMARY KEY,
        full_date DATE UNIQUE,
        year INT,
        month INT,
        day INT,
        week INT
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimEnergy (
        energy_id INT AUTO_INCREMENT PRIMARY KEY,
        type VARCHAR(50) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimGearbox (
        gearbox_id INT AUTO_INCREMENT PRIMARY KEY,
        type VARCHAR(50) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimTransmission (
        transmission_id INT AUTO_INCREMENT PRIMARY KEY,
        type VARCHAR(50) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimColor (
        color_id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimGeneration (
        generation_id INT AUTO_INCREMENT PRIMARY KEY,
        period VARCHAR(50) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS DimCarrosserie (
        car_body_id INT AUTO_INCREMENT PRIMARY KEY,
        type VARCHAR(50) UNIQUE
    )""",
    """
    CREATE TABLE IF NOT EXISTS FactVehicleSale (
        vehicle_id INT PRIMARY KEY,
        price INT,
        mileage_km INT,
        year INT,
        days_online INT,
        brand_id INT,
        model_id INT,
        date_id INT,
        location_id INT,
        energy_id INT,
        gearbox_id INT,
        transmission_id INT,
        color_ext_id INT,
        color_int_id INT,
        generation_id INT,
        car_body_id INT,
        FOREIGN KEY (brand_id) REFERENCES DimBrand(brand_id),
        FOREIGN KEY (model_id) REFERENCES DimModel(model_id),
        FOREIGN KEY (date_id) REFERENCES DimDate(date_id),
        FOREIGN KEY (location_id) REFERENCES DimLocation(location_id),
        FOREIGN KEY (energy_id) REFERENCES DimEnergy(energy_id),
        FOREIGN KEY (gearbox_id) REFERENCES DimGearbox(gearbox_id),
        FOREIGN KEY (transmission_id) REFERENCES DimTransmission(transmission_id),
        FOREIGN KEY (color_ext_id) REFERENCES DimColor(color_id),
        FOREIGN KEY (color_int_id) REFERENCES DimColor(color_id),
        FOREIGN KEY (generation_id) REFERENCES DimGeneration(generation_id),
        FOREIGN KEY (car_body_id) REFERENCES DimCarrosserie(car_body_id)
    )
    """
]

for stmt in table_statements:
    cursor.execute(stmt)

# Mapping table names to their ID columns
id_columns = {
    "DimBrand": "brand_id",
    "DimModel": "model_id",
    "DimLocation": "location_id",
    "DimDate": "date_id",
    "DimEnergy": "energy_id",
    "DimGearbox": "gearbox_id",
    "DimTransmission": "transmission_id",
    "DimColor": "color_id",
    "DimGeneration": "generation_id",
    "DimCarrosserie": "car_body_id",
}

def get_or_create_id(table, column, value):
    if value is None:
        return None
    id_col = id_columns.get(table)
    if not id_col:
        raise ValueError(f"Unknown table: {table}")
    cursor.execute(f"SELECT {id_col} FROM {table} WHERE {column} = %s", (value,))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute(f"INSERT INTO {table} ({column}) VALUES (%s)", (value,))
    conn.commit()
    return cursor.lastrowid

def get_or_create_date_id(date_str):
    if not date_str:
        return None
    date_obj = datetime.strptime(date_str, "%d.%m.%Y").date()
    cursor.execute("SELECT date_id FROM DimDate WHERE full_date = %s", (date_obj,))
    result = cursor.fetchone()
    if result:
        return result[0]
    year, month, day, week = date_obj.year, date_obj.month, date_obj.day, date_obj.isocalendar()[1]
    cursor.execute("""
        INSERT INTO DimDate (full_date, year, month, day, week)
        VALUES (%s, %s, %s, %s, %s)
    """, (date_obj, year, month, day, week))
    conn.commit()
    return cursor.lastrowid

with open("car_listings.json", encoding="utf-8") as f:
    listings = json.load(f)

for car in listings:
    try:
        vehicle_id = int(car.get("ID"))
        price = int(car.get("price", "0").replace(" ", ""))
        mileage_km = int(car.get("Kilométrage", "0").replace("km", "").replace(" ", ""))
        mise_en_circ = car.get("Mise en circulation", "0.0")
        year = int(mise_en_circ.split(".")[1]) if "." in mise_en_circ else 0

        brand_name = car.get("Marque")
        model_name = car.get("Modèle")

        brand_id = get_or_create_id("DimBrand", "brand_name", brand_name)
        model_id = None
        if model_name and brand_id:
            # Check if model exists for this brand
            cursor.execute("SELECT model_id FROM DimModel WHERE model_name = %s AND brand_id = %s", (model_name, brand_id))
            res = cursor.fetchone()
            if res:
                model_id = res[0]
            else:
                cursor.execute("INSERT INTO DimModel (model_name, brand_id) VALUES (%s, %s)", (model_name, brand_id))
                conn.commit()
                model_id = cursor.lastrowid

        location_id = get_or_create_id("DimLocation", "governorate_name", car.get("Gouvernorat"))
        energy_id = get_or_create_id("DimEnergy", "type", car.get("Énergie"))
        gearbox_id = get_or_create_id("DimGearbox", "type", car.get("Boite vitesse"))
        transmission_id = get_or_create_id("DimTransmission", "type", car.get("Transmission"))
        color_ext_id = get_or_create_id("DimColor", "name", car.get("Couleur extérieure"))
        color_int_id = get_or_create_id("DimColor", "name", car.get("Couleur intérieure"))
        generation_id = get_or_create_id("DimGeneration", "period", car.get("Génération"))
        car_body_id = get_or_create_id("DimCarrosserie", "type", car.get("Carrosserie"))
        date_id = get_or_create_date_id(car.get("Date de l'annonce"))

        cursor.execute("""
            INSERT IGNORE INTO FactVehicleSale (
                vehicle_id, price, mileage_km, year, days_online,
                brand_id, model_id, date_id, location_id,
                energy_id, gearbox_id, transmission_id,
                color_ext_id, color_int_id,
                generation_id, car_body_id
            ) VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            vehicle_id, price, mileage_km, year,
            brand_id, model_id, date_id, location_id,
            energy_id, gearbox_id, transmission_id,
            color_ext_id, color_int_id,
            generation_id, car_body_id
        ))
        conn.commit()
        print(f"✅ Inserted listing {vehicle_id}")

    except Exception as e:
        print(f"❌ Skipped listing {car.get('ID')}: {e}")

cursor.close()
conn.close()
