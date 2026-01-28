import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
import mysql.connector
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = 'uploads/'  # Make sure this folder exists and is writable
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# MySQL connection config
def get_db_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3307,
        user="root",
        password="",
        database="car_data"  # Replace with your DB name
    )

@app.route('/announce', methods=['POST'])
def announce():
    if 'announceImage' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['announceImage']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        ext = original_filename.rsplit('.', 1)[1].lower()
        # Generate a unique filename to prevent overwriting
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        return jsonify({'error': 'Invalid image file type'}), 400
    predicted_price = request.form.get('predicted_price')
    try:
        predicted_price_float = float(predicted_price) if predicted_price else None
    except ValueError:
        predicted_price_float = None

    # Get other form data
    brand_name = request.form.get('brand_name')
    model_name = request.form.get('model_name')
    energy = request.form.get('energy')
    gearbox = request.form.get('gearbox')
    transmission = request.form.get('transmission')
    carrosserie = request.form.get('carrosserie')
    year = request.form.get('year')
    mileage_km = request.form.get('mileage_km')
    announce_title = request.form.get('announceTitle')
    announce_description = request.form.get('announceDescription')

    # Basic validation
    required_fields = [brand_name, model_name, energy, gearbox, transmission,
                       carrosserie, year, mileage_km, announce_title, announce_description]
    if any(field is None or field.strip() == '' for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        mileage_km_int = int(mileage_km)
    except ValueError:
        return jsonify({'error': 'Mileage must be a number'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO announce
            (brand_name, model_name, energy, gearbox, transmission, carrosserie, year, mileage_km, announce_title, announce_description, image_filename, predicted_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (brand_name, model_name, energy, gearbox, transmission, carrosserie,
                             year, mileage_km_int, announce_title, announce_description,
                             filename, predicted_price_float))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'message': 'Annonce créée avec succès'})
    except mysql.connector.Error as err:
        return jsonify({'error': f'Database error: {err}'}), 500

@app.route('/announces', methods=['GET'])
def get_announces():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM announce ORDER BY created_at DESC")
        announces = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(announces)
    except mysql.connector.Error as err:
        return jsonify({'error': f'Database error: {err}'}), 500

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=False)
