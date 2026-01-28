# Use slim Python image (smaller, faster)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy and install dependencies first (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Flask will run on
EXPOSE 8000

# Use gunicorn to serve the Flask app
# app:app means file=app.py, variable=app (change if your file is different)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "prediction_flask:app"]