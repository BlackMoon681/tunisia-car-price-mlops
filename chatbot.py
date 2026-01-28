import os
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import time
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import FAISS
try:
    import faiss
    logger.info("Imported faiss module")
except ImportError as e:
    logger.error(f"FAISS import failed: {e}. Please install faiss-cpu with: pip install faiss-cpu")
    raise

app = Flask(__name__)
CORS(app)

# Groq API Setup
MODEL = "qwen/qwen3-32b"

# Load FAISS index and metadata
INDEX_PATH = "car_listings_index.faiss"
METADATA_PATH = "car_listings_metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    descriptions = metadata['descriptions']
    metadata_records = metadata['data']
    logger.info(f"Loaded FAISS index with {len(descriptions)} listings")
except Exception as e:
    logger.error(f"Error loading FAISS index/metadata: {e}")
    raise

conversation_history = []

# User profile keys and default values
def get_default_user_profile():
    return {
        "brand": None,
        "min_year": None,
        "max_year": None,
        "fuel_type": None,
        "location": None,
        "max_price": None,
    }

last_results = []


def is_query_unclear(query):
    query = query.strip().lower()
    if len(query) < 5:
        return True
    unclear_phrases = [
        "hello", "hi", "hey", "how are you", "what's up", "help", "i need help",
        "i want a car", "show me cars", "any car"
    ]
    for phrase in unclear_phrases:
        if phrase in query:
            return True
    return False


def update_user_profile(query, user_profile):
    current_year = time.localtime().tm_year
    query_lower = query.lower()

    # Extract year filter
    match = re.search(r'not aged more than (\d+) years', query_lower)
    if match:
        max_age = int(match.group(1))
        user_profile["min_year"] = current_year - max_age

    # Extract brand
    brand_match = re.search(r'\b(toyota|opel|seat|volkswagen|renault)\b', query_lower)
    if brand_match:
        user_profile["brand"] = brand_match.group(1)

    # Extract fuel type
    fuel_match = re.search(r'\b(petrol|diesel|electric|hybrid|essence)\b', query_lower)
    if fuel_match:
        fuel_type = fuel_match.group(1)
        if fuel_type == "essence":
            fuel_type = "petrol"
        user_profile["fuel_type"] = fuel_type

    # Extract location
    location_match = re.search(r'in (\w+)', query_lower)
    if location_match:
        user_profile["location"] = location_match.group(1)

    # Extract max price
    price_match = re.search(r'(?:under|below|less than) (\d+(?:,\d+)?(?:\.\d+)?)', query_lower)
    if price_match:
        price_str = price_match.group(1).replace(',', '')
        try:
            user_profile["max_price"] = float(price_str)
        except ValueError:
            pass


def build_context_summary(user_profile):
    facts = []
    if user_profile["brand"]:
        facts.append(f"Brand: {user_profile['brand'].title()}.")
    if user_profile["min_year"]:
        facts.append(f"Year: from {user_profile['min_year']} and newer.")
    if user_profile["fuel_type"]:
        facts.append(f"Fuel Type: {user_profile['fuel_type'].title()}.")
    if user_profile["location"]:
        facts.append(f"Location: {user_profile['location'].title()}.")
    if user_profile["max_price"]:
        facts.append(f"Max Price: {user_profile['max_price']}.")
    return " ".join(facts)


def clean_response(response):
    try:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'\bI\s.*?\.', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\s+', ' ', response).strip()
        return response
    except Exception as e:
        logger.error(f"Error cleaning response: {e}")
        return response


def ask_llm(history):
    try:
        for attempt in range(3):
            try:
                chat_completion = client.chat.completions.create(
                    messages=history,
                    model=MODEL,
                    temperature=0.4
                )
                response = chat_completion.choices[0].message.content
                if not response:
                    raise ValueError("Empty response from API")
                return clean_response(response)
            except Exception as e:
                logger.error(f"Groq API attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(1)
        return "Sorry, I couldn't process the request due to an API error."
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return "Sorry, I couldn't process the request due to an API error."


def filter_by_profile(listings, user_profile):
    filtered = []
    for listing in listings:
        listing_lower = listing.lower()

        # Brand filter
        if user_profile['brand'] and user_profile['brand'].lower() not in listing_lower:
            logger.debug(f"Excluded by brand filter: {listing}")
            continue

        # Year filter
        if user_profile['min_year']:
            year_match = re.search(r'(19|20)\d{2}', listing)
            if year_match:
                year = int(year_match.group())
                if year < user_profile['min_year']:
                    logger.debug(f"Excluded by year filter: {listing}")
                    continue

        # Fuel type filter
        if user_profile['fuel_type']:
            if user_profile['fuel_type'].lower() not in listing_lower:
                logger.debug(f"Excluded by fuel filter: {listing}")
                continue

        # Max price filter
        if user_profile['max_price']:
            price_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)', listing.replace(' ', ''))
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                try:
                    price = float(price_str)
                    if price > user_profile['max_price']:
                        logger.debug(f"Excluded by price filter: {listing}")
                        continue
                except ValueError:
                    pass

        filtered.append(listing)
    return filtered


def retrieve_with_faiss(query, user_profile, k=10):
    global last_results
    try:
        query_embedding = model.encode([query], show_progress_bar=False).astype('float32')
        distances, indices = index.search(query_embedding, k)

        retrieved = []
        for i in indices[0]:
            if i == -1 or i >= len(descriptions):
                continue
            retrieved.append(descriptions[i])

        filtered = filter_by_profile(retrieved, user_profile)
        last_results = filtered[:5]
        logger.info(f"Retrieved {len(filtered)} listings after filtering for query: {query}")
        return last_results
    except Exception as e:
        logger.error(f"FAISS retrieval error: {e}")
        return []


def chatbot_with_rag(question):
    try:
        if is_query_unclear(question):
            return "Could you please specify more details? For example, the brand, year, or type of car you're looking for."

        # Reset user profile at each question
        user_profile = get_default_user_profile()
        update_user_profile(question, user_profile)

        retrieved_listings = retrieve_with_faiss(question, user_profile)
        context = build_context_summary(user_profile)

        if not retrieved_listings:
            return "I couldn’t find matching cars. Could you provide more details, like the brand, year, or fuel type?"

        prompt = f"""
{context}
User query: "{question}"

Here are some retrieved car listings:
{json.dumps(retrieved_listings, indent=2)}

Please list up to 5 matching car listings in this numbered format:
1. Model, Year, Price, Mileage, Location, Type, Fuel
If no relevant cars are found, say: "No matching cars found."
"""

        conversation_history.append({"role": "user", "content": question})

        response = ask_llm([
            {"role": "system", "content": "You are a friendly car assistant. Answer clearly using numbered lists only."},
            {"role": "user", "content": prompt}
        ])

        conversation_history.append({"role": "assistant", "content": response})

        prefix = random.choice([
            "Here’s what I found:",
            "Check these out:",
            "These might interest you:",
        ])

        return f"{prefix}\n{response}"

    except Exception as e:
        logger.error(f"RAG processing error: {e}")
        return "Sorry, I couldn't process the request due to an error."


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data["query"]
        response = chatbot_with_rag(query)
        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
