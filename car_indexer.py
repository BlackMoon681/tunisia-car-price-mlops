import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import pickle
import re
from tqdm import tqdm

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

# Try to access IndexFlatL2 to verify FAISS setup
try:
    faiss.IndexFlatL2(128)  # Test import
except AttributeError:
    logger.warning("Standard faiss.IndexFlatL2 not available, trying faiss.swigfaiss")
    try:
        from faiss import swigfaiss as faiss
        logger.info("Successfully imported faiss.swigfaiss")
    except ImportError as e:
        logger.error(f"FAISS swigfaiss import failed: {e}. Ensure faiss-cpu is properly installed.")
        raise

def load_car_listings(json_path):
    """Load car listings from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} car listings from {json_path}")

        # Preprocess fields
        df['price'] = df['price'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
        df['mileage_km'] = df['Kilométrage'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
        df['year'] = df['Mise en circulation'].apply(
            lambda x: int(x.split('.')[-1]) if isinstance(x, str) and '.' in x else int(x))
        df = df.rename(columns={
            'Marque': 'brand_name',
            'Modèle': 'model_name',
            'Gouvernorat': 'governorate_name',
            'Énergie': 'energy',
            'Carrosserie': 'carrosserie'
        })
        return df
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise

def create_text_descriptions(df):
    """Create text descriptions for each car listing."""
    try:
        descriptions = []
        for _, row in df.iterrows():
            desc = (f"{row['brand_name']} {row['model_name']}, {row['year']}, "
                    f"{row['price']} TND, {row['mileage_km']} km, {row['governorate_name']}, "
                    f"{row['carrosserie']}, {row['energy']}")
            descriptions.append(desc)
        logger.info(f"Created {len(descriptions)} text descriptions")
        return descriptions
    except Exception as e:
        logger.error(f"Error creating descriptions: {e}")
        raise

def build_faiss_index(descriptions, model_name='all-MiniLM-L6-v2', batch_size=128):
    """Build FAISS index from text descriptions with batch processing."""
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Loaded sentence transformer model: {model_name}")

        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating embeddings"):
            batch = descriptions[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings).astype('float32')
        logger.info(f"Generated embeddings for {len(descriptions)} listings")

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"Built FAISS index with {index.ntotal} vectors")

        return index, embeddings, model
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise

def save_index_and_metadata(index, df, descriptions, output_index_path, output_metadata_path):
    """Save FAISS index and metadata."""
    try:
        faiss.write_index(index, output_index_path)
        logger.info(f"Saved FAISS index to {output_index_path}")

        metadata = {'descriptions': descriptions, 'data': df.to_dict(orient='records')}
        with open(output_metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {output_metadata_path}")
    except Exception as e:
        logger.error(f"Error saving index/metadata: {e}")
        raise

if __name__ == "__main__":
    json_path = "car_listings.json"
    output_index_path = "car_listings_index.faiss"
    output_metadata_path = "car_listings_metadata.pkl"

    try:
        df = load_car_listings(json_path)
        descriptions = create_text_descriptions(df)
        index, embeddings, model = build_faiss_index(descriptions, batch_size=128)
        save_index_and_metadata(index, df, descriptions, output_index_path, output_metadata_path)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise