import pinecone
import json
import os
import time
from rag.embeddings import generate_embedding

# --- Configuration (from .env) ---
PINECECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "clinic-faqs-index" 
FAQ_DATA_PATH = "data/clinic_info.json"
EMBEDDING_DIMENSION = 768 

def initialize_pinecone_index() -> pinecone.Index:
    """Initializes Pinecone and returns the index object."""
    if not all([PINECECONE_API_KEY, PINECONE_ENVIRONMENT]):
        raise ValueError("Pinecone environment variables not fully set.")

    pinecone.init(api_key=PINECECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    if INDEX_NAME not in pinecone.list_indexes():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pinecone.create_index(INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric='cosine')
        while not pinecone.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1) 

    return pinecone.Index(INDEX_NAME)

def index_faq_data():
    """Loads FAQ data, embeds it, and upserts to Pinecone."""
    print("Starting FAQ data indexing...")
    try:
        with open(FAQ_DATA_PATH, 'r') as f:
            faq_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {FAQ_DATA_PATH} not found. Indexing aborted.")
        return

    index = initialize_pinecone_index()
    vectors_to_upsert = []
    batch_size = 100 
    
    for i, item in enumerate(faq_data):
        doc_id = f"faq_{i}"
        document_text = f"Q: {item.get('question', '')}. A: {item.get('answer', '')} (Topic: {item.get('topic', 'General')})"
        
        embedding_vector = generate_embedding(document_text)
        
        vectors_to_upsert.append((
            doc_id, 
            embedding_vector, 
            {"topic": item.get('topic', 'General'), "original_text": document_text} 
        ))

        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert = []

    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

    print("Indexing complete.")