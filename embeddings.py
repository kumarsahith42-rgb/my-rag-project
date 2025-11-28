import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = 'text-embedding-004' 

def get_gemini_client():
    """Initializes and returns the Google Generative AI client."""
    if not GEMINI_API_KEY:
        
        raise ValueError("GEMINI_API_KEY not set in environment.")
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """Generates an embedding vector for a single string."""
    try:
        client = get_gemini_client()
        if not text:
            return []
            
        response = client.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise