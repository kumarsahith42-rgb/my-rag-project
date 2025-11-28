from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
# Import RAG components
from rag.vector_store import index_faq_data
from rag.faq_rag import retrieve_and_generate_answer

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- Initialization on Startup ---
# NOTE: Call this once to build the Pinecone index from clinic_info.json
# You might want to move this to a separate setup script for production
try:
    print("Indexing FAQ data on startup...")
    # index_faq_data()
    print("Indexing complete/Index ready.")
except Exception as e:
    print(f"Failed to initialize RAG index: {e}")


@app.route('/chat', methods=['POST'])
def chat():
    """
    Simulated chat endpoint. 
    In a full agent, the Scheduling Agent would call the RAG function
    when an FAQ intent is detected.
    """
    data = request.get_json()
    user_message = data.get('message', '')

    # --- SIMULATE INTENT DETECTION ---
    # For this example, we assume any message is an FAQ
    if "insurance" in user_message.lower() or "cancel" in user_message.lower():
        # This simulates the context switch to the RAG component
        rag_response = retrieve_and_generate_answer(user_message)
        
        # In the full agent, you would transition back to the scheduling flow here.
        return jsonify({"agent_response": rag_response, "intent": "FAQ_ANSWER"})
    
    # Fallback response for scheduling/other intents (not implemented here)
    return jsonify({"agent_response": "I see you are asking about scheduling. Please tell me the reason for your visit.", "intent": "SCHEDULING"})


if __name__ == '__main__':
    # Flask typically runs on port 5000 by default
    app.run(host='0.0.0.0', port=os.getenv("BACKEND_PORT", 5000), debug=True)