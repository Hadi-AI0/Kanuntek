import os
import sys
import json

# Sync PYTHONPATH so imports work correctly when running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from src.utils import clean_turkish_text, prepare_e5_input

class KanuntekEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, device='cpu')
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def verify_retrieval(query_text: str, k: int = 2):
    print(f"--- KANUNTEK PRODUCTION RETRIEVAL TEST ---")
    MODEL_PATH = "models/legal-e5-final"
    VECTOR_DB_PATH = "models/vector_store"

    # 1. Load Custom Wrapper
    print(f"Loading Kanuntek Model: {MODEL_PATH}")
    embeddings = KanuntekEmbeddings(MODEL_PATH)

    # 2. Load FAISS
    print(f"Connecting to FAISS database...")
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Clean and Search
    cleaned_q = clean_turkish_text(query_text)
    formatted_q = prepare_e5_input(cleaned_q, is_query=True)

    print(f"Query: '{cleaned_q}'")
    results = vector_store.similarity_search_with_score(formatted_q, k=k)

    print("\n" + "="*60)
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] SCORE: {score:.4f} | ID: {doc.metadata.get('fikra_id')}")
        content = doc.page_content.replace("passage: ", "")
        print(f"CONTENT: {content[:200]}...")
        print("-" * 30)
    print("="*60)

if __name__ == "__main__":
    verify_retrieval("Pakistan ticaret anlaşması onayı")
