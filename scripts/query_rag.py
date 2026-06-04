import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.utils import clean_turkish_text, prepare_e5_input

# 1. CONFIGURATION
VECTOR_DB_PATH = "models/vector_store"
MODEL_NAME = "models/legal-e5-final"
if not os.path.exists(MODEL_NAME):
    MODEL_NAME = "models/legal-e5-mlm"

def verify_retrieval(query_text: str, k: int = 3):
    """
    Test script to verify Phase 2 model + Phase 3 index accuracy.
    """
    print(f"--- KANUNTEK RETRIEVAL TEST ---")
    print(f"Loading embedding engine: {MODEL_NAME}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"Connecting to FAISS index...")
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"Error: Vector store not found. Run index_data.py first.")
        return

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # engineering standard: Clean and Prefix
    cleaned_q = clean_turkish_text(query_text)
    formatted_q = prepare_e5_input(cleaned_q, is_query=True)

    print(f"Searching for: '{cleaned_q}'")
    results = vector_store.similarity_search_with_score(formatted_q, k=k)

    print("\n" + " TOP MATCHES ".center(60, "="))
    for i, (doc, score) in enumerate(results):
        # Lower score is better (L2 distance)
        print(f"\n[{i+1}] RELEVANCE SCORE: {score:.4f}")
        print(f"FIKRA ID: {doc.metadata.get('fikra_id')}")
        content = doc.page_content.replace("passage: ", "")
        print(f"TEXT: {content[:300]}...")
    print("\n" + "="*60)

if __name__ == "__main__":
    test_query = "Pakistan ile yapılan mal ticareti anlaşmasının onaylanması"
    verify_retrieval(test_query)
