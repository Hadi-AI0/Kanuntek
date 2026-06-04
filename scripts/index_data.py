import json
import os
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.utils import clean_turkish_text, prepare_e5_input, ensure_dir

# 1. PATH CONFIGURATION
# We prioritize the Phase 2 (Contrastive) model if it exists.
MODEL_NAME = "models/legal-e5-mlm" 
if os.path.exists("models/legal-e5-final"):
    MODEL_NAME = "models/legal-e5-final"
    print(f"Using Phase 2 model: {MODEL_NAME}")
else:
    print(f"Phase 2 not found. Falling back to Phase 1 model: {MODEL_NAME}")

VECTOR_DB_PATH = "models/vector_store"
DATA_PATH = "data/processed"

def load_documents():
    """
    Loads atomic Fıkra units from JSONL files, cleans them, 
    and prepares them for E5 vectorization.
    """
    documents = []
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    print(f"Scanning {DATA_PATH} for legal units...")
    for filename in files:
        file_path = os.path.join(DATA_PATH, filename)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {filename}", unit="units"):
                data = json.loads(line)
                
                # We only index 'embedding' types (Atomic Fıkra paragraphs)
                if data.get("type") == "embedding":
                    # engineering standard: Clean mojibake before indexing
                    original_text = data.get("text", "")
                    cleaned_text = clean_turkish_text(original_text)
                    
                    # engineering standard: E5 requires 'passage: ' prefix for indexing
                    formatted_text = prepare_e5_input(cleaned_text, is_query=False)
                    
                    metadata = data.get("metadata", {})
                    metadata["fikra_id"] = data.get("fikra_id")
                    metadata["source"] = filename
                    
                    documents.append(Document(page_content=formatted_text, metadata=metadata))
    
    return documents

def main():
    # Ensure output directory exists
    ensure_dir("models")

    # 1. Load and Clean Data
    docs = load_documents()
    if not docs:
        print("Error: No documents found to index. Check data/processed/ directory.")
        return

    # 2. Initialize Embeddings
    # Using CPU for indexing to save GPU VRAM for LLM inference later
    print(f"Initializing embedding engine from: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 3. Build FAISS Vector Store
    # Process in batches to manage memory efficiently
    batch_size = 2000
    print(f"Starting vectorization of {len(docs)} documents (Batch size: {batch_size})...")
    
    # Create the first batch
    vector_store = FAISS.from_documents(docs[:batch_size], embeddings)
    
    # Add subsequent batches
    for i in range(batch_size, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"Indexed {min(i + batch_size, len(docs))}/{len(docs)}...")

    # 4. Persistence
    print(f"Saving vector database to: {VECTOR_DB_PATH}")
    vector_store.save_local(VECTOR_DB_PATH)
    print("🚀 Phase 3: Indexing complete! System is ready for RAG inference.")

if __name__ == "__main__":
    main()
