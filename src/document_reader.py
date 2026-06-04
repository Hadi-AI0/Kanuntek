import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.utils import clean_turkish_text, prepare_e5_input

def load_secure_document(file_path, embeddings):
    """
    Parses a local document (PDF or TXT) and returns a temporary FAISS index.
    Everything is processed in-memory or in a temporary local state.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text_content = ""
    extension = os.path.splitext(file_path)[1].lower()

    print(f"[*] Securely parsing: {os.path.basename(file_path)}")
    
    if extension == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
    elif extension == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    else:
        raise ValueError("Unsupported file format. Please use PDF or TXT.")

    # Split into chunks (Simple paragraph-based split for legal clarity)
    raw_chunks = [c.strip() for c in text_content.split("\n\n") if len(c.strip()) > 20]
    
    processed_docs = []
    for chunk in raw_chunks:
        cleaned = clean_turkish_text(chunk)
        # Prefix for E5
        formatted = prepare_e5_input(cleaned, is_query=False)
        processed_docs.append(Document(page_content=formatted, metadata={"source": file_path}))

    print(f"[*] Creating temporary session index ({len(processed_docs)} units)...")
    temp_vector_store = FAISS.from_documents(processed_docs, embeddings)
    
    return temp_vector_store
