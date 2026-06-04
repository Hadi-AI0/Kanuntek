import os
import sys
import torch
from collections import deque

# 1. ROBUST IMPORTS
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms import LlamaCpp
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("Please run: pip install langchain-huggingface faiss-cpu llama-cpp-python pypdf")
    sys.exit(1)

from src.prompts import RAG_PROMPT
from src.utils import clean_turkish_text, prepare_e5_input
from src.document_reader import load_secure_document

# 2. CONFIGURATION
VECTOR_DB_PATH = "models/vector_store"
EMBED_MODEL_NAME = "models/legal-e5-mlm"
if os.path.exists("models/legal-e5-final"):
    EMBED_MODEL_NAME = "models/legal-e5-final"

LLM_MODEL_PATH = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Memory Configuration: Keep last 5 exchanges
CHAT_HISTORY = deque(maxlen=5)

def setup_rag():
    print(f"\n--- KANUNTEK INITIALIZATION ---")
    
    # Initialize Embeddings
    print(f"[*] Loading Embedding Engine ({EMBED_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load Main Vector Store (Resmi Gazete)
    print(f"[*] Connecting to Resmi Gazete Database...")
    main_vector_store = None
    if os.path.exists(VECTOR_DB_PATH):
        main_vector_store = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print("Warning: Main database not found. Assistant will rely on user documents.")

    # Initialize LLM
    print(f"[*] Loading LLM ({LLM_MODEL_PATH})...")
    if not os.path.exists(LLM_MODEL_PATH):
        print(f"Error: LLM file not found.")
        return None, None, None

    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_ctx=4096,      # Balanced window for CPU speed
        n_gpu_layers=10, # Minimal GPU assist if available, else CPU
        n_batch=512,     # Processing efficiency
        n_threads=os.cpu_count(),
        temperature=0.01,
        repeat_penalty=1.2,
        verbose=False
    )
    
    return main_vector_store, llm, embeddings

def ask_kanuntek(main_vs, session_vs, llm, query):
    global CHAT_HISTORY
    
    # 1. Clean and Prefix Query
    cleaned_q = clean_turkish_text(query)
    formatted_q = prepare_e5_input(cleaned_q, is_query=True)
    
    # 2. Retrieve Context (Search both databases if available)
    print(f"\nSearching databases...")
    all_docs = []
    
    if session_vs:
        all_docs.extend(session_vs.similarity_search(formatted_q, k=3))
    if main_vs:
        all_docs.extend(main_vs.similarity_search(formatted_q, k=2))
        
    context = "\n\n".join([d.page_content.replace("passage: ", "") for d in all_docs])
    
    # 3. Format Memory
    history_str = ""
    for q, a in CHAT_HISTORY:
        history_str += f"User: {q}\nAI: {a}\n"
    
    # 4. Format Prompt
    final_prompt = RAG_PROMPT.format(
        chat_history=history_str, 
        context=context, 
        question=cleaned_q
    )
    
    # 5. Generate Response
    print(f"Generating answer...")
    response = llm.invoke(final_prompt)
    
    # Update History
    CHAT_HISTORY.append((cleaned_q, response))
    
    print("\n" + "="*60)
    print(f"KANUNTEK CEVABI:\n")
    print(response)
    print("="*60 + "\n")

if __name__ == "__main__":
    main_vs, model, embed_engine = setup_rag()
    session_vs = None # Temporary index for uploaded files
    
    if model:
        print("\n" + "*"*50)
        print("KANUNTEK: Profesyonel Hukuk Yapay Zekası Hazır!")
        print("Komutlar:")
        print(" - /load [path] : Belge yükle (PDF/TXT)")
        print(" - /reset       : Sohbet geçmişini sil")
        print(" - q            : Çıkış")
        print("*"*50 + "\n")
        
        while True:
            u_input = input("Soru: ")
            
            if u_input.lower() == 'q': break
            
            if u_input.startswith("/load "):
                file_path = u_input.replace("/load ", "").strip()
                try:
                    session_vs = load_secure_document(file_path, embed_engine)
                    print("✅ Belge başarıyla yüklendi ve oturuma eklendi.")
                except Exception as e:
                    print(f"Hata: {e}")
                continue

            if u_input.lower() == "/reset":
                CHAT_HISTORY.clear()
                print("Oturum geçmişi silindi.")
                continue

            if not u_input.strip(): continue
            
            try:
                ask_kanuntek(main_vs, session_vs, model, u_input)
            except Exception as e:
                print(f"Hata olustu: {e}")
