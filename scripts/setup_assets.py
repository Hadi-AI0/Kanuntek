import os
from huggingface_hub import snapshot_download, hf_hub_download

# ==========================================
# CONFIGURATION - CHANGE THIS AFTER UPLOAD
# ==========================================
HF_USERNAME = "Hadi-AI0" # <-- Replace with your actual HF username
HF_REPO_NAME = "kanuntek-assets"
# ==========================================

def setup_kanuntek_assets():
    """
    Synchronizes local models with the Hugging Face asset repository.
    Handles E5 weights, FAISS index, and the Llama GGUF model.
    """
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"

    print(f"\n--- KANUNTEK ASSET SYNCHRONIZATION ---")
    print(f"[*] Target Repository: {repo_id}")

    # 1. Download Legal E5 Weights and FAISS Index
    print("[*] Downloading legal-e5-final and vector_store...")
    try:
        # We download the folders to maintain the directory structure
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["config.json", "*.safetensors", "*.json", "*.pkl", "*.faiss", "*.model"],
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("✅ Core embedding assets synced.")
    except Exception as e:
        print(f"❌ Core asset sync failed: {e}")
        print("Tip: Ensure your repo is Public or you are logged in via huggingface-cli.")

    # 2. Download Llama-3.2-3B LLM (GGUF)
    llm_filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    print(f"[*] Checking for {llm_filename}...")
    
    if os.path.exists(os.path.join(MODELS_DIR, llm_filename)):
        print(f"✅ {llm_filename} already exists locally.")
    else:
        try:
            print("[*] Downloading LLM from Hub...")
            hf_hub_download(
                repo_id=repo_id,
                filename=llm_filename,
                local_dir=MODELS_DIR
            )
            print(f"✅ {llm_filename} synced.")
        except Exception as e:
            print(f"❌ LLM sync failed: {e}")

    print("\n🚀 System Ready. You can now launch Kanuntek using ./start_kanuntek.bat")

if __name__ == "__main__":
    setup_kanuntek_assets()
