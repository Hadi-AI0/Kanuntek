import os
from huggingface_hub import snapshot_download, hf_hub_download

def setup_kanuntek_assets():
    """
    Downloads all necessary models and indices from Hugging Face.
    This makes the project 'plug and play' for new users.
    """
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("--- KANUNTEK ASSET SETUP ---")

    # 1. Download Fine-tuned E5 Embeddings
    # REPLACE 'YOUR_HF_USERNAME' below after you create your Hugging Face repo!
    HF_REPO = "YOUR_HF_USERNAME/kanuntek-assets"
    
    print(f"[*] Downloading legal-e5-final weights from {HF_REPO}...")
    try:
        snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=["legal-e5-final/*", "vector_store/*"],
            local_dir=MODELS_DIR
        )
    except Exception as e:
        print(f"Download failed: {e}. Please ensure repo '{HF_REPO}' is public.")

    # 2. Download Llama-3.2-3B GGUF
    print("[*] Downloading Llama-3.2-3B LLM (GGUF)...")
    try:
        hf_hub_download(
            repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            local_dir=MODELS_DIR
        )
    except Exception as e:
        print(f"LLM download failed: {e}")

    print("\n✅ Setup Complete. Run 'python scripts/api_server.py' to start.")

if __name__ == "__main__":
    setup_kanuntek_assets()
