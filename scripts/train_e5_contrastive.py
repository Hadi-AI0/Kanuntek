import json
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Configuration
MODEL_PATH = "models/legal-e5-mlm"
OUTPUT_PATH = "models/legal-e5-final"

def load_contrastive_data():
    examples = []
    files = ["train.jsonl", "val.jsonl"]
    print("Loading instruction pairs for retrieval training...")
    for filename in files:
        file_path = os.path.join("data/processed", filename)
        if not os.path.exists(file_path): continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "instruction":
                    # E5 requirement: prefix instructions with "query: " and context with "passage: "
                    examples.append(InputExample(texts=[f"query: {data['instruction']}", f"passage: {data['context']}"]))
    return examples

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading MLM model from {MODEL_PATH}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_PATH, device=device)
    
    train_examples = load_contrastive_data()
    # Batch size 16 is safe for 8GB GPU with sentence-transformers
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"Starting Contrastive Fine-tuning on {device.upper()} (Phase 2)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=1, 
        show_progress_bar=True,
        checkpoint_path=f"{OUTPUT_PATH}_checkpoints",
        use_amp=True, # Mixed precision speedup
        checkpoint_save_steps=500
    )
    model.save(OUTPUT_PATH)
    print(f"Phase 2 complete. Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
