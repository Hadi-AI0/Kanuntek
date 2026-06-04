import os
import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Configuration
BASE_MODEL = "intfloat/multilingual-e5-small"
OUTPUT_DIR = "models/legal-e5-mlm"

def load_text_data():
    texts = set()
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    print("Loading ALL available text for maximum vocabulary exposure...")
    for filename in files:
        file_path = os.path.join("data/processed", filename)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if "text" in data and data["text"]: texts.add(data["text"])
                if "context" in data and data["context"]: texts.add(data["context"])
                if "response" in data and data["response"]: texts.add(data["response"])
                if "instruction" in data and data["instruction"]: texts.add(data["instruction"])
                    
    print(f"Total unique legal text units loaded: {len(texts)}")
    return Dataset.from_dict({"text": list(texts)})

def main():
    raw_dataset = load_text_data()
    subset_size = int(len(raw_dataset) * 0.05)
    print(f"Selecting random 5% subset ({subset_size} samples)...")
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(subset_size))
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Logic to find last checkpoint for weight loading
    load_path = BASE_MODEL
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            load_path = max(checkpoints, key=os.path.getmtime)
            print(f"Loading weights from existing checkpoint: {load_path}")
    
    model = AutoModelForMaskedLM.from_pretrained(load_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Force clear GPU cache and set memory config
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, 
        per_device_train_batch_size=4,   # Drastically reduced to fit 8GB VRAM
        gradient_accumulation_steps=8,  # Effective batch size 32
        gradient_checkpointing=True,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Start trainer without resume_from_checkpoint to force active training on new snapshot
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=tokenized_dataset)

    print("Continuing MLM training on GPU with fresh trainer state...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
