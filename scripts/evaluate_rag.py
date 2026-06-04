import json
import random
import os
from tqdm import tqdm
from scripts.query_rag import load_vector_store

# Configuration
TEST_DATA_PATH = "data/processed/test.jsonl"
SAMPLE_SIZE = 50

def load_test_samples(k=SAMPLE_SIZE):
    samples = []
    print(f"Loading {k} random samples from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        all_instructions = [json.loads(line) for line in f if '"type": "instruction"' in line]
    return random.sample(all_instructions, min(k, len(all_instructions)))

def evaluate_retrieval():
    samples = load_test_samples()
    vector_store = load_vector_store()
    hits = 0
    print("\nStarting Retrieval Evaluation...")
    for item in tqdm(samples):
        query = f"query: {item['instruction']}"
        expected_fikra_id = item['metadata'].get('fikra_id')
        results = vector_store.similarity_search(query, k=5)
        found_ids = [res.metadata.get('fikra_id') for res in results]
        if expected_fikra_id in found_ids: hits += 1
    accuracy = (hits / len(samples)) * 100
    print(f"\n--- Evaluation Results ---")
    print(f"Total Samples Tested: {len(samples)}")
    print(f"Retrieval Recall@5: {accuracy:.2f}%")
    print(f"--------------------------")

if __name__ == "__main__":
    if not os.path.exists("models/vector_store"):
        print("Error: Vector store not found. Please run scripts/index_data.py first.")
    else:
        evaluate_retrieval()
