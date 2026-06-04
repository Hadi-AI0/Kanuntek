import json
import os

MOJIBAKE_MAP = {
    "ÄŸ": "ğ", "Ä": "Ğ", "Ä±": "ı", "Ä°": "İ", "Ã¶": "ö", "Ã–": "Ö",
    "Ã¼": "ü", "Ãœ": "Ü", "ÅŸ": "ş", "Å": "Ş", "Ã§": "ç", "Ã‡": "Ç",
    "â€™": "'", "â€œ": "“", "â€": "”",
}

def manual_replace(text):
    if not text: return text
    for mojibake, correct in MOJIBAKE_MAP.items():
        text = text.replace(mojibake, correct)
    return text

def fix_file(file_path):
    print(f"Fixing {file_path}...")
    temp_path = file_path + ".tmp"
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(temp_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            for key in ["text", "context", "response", "instruction"]:
                if key in data: data[key] = manual_replace(data[key])
            if "metadata" in data and "title" in data["metadata"]:
                data["metadata"]["title"] = manual_replace(data["metadata"]["title"])
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    os.replace(temp_path, file_path)

if __name__ == "__main__":
    for p in ["data/processed/train.jsonl", "data/processed/val.jsonl", "data/processed/test.jsonl"]:
        if os.path.exists(p): fix_file(p)
