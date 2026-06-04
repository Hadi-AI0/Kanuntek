import hashlib
import re
import os

def clean_turkish_text(text: str) -> str:
    """
    Rigorously cleans Turkish text from Resmi Gazete sources.
    Handles UTF-8 mojibake, normalizes whitespace, and preserves legal punctuation.
    """
    if not text:
        return ""

    # Comprehensive map for Turkish UTF-8 mojibake (Latin-1 misinterpreted as UTF-8)
    # This is a critical engineering standard for this project.
    mapping = {
        "ÄŸ": "ğ", "Ä": "Ğ",
        "Ä±": "ı", "Ä°": "İ",
        "Ã¶": "ö", "Ã–": "Ö",
        "Ã¼": "ü", "Ãœ": "Ü",
        "ÅŸ": "ş", "Å": "Ş",
        "Ã§": "ç", "Ã‡": "Ç",
        "â€™": "'", "â€œ": "“", "â€": "”",
        "â€“": "–", "Â": ""
    }

    for mojibake, correct in mapping.items():
        text = text.replace(mojibake, correct)

    # Normalize whitespace while preserving line breaks that might indicate paragraph splits
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def generate_deterministic_id(prefix: str, content: str) -> str:
    """
    Generates a deterministic ID for any project entity (doc, fikra, pair).
    Follows the format: {prefix}-{md5_hash}
    """
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    return f"{prefix}-{content_hash}"

def prepare_e5_input(text: str, is_query: bool = True) -> str:
    """
    Ensures text is properly prefixed for the intfloat/multilingual-e5-small model.
    Mandatory for maintaining high retrieval accuracy in Phase 3.
    """
    prefix = "query: " if is_query else "passage: "
    if not text.startswith(prefix):
        return f"{prefix}{text}"
    return text

def ensure_dir(path: str):
    """Utility to safely create directories for models or data logs."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
