# backend/prototypes/extract/extract.py

import json
import re
import spacy
from spacy_layout import spaCyLayout
from pathlib import Path

def load_config(config_path: str = "extract_config.json") -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' is invalid.")
        exit()

def save_chunks(chunks: list, filepath: str):
    # Ensure the directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n--- Text chunks saved to '{filepath}' ---")

def clean_markdown(text: str) -> str:
    text = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
    text = text.replace("�", " ").replace("·", "- ").replace("•", "- ").replace("○", "- ")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text).replace("\\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def extract_text_from_pdf(pdf_path: str, spacy_model_name: str) -> str:
    try:
        nlp_layout = spacy.load(spacy_model_name)
        layout = spaCyLayout(nlp_layout)
        layout_doc = layout(pdf_path)
        raw_markdown = layout_doc._.markdown
        cleaned_markdown = clean_markdown(raw_markdown)
        print("--- Markdown text extracted and cleaned ---")
        return cleaned_markdown
    except OSError:
        print(f"Error: spaCy model '{spacy_model_name}' not found.")
        print("Run: python -m spacy download en_core_web_sm")
        exit()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        exit()

def main():
    config = load_config()
    cleaned_markdown = extract_text_from_pdf(config["pdf_path"], config["spacy_model"])
    chunks = chunk_text(cleaned_markdown, config["chunking"]["size"], config["chunking"]["overlap"])
    print(f"Text split into {len(chunks)} chunks.")
    save_chunks(chunks, config["output_path"])

if __name__ == "__main__":
    main()