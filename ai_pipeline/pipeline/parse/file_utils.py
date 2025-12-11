# ai_pipeline/pipeline/parse/file_utils.py

import os
import json
from pathlib import Path
from typing import Any, Dict, List

def load_json(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'")
        return {}

def save_json(data: Any, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n--- Result saved to '{filepath}' ---")

def load_text(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return ""

def get_output_filename(input_path: str, output_dir: str = "ai_pipeline/data/output/parse", prefix: str = "", suffix: str = "_parsed") -> str:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{prefix}{input_path.stem}{suffix}.json"
    return str(output_dir / filename)

def save_metadata(data: Dict[str, Any], filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n--- Metadata saved to '{filepath}' ---")

def get_metadata_filename(input_path: str, output_dir: str = "ai_pipeline/data/metadata/parse") -> str:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"metadata_{input_path.stem}.json"
    return str(output_dir / filename)