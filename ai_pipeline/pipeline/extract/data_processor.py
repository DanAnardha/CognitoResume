# ai_pipeline/pipeline/extract/data_processor.py

import re
from typing import List, Dict, Any

class TextProcessor:
    def __init__(self, chunking_config: Dict[str, Any]):
        self.size = chunking_config.get("size", 13000)
        self.overlap = chunking_config.get("overlap", 400)

    def clean_markdown(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
        text = text.replace("�", " ").replace("·", "- ").replace("•", "- ").replace("○", "- ") 
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text) 
        text = re.sub(r"[ ]{2,}", " ", text).replace("\\n", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]: 
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.size, len(text))
            chunks.append(text[start:end])
            start += self.size - self.overlap
        return chunks
