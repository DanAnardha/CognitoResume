# ai_pipeline/pipeline/skill_match/data_processor.py

import re
import string
from typing import Dict, Any

class TextNormalizer:
    def __init__(self, normalization_config: Dict[str, Any], synonym_map: Dict[str, Any], acronym_map: Dict[str, Any]):
        self.config = normalization_config
        self.synonym_map = synonym_map
        self.acronym_map = acronym_map

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        original_text = text
        if self.config.get("apply_acronyms", False):
            for lower_acronym, upper_acronym in self.acronym_map.items():
                text = re.sub(r'\b' + re.escape(lower_acronym) + r'\b', upper_acronym, text, flags=re.IGNORECASE)

        if self.config.get("apply_synonyms", False):
            normalized_input = text.lower()
            for canonical_term, synonyms in self.synonym_map.items():
                if normalized_input == canonical_term.lower() or normalized_input in [s.lower() for s in synonyms]:
                    return canonical_term.lower()

        if self.config.get("lowercase", False):
            text = text.lower()

        if self.config.get("strip_whitespace", False):
            text = text.strip()

        if self.config.get("remove_punctuation", False):
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        return text

    def lexical_similarity(self, text1: str, text2: str) -> float:
        from rapidfuzz import fuzz 
        
        norm1 = self.normalize(text1)
        norm2 = self.normalize(text2)
        return fuzz.token_set_ratio(norm1, norm2) / 100.0