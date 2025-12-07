# backend/prototypes/skill_match/skill_matcher.py

import json
import numpy as np
import os
import joblib
from pathlib import Path
from rapidfuzz import fuzz
import logging
from typing import List, Dict, Any
import time
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedSkillMatcher:
    def __init__(self, config_path: str = "skill_config.json"):
        self.config = self._load_config(config_path)
        
        self.data_dir = Path("data")
        self.cache_dir = Path("cache")
        self.synonyms_file = self.data_dir / "synonyms.json"
        self.synonym_map = self._load_json_file(self.synonyms_file)
        self.acronyms_file = self.data_dir / "acronyms.json"
        self.cache_file = self.cache_dir / "embeddings.joblib"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.common_acronyms = self._load_json_file(self.acronyms_file)

        logging.info("Loading sentence transformer model... (this may take a moment)")
        self.model = SentenceTransformer(self.config["model_settings"]["model_name"])
        logging.info("Model loaded successfully.")
        
        self.strong_threshold = self.config["skill_thresholds"]["strong"]
        self.weak_threshold = self.config["skill_thresholds"]["weak"]
        self.nice_threshold = self.config["skill_thresholds"]["nice"]
        self.semantic_weight = self.config["similarity_weights"]["semantic"]
        self.lexical_weight = self.config["similarity_weights"]["lexical"]
        self.scoring_weights = self.config["scoring_weights"]
        
        total_skill_weight = self.scoring_weights["required_skill"] + self.scoring_weights["optional_skill"]
        self.required_skill_relative_weight = self.scoring_weights["required_skill"] / total_skill_weight
        self.optional_skill_relative_weight = self.scoring_weights["optional_skill"] / total_skill_weight
        
        self.embedding_cache = {}
        self._load_cache()

    def _load_config(self, config_path: str) -> dict:
        default_config = {
            "model_settings": {"model_name": "all-mpnet-base-v2"},
            "skill_thresholds": {"strong": 0.85, "weak": 0.65, "nice": 0.6},
            "similarity_weights": {"semantic": 0.7, "lexical": 0.3},
            "scoring_weights": {"required_skill": 0.8, "optional_skill": 0.2}
        }
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"Config file not found or invalid at {config_path}. Using default configuration.")
            return default_config

    def _load_json_file(self, path: Path) -> dict:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Data file not found at {path}. Using empty dict.")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {path}. Using empty dict.")
            return {}

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                self.embedding_cache = joblib.load(self.cache_file)
                logging.info(f"Loaded {len(self.embedding_cache)} embeddings from cache.")
            except Exception as e:
                logging.warning(f"Could not load cache file. Starting fresh. Error: {e}")
                self.embedding_cache = {}
        else:
            logging.info("Cache file not found. Starting with an empty cache.")
            self.embedding_cache = {}

    def _save_cache(self):
        try:
            joblib.dump(self.embedding_cache, self.cache_file)
        except Exception as e:
            logging.error(f"Error: Could not save cache file. Error: {e}")

    def _get_relative_weights(self) -> dict:
        return {
            "relative_skill_weights": {
                "required": self.required_skill_relative_weight,
                "optional": self.optional_skill_relative_weight
            }
        }

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        
        if text in self.synonym_map:
            return self.synonym_map[text]
        for canonical, synonyms in self.synonym_map.items():
            if text in synonyms:
                return canonical
        for lower, upper in self.common_acronyms.items():
            import re
            text = re.sub(r'\b' + re.escape(lower) + r'\b', upper, text)
            
        return text

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        return fuzz.token_set_ratio(norm1, norm2) / 100.0

    def _find_best_match(self, job_skill: str, candidate_skills: List[str], embedding_map: Dict[str, Any]) -> Dict[str, Any]:
        best_match = None
        best_score = 0
        best_semantic_sim = 0
        best_lexical_sim = 0
        
        parts = [s.strip() for s in job_skill.split(" or ")]
        
        for part in parts:
            part_embedding = embedding_map.get(part)
            if part_embedding is None: continue
            
            for cand_skill in candidate_skills:
                cand_embedding = embedding_map.get(cand_skill)
                if cand_embedding is None: continue

                semantic_sim = util.cos_sim(part_embedding, cand_embedding).item()
                lexical_sim = self._lexical_similarity(part, cand_skill)
                score = self.semantic_weight * semantic_sim + self.lexical_weight * lexical_sim
                
                if score > best_score:
                    best_score = score
                    best_match = cand_skill
                    best_semantic_sim = semantic_sim
                    best_lexical_sim = lexical_sim

        return {
            "skill": job_skill,
            "match": best_match,
            "sim": round(best_score, 4),
            "components": {
                "semantic": round(best_semantic_sim, 4),
                "lexical": round(best_lexical_sim, 4)
            }
        }

    def match_skills(self, candidate_skills: List[str], job_required: List[str], job_optional: List[str] = None) -> dict:
        if job_optional is None:
            job_optional = []

        start_total_time = time.time()
        
        required_parts = []
        for skill in job_required:
            required_parts.extend([s.strip() for s in skill.split(" or ")])
        
        all_texts = list(set(candidate_skills + required_parts + job_optional))
        
        texts_to_encode = [text for text in all_texts if text not in self.embedding_cache]
        if texts_to_encode:
            logging.info(f"Encoding {len(texts_to_encode)} unique skills in batch...")
            new_embeddings = self.model.encode(texts_to_encode, convert_to_tensor=True)
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self.embedding_cache[text] = embedding
            self._save_cache()
        
        embedding_map = {text: self.embedding_cache[text] for text in all_texts}
        logging.info("Starting skill matching...")

        required_results = [self._find_best_match(skill, candidate_skills, embedding_map) for skill in job_required]
        optional_results = [self._find_best_match(skill, candidate_skills, embedding_map) for skill in job_optional]

        for res in optional_results:
            res["ok"] = res["sim"] >= self.nice_threshold

        strong = [r for r in required_results if r["sim"] >= self.strong_threshold]
        weak = [r for r in required_results if self.weak_threshold <= r["sim"] < self.strong_threshold]
        missing = [r for r in required_results if r["sim"] < self.weak_threshold]

        summary = {
            "req_strong": len(strong),
            "req_weak": len(weak),
            "req_missing": len(missing),
            "nice_relevant": sum(1 for r in optional_results if r["ok"])
        }

        req_scores = [r["sim"] for r in required_results]
        required_match_score = np.mean(req_scores) if req_scores else 0
        opt_scores = [r["sim"] for r in optional_results]
        optional_match_score = np.mean(opt_scores) if opt_scores else 0
        skill_component_score = (
            self.required_skill_relative_weight * required_match_score + 
            self.optional_skill_relative_weight * optional_match_score
        )

        total_processing_time = time.time() - start_total_time
        logging.info(f"Matching completed in {total_processing_time:.2f} seconds.")

        return {
            "score": round(skill_component_score, 4),
            "required_match_score": round(required_match_score, 4),
            "optional_match_score": round(optional_match_score, 4),
            "required": {
                "strong": strong,
                "weak": weak,
                "missing": missing
            },
            "nice": optional_results,
            "summary": summary,
            **self._get_relative_weights()
        }

    def __del__(self):
        self._save_cache()