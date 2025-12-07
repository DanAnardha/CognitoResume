# backend/prototypes/skill_match/skill_matcher.py

import json
import numpy as np
from rapidfuzz import fuzz
import logging
from typing import List
import time
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedSkillMatcher:
    def __init__(self, config_path: str = "skill_config.json"):
        self.config = self._load_config(config_path)
        logging.info("Loading sentence transformer model... (this may take a moment)")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        logging.info("Model loaded successfully.")
        
        # Thresholds for categorizing matches
        self.strong_threshold = self.config["skill_thresholds"]["strong"]
        self.weak_threshold = self.config["skill_thresholds"]["weak"]
        self.nice_threshold = self.config["skill_thresholds"]["nice"]
        
        # Weights for combining semantic and lexical similarity
        self.semantic_weight = self.config["similarity_weights"]["semantic"]
        self.lexical_weight = self.config["similarity_weights"]["lexical"]
        self.scoring_weights = self.config["scoring_weights"]
        
        # Relative weights calculation
        total_skill_weight = self.scoring_weights["required_skill"] + self.scoring_weights["optional_skill"]
        self.required_skill_relative_weight = self.scoring_weights["required_skill"] / total_skill_weight
        self.optional_skill_relative_weight = self.scoring_weights["optional_skill"] / total_skill_weight
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.common_acronyms = {
            "deams": "DEAMS",
            "gafs/bq": "GAFS/BQ",
            "dts": "DTS",
            "gpc": "GPC",
            "fmiia": "FMFIA"
        }

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in configuration file at {config_path}")
            raise

    def _get_relative_weights(self) -> dict:
        return {
            "relative_skill_weights": {
                "required": self.required_skill_relative_weight,
                "optional": self.optional_skill_relative_weight
            }
        }

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        for lower, upper in self.common_acronyms.items():
            text = text.replace(lower, upper)
        return text

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        return fuzz.token_set_ratio(norm1, norm2) / 100.0

    def match_skills(self, candidate_skills: List[str], job_required: List[str], job_optional: List[str] = None) -> dict:
        if job_optional is None:
            job_optional = []

        start_total_time = time.time()
        
        # Process skills with "OR" to identify them and split them for matching
        or_skills = []
        processed_required_skills = []
        for skill in job_required:
            if " or " in skill.lower():
                parts = [s.strip() for s in skill.split("or")]
                or_skills.append({
                    "original": skill,
                    "parts": parts
                })
                processed_required_skills.extend(parts)
            else:
                processed_required_skills.append(skill)

        # Create a unique list of all texts to be embedded
        all_texts = list(set(candidate_skills + processed_required_skills + job_optional))
        
        # Check cache to avoid re-encoding
        texts_to_encode = [text for text in all_texts if text not in self.embedding_cache]
        
        if texts_to_encode:
            logging.info(f"Encoding {len(texts_to_encode)} unique skills in batch...")
            new_embeddings = self.model.encode(texts_to_encode, convert_to_tensor=True)
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self.embedding_cache[text] = embedding
        embedding_map = {text: self.embedding_cache[text] for text in all_texts}
        
        logging.info("Starting skill matching...")

        # Match required skills
        required_results = []
        
        # First process regular skills (non-OR)
        for req_skill in job_required:
            if " or " in req_skill.lower():
                continue
                
            best_match = None
            best_score = -1
            best_semantic_sim = 0
            best_lexical_sim = 0
            req_embedding = embedding_map.get(req_skill)
            if req_embedding is None: continue
            for cand_skill in candidate_skills:
                cand_embedding = embedding_map.get(cand_skill)
                if cand_embedding is None: continue

                semantic_sim = util.cos_sim(req_embedding, cand_embedding).item()
                lexical_sim = self._lexical_similarity(req_skill, cand_skill)
                score = self.semantic_weight * semantic_sim + self.lexical_weight * lexical_sim
                if score > best_score:
                    best_score = score
                    best_match = cand_skill
                    best_semantic_sim = semantic_sim
                    best_lexical_sim = lexical_sim
            
            required_results.append({
                "skill": req_skill,
                "match": best_match,
                "sim": round(best_score, 4),
                "components": {
                    "semantic": round(best_semantic_sim, 4),
                    "lexical": round(best_lexical_sim, 4)
                }
            })
        
        # Now process OR skills
        for or_skill in or_skills:
            original_skill = or_skill["original"]
            parts = or_skill["parts"]
            
            best_match = None
            best_score = -1
            best_semantic_sim = 0
            best_lexical_sim = 0
            best_part = None
            
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
                        best_part = part
            
            required_results.append({
                "skill": original_skill,
                "match": best_match,
                "sim": round(best_score, 4),
                "components": {
                    "semantic": round(best_semantic_sim, 4),
                    "lexical": round(best_lexical_sim, 4)
                }
            })

        # Match optional skills
        optional_results = []
        for opt_skill in job_optional:
            best_match = None
            best_score = -1
            best_semantic_sim = 0
            best_lexical_sim = 0
            opt_embedding = embedding_map.get(opt_skill)
            if opt_embedding is None: continue
            for cand_skill in candidate_skills:
                cand_embedding = embedding_map.get(cand_skill)
                if cand_embedding is None: continue
                semantic_sim = util.cos_sim(opt_embedding, cand_embedding).item()
                lexical_sim = self._lexical_similarity(opt_skill, cand_skill)
                score = self.semantic_weight * semantic_sim + self.lexical_weight * lexical_sim

                if score > best_score:
                    best_score = score
                    best_match = cand_skill
                    best_semantic_sim = semantic_sim
                    best_lexical_sim = lexical_sim
            
            optional_results.append({
                "skill": opt_skill,
                "match": best_match,
                "sim": round(best_score, 4),
                "ok": best_score >= self.nice_threshold,
                "components": {
                    "semantic": round(best_semantic_sim, 4),
                    "lexical": round(best_lexical_sim, 4)
                }
            })

        # Categorize required skills
        strong = [r for r in required_results if r["sim"] >= self.strong_threshold]
        weak = [r for r in required_results if self.weak_threshold <= r["sim"] < self.strong_threshold]
        missing = [r for r in required_results if r["sim"] < self.weak_threshold]

        # Calculate summary
        summary = {
            "req_strong": len(strong),
            "req_weak": len(weak),
            "req_missing": len(missing),
            "nice_relevant": sum(1 for r in optional_results if r["ok"])
        }

        # Calculate scores
        req_scores = [r["sim"] for r in required_results]
        required_match_score = np.mean(req_scores) if req_scores else 0
        opt_scores = [r["sim"] for r in optional_results]
        optional_match_score = np.mean(opt_scores) if opt_scores else 0
        skill_component_score = (
            self.required_skill_relative_weight * required_match_score + 
            self.optional_skill_relative_weight * optional_match_score
        )
        
        # Save scoring breakdown to separate file
        scoring_breakdown = {
            "skill_component": {
                "required_score": required_match_score,
                "optional_score": optional_match_score,
                "required_weight": self.required_skill_relative_weight,
                "optional_weight": self.optional_skill_relative_weight,
                "calculation": f"{self.required_skill_relative_weight:.4f} * {required_match_score:.4f} + {self.optional_skill_relative_weight:.4f} * {optional_match_score:.4f} = {skill_component_score:.4f}"
            }
        }
        self._save_scoring_breakdown(scoring_breakdown)

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

    def _save_scoring_breakdown(self, scoring_breakdown: dict, file_path: str = "results/scoring_breakdown.json"):
        try:
            with open(file_path, 'w') as f:
                json.dump(scoring_breakdown, f, indent=2)
            logging.info(f"Scoring breakdown saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving scoring breakdown: {e}")