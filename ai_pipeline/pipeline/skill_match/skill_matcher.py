# ai_pipeline/pipeline/skill_match/skill_matcher.py

import json
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import List, Dict, Any, Union
import time
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util

from .config import Config
from .data_processor import TextNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SkillMatcher:
    def __init__(self, config_source: Union[str, Dict[str, Any]] = "ai_pipeline/pipeline/skill_match/config.json"):
        self.embedding_cache = {}
        self.config = Config(config_source)
        logging.info(f"Loaded skill match configuration version {self.config.version}")

        self.synonym_map = self._load_json_file(Path(self.config.synonym_file))
        self.acronym_map = self._load_json_file(Path(self.config.acronym_file))
        
        self.normalizer = TextNormalizer(self.config.normalization, self.synonym_map, self.acronym_map)

        logging.info("Loading sentence transformer model... (this may take a moment)")
        self.model = SentenceTransformer(self.config.model_settings["model_name"])
        logging.info("Model loaded successfully.")
        
        self.cache_dir = Path("ai_pipeline/data/cache")
        model_name = self.config.model_settings.get("model_name", "default_model").replace("/", "_")
        self.cache_file = self.cache_dir / f"embeddings_{model_name}.joblib"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_cache()

        self.strong_threshold = self.config.skill_thresholds["strong"]
        self.weak_threshold = self.config.skill_thresholds["weak"]
        self.nice_threshold = self.config.skill_thresholds["nice"]
        self.semantic_weight = self.config.similarity_weights["semantic"]
        self.lexical_weight = self.config.similarity_weights["lexical"]
        self.scoring_weights = self.config.scoring_weights
        
        total_skill_weight = self.scoring_weights["required_skill"] + self.scoring_weights["optional_skill"]
        self.required_skill_relative_weight = self.scoring_weights["required_skill"] / total_skill_weight
        self.optional_skill_relative_weight = self.scoring_weights["optional_skill"] / total_skill_weight

    def _load_json_file(self, path: Path) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
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

    def _save_cache(self):
        try:
            joblib.dump(self.embedding_cache, self.cache_file)
        except Exception as e:
            logging.error(f"Error: Could not save cache file. Error: {e}")

    def __del__(self):
        try:
            self._save_cache()
        except Exception:
            pass
            
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
                lexical_sim = self.normalizer.lexical_similarity(part, cand_skill)
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

    def _generate_metadata(self, candidate_source_type: str, candidate_id: Any, job_source_type: str, job_id: Any, processing_time: float, candidate_skills_list: List[str], job_required: List[str], job_optional: List[str], results: Dict[str, Any] = None, error_message: str = None) -> Dict[str, Any]:
        status = "success" if error_message is None else "failed"
        metadata = {
            "source_identifiers": { 
                "candidate": {
                    "type": candidate_source_type, 
                    "id": candidate_id
                },
                "job": {
                    "type": job_source_type, 
                    "id": job_id
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "matching_details": {
                "status": status,
                "config_used": {
                    "model_name": self.config.model_settings.get("model_name"),
                    "thresholds": self.config.skill_thresholds,
                    "weights": {
                        "similarity": self.config.similarity_weights,
                        "skill_scoring": self.config.scoring_weights
                    },
                    "normalization": self.config.normalization,
                    "data_files": {
                        "synonym_file": self.config.synonym_file,
                        "acronym_file": self.config.acronym_file
                    }
                },
                "processing_time_seconds": round(processing_time, 2),
                "error_message": error_message
            },
            "auditing_notes": {
                "purpose": "To track the skill matching process for quality control and debugging.",
                "what_is_tracked": [
                    "Source types and identifiers for candidate and job (e.g., 'database', 'file_system').",
                    "Configuration used (model, thresholds, weights, normalization).",
                    "Paths to data files (synonyms, acronyms).",
                    "Success status and processing time.",
                    "Summary of matching results (scores, counts).",
                    "Error messages for debugging."
                ]
            }
        }

        if status == "success" and results:
            metadata["matching_details"]["input_counts"] = {
                "total_candidate_skills": len(candidate_skills_list),
                "total_job_required_skills": len(job_required),
                "total_job_optional_skills": len(job_optional)
            }
            metadata["matching_results_summary"] = {
                "overall_score": results.get("score"),
                "required_match_score": results.get("required_match_score"),
                "optional_match_score": results.get("optional_match_score"),
                "summary_counts": results.get("summary")
            }

        return metadata

    def match_skills(self, candidate_skills: List[str], job_required: List[str], job_optional: List[str] = None, candidate_source_type: str = "file_system", candidate_id: Any = "unknown_candidate", job_source_type: str = "file_system", job_id: Any = "unknown_job") -> Dict[str, Any]:
        start_time = time.time()
        results = None
        
        if job_optional is None:
            job_optional = []

        try:
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
            
            results = {
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
                "relative_skill_weights": {
                    "required": round(self.required_skill_relative_weight, 4),
                    "optional": round(self.optional_skill_relative_weight, 4)
                }
            }
            
        except Exception as e:
            logging.error(f"An error occurred during matching: {e}")

        finally:
            end_time = time.time()
            processing_time = end_time - start_time
            metadata = self._generate_metadata(
                candidate_source_type=candidate_source_type, # <--- TERUSKAN
                candidate_id=candidate_id,
                job_source_type=job_source_type, # <--- TERUSKAN
                job_id=job_id,
                processing_time=processing_time,
                candidate_skills_list=candidate_skills,
                job_required=job_required,
                job_optional=job_optional,
                results=results,
                error_message=str(e) if results is None else None
            )
            return {"results": results or {}, "metadata": metadata}