# backend/prototypes/skill_matcher.py

import json
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SkillMatcher:
    def __init__(self, config_path: str = "config.json"):
        # Load configuration file
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logging.info(f"Successfully loaded configuration from {config_path}")
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}. Exiting.")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {config_path}. Exiting.")
            raise

        # Initialize components from config
        self.model = SentenceTransformer(self.config["model_settings"]["model_name"])
        self._emb_cache = {}
        self.cache_file = Path(self.config["model_settings"]["cache_file_path"])
        self.max_cache_size = self.config["model_settings"]["max_cache_size"]
        
        # Load knowledge base files (JSON for domain map, YAML for skills)
        self.domain_map = self._load_json_file(self.config["knowledge_settings"]["domain_map_file"])
        self.skill_knowledge = self._load_yaml_file(self.config["knowledge_settings"]["skill_knowledge_file"])

        # Build skill-to-canonical maps and load embeddings cache
        self._build_reverse_maps()
        self._load_cache()

    def _load_json_file(self, path: str) -> dict:
        # Load data from a JSON file.
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Knowledge file not found at {path}. Using empty dict.")
            return {}

    def _load_yaml_file(self, path: str) -> dict:
        # Load data from a YAML file, handling PyYAML import error.
        try:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            logging.warning("PyYAML not installed. Skipping YAML knowledge loading.")
            return {}
        except FileNotFoundError:
            logging.warning(f"Knowledge file not found at {path}. Using empty dict.")
            return {}

    def _load_cache(self):
        # Load cached skill embeddings from file if it exists.
        if self.cache_file.exists():
            try:
                import joblib
                self._emb_cache = joblib.load(self.cache_file)
                logging.info(f"Loaded {len(self._emb_cache)} embeddings from cache.")
            except Exception as e:
                logging.warning(f"Could not load cache file. Starting fresh. Error: {e}")

    def _save_cache(self):
        # Save current skill embeddings cache to file.
        try:
            import joblib
            joblib.dump(self._emb_cache, self.cache_file)
        except Exception as e:
            logging.error(f"Error: Could not save cache file. Error: {e}")

    @staticmethod
    def _normalize(s: str) -> str:
        # Basic skill normalization: strip whitespace and lowercase.
        return str(s).strip().lower() if s else ""

    def _get_domain(self, skill: str) -> str | None:
        # Determine the skill domain based on keywords in the domain_map.
        s = self._normalize(skill)
        # Specific rule for Web
        if "javascript" in s and ("html" in s or "css" in s or "web" in s):
            return "web"
        # Check against general domain map
        for dom, words in self.domain_map.items():
            for w in words:
                if w in s:
                    return dom
        return None

    def _domain_similarity(self, a: str, b: str) -> float:
        # Check if two skills belong to the same domain.
        da = self._get_domain(a)
        db = self._get_domain(b)
        if da and db and da == db:
            return 1.0
        return 0.0

    # Build lookup map for canonical synonyms
    def _build_reverse_maps(self):
        # Create a mapping from synonym to canonical skill name.
        self.synonym_to_canonical = {}
        self.all_canonical_skills = set()

        # 1. Process synonyms
        for canonical, synonyms in self.skill_knowledge.get("synonyms", {}).items():
            self.all_canonical_skills.add(canonical)
            for synonym in synonyms:
                self.synonym_to_canonical[synonym] = canonical
        
        # 2. Process hierarchy to capture all canonical skills
        def add_skills_from_hierarchy(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    self.all_canonical_skills.add(key)
                    add_skills_from_hierarchy(value.get("children", []))
            elif isinstance(node, list):
                for item in node:
                    # Assumes list items are canonical skill names
                    self.all_canonical_skills.add(item)
        
        add_skills_from_hierarchy(self.skill_knowledge.get("hierarchy", {}))

    
    def _normalize_with_knowledge(self, skill: str) -> str:
        """
        Normalize skill using the knowledge base.
        1. Look up in the synonym map.
        2. If not found, use strict fuzzy matching against canonical skills.
        3. If no match, return the basic normalized skill.
        """
        if not skill: return ""

        norm_skill = self._normalize(skill)

        # 1. Check synonym map
        if norm_skill in self.synonym_to_canonical:
            return self.synonym_to_canonical[norm_skill]
        
        # 2. Fuzzy matching against all canonical and synonym keys
        all_keys = list(self.all_canonical_skills) + list(self.synonym_to_canonical.keys())
        
        # Optimization: Skip fuzzy if knowledge base is empty
        if not all_keys:
            return norm_skill

        # CRUCIAL FIX: Use a stricter scorer (fuzz.ratio) to avoid strange matches
        # and slightly lower the score_cutoff to catch more common typos.
        result = process.extractOne(
            norm_skill, 
            all_keys, 
            score_cutoff=85,  # Lowered from 90
            scorer=fuzz.ratio # Stricter scorer than default WRatio
        )
        
        if not result:
            return norm_skill # No match, return original normalized skill

        best_match, score, _ = result
        
        # If the best match is a synonym, get its canonical form
        return self.synonym_to_canonical.get(best_match, best_match)

    def _is_parent_child(self, parent: str, child: str) -> bool:
        """Recursively checks if 'parent' is an ancestor of 'child' in the hierarchy map."""
        hierarchy = self.skill_knowledge.get("hierarchy", {})
        
        # Helper to find all children of a specific node
        def find_children(node_name):
            for node, data in hierarchy.items():
                if node == node_name:
                    return data.get("children", [])
                found_children = find_children_recursive(data, node_name)
                if found_children is not None:
                    return found_children
            return []

        # Recursive helper for nested children
        def find_children_recursive(data, node_name):
            if isinstance(data, dict):
                for node, sub_data in data.items():
                    if node == node_name:
                        return sub_data.get("children", [])
                    result = find_children_recursive(sub_data, node_name)
                    if result is not None:
                        return result
            return None

        # Check direct children
        children = find_children(parent)
        if child in children:
            return True
        
        # Check grand-children (recursive check)
        for sub_child in children:
            if self._is_parent_child(sub_child, child):
                return True
        
        return False

    def _embed(self, text: str) -> np.ndarray:
        # Get the sentence embedding for the text, using cache if available.
        if text in self._emb_cache:
            return self._emb_cache[text]

        normalized_text = self._normalize(text)
        # Encode and L2-normalize the vector
        v = self.model.encode([normalized_text], convert_to_numpy=True)[0]
        v = v / (np.linalg.norm(v) + 1e-12)

        self._emb_cache[text] = v
        
        # Save cache periodically
        if len(self._emb_cache) % 50 == 0:
            self._save_cache()
            
        return v

    def _hybrid_similarity(self, a: str, b: str) -> float:
        # Calculate hybrid similarity score (Knowledge, Semantic, Lexical, Domain).
        if not a or not b:
            return 0.0

        # Normalize using knowledge base (canonicalize)
        norm_a = self._normalize_with_knowledge(a)
        norm_b = self._normalize_with_knowledge(b)

        # 1. Exact/Synonym match check
        if norm_a == norm_b:
            return 1.0

        # 2. Hierarchy (Parent-Child) check
        if self._is_parent_child(norm_a, norm_b) or self._is_parent_child(norm_b, norm_a):
            return 0.90 # High score for hierarchical match

        # 3. Semantic Similarity (Sentence Embeddings)
        va = self._embed(norm_a)
        vb = self._embed(norm_b)
        semantic = float(np.dot(va, vb))
        # Rescale from [-1, 1] to [0, 1]
        semantic = (semantic + 1) / 2

        # 4. Lexical Similarity (Partial Ratio)
        lexical = fuzz.partial_ratio(norm_a, norm_b) / 100.0
        
        # 5. Domain Match
        dom = self._domain_similarity(norm_a, norm_b)

        # Determine weights based on skill type (Soft vs Technical)
        if self._get_domain(norm_a) == "softskill" or self._get_domain(norm_b) == "softskill":
            weights = self.config["matching_logic"]["similarity_weights"]["soft_skills"]
        else:
            weights = self.config["matching_logic"]["similarity_weights"]["technical_skills"]

        # Calculate weighted average score
        score = (
            weights["semantic"] * semantic +
            weights["lexical"] * lexical +
            weights["domain"] * dom
        )

        # Apply penalty for cross-domain skills if weights allow
        if dom == 0:
            score *= self.config["matching_logic"]["penalties"]["cross_domain_factor"]

        return score

    def match_skills(self, candidate_skills: list, job_required: list, job_nice: list | None = None) -> dict:
        # Main function to match candidate skills against job requirements.
        if not candidate_skills or not isinstance(candidate_skills, list):
            raise ValueError("candidate_skills must be a non-empty list.")
        if not job_required or not isinstance(job_required, list):
            raise ValueError("job_required must be a non-empty list.")
        if job_nice is None: job_nice = []
        elif not isinstance(job_nice, list): raise ValueError("job_nice must be a list.")

        # Pre-embed candidate and required skills
        candidate_embeddings = {skill: self._embed(skill) for skill in candidate_skills}
        required_embeddings = {skill: self._embed(skill) for skill in job_required}

        results = []
        # Match each required job skill to the BEST available candidate skill
        for req_skill, req_vec in required_embeddings.items():
            best_skill, best_score = None, -1
            for cand_skill in candidate_skills:
                sc = self._hybrid_similarity(cand_skill, req_skill)
                if sc > best_score:
                    best_score, best_skill = sc, cand_skill
            results.append({"job_skill": req_skill, "best_candidate_skill": best_skill, "similarity": round(best_score, 4)})

        # Calculate core metrics
        strong_thresh = self.config["matching_logic"]["thresholds"]["strong_match"]
        strong_matches = sum(1 for r in results if r["similarity"] >= strong_thresh)
        soft_mean = float(np.mean([r["similarity"] for r in results])) if results else 0
        coverage = strong_matches / len(job_required) if job_required else 0
        
        strong_scores = [r["similarity"] for r in results if r["similarity"] >= strong_thresh]
        strong_ratio = np.mean(strong_scores) if strong_scores else 0 # Mean similarity of only strong matches

        # Calculate 'Nice to Have' bonus
        nice_bonus = 0
        if job_nice:
            nice_matches = 0
            nice_thresh = self.config["scoring_logic"]["nice_to_have"]["relevance_threshold"]
            for nice_skill in job_nice:
                # Check if ANY candidate skill is relevant to the nice-to-have skill
                if any(self._hybrid_similarity(cand, nice_skill) >= nice_thresh for cand in candidate_skills):
                    nice_matches += 1
            max_bonus = self.config["scoring_logic"]["nice_to_have"]["max_bonus_contribution"]
            nice_bonus = (nice_matches / len(job_nice)) * max_bonus

        # Calculate final composite score
        final_weights = self.config["scoring_logic"]["final_score_weights"]
        final_score = (
            final_weights["coverage"] * coverage +
            final_weights["soft_mean"] * soft_mean +
            final_weights["strong_ratio"] * strong_ratio +
            nice_bonus
        )

        # Ensure score does not exceed 1.0
        return {"final_score": round(min(final_score, 1.0), 4), "details": results}

    def generate_report(self, candidate_skills: list, job_required: list, job_nice: list, match_result: dict) -> dict:
        # Generates a detailed report based on the matching results.
        details = match_result["details"]
        
        strong_thresh = self.config["matching_logic"]["thresholds"]["strong_match"]
        weak_thresh = self.config["matching_logic"]["thresholds"]["weak_match"]
        
        # Categorize required skills
        strong = [d for d in details if d["similarity"] >= strong_thresh]
        weak = [d for d in details if weak_thresh <= d["similarity"] < strong_thresh]
        missing = [d for d in details if d["similarity"] < weak_thresh]

        # Analyze nice-to-have skills
        nice_results = []
        nice_thresh = self.config["scoring_logic"]["nice_to_have"]["relevance_threshold"]
        for nice in job_nice:
            best_skill, best_score = None, -1
            for cand in candidate_skills:
                sc = self._hybrid_similarity(cand, nice)
                if sc > best_score:
                    best_score, best_skill = sc, cand
            nice_results.append({
                "nice_skill": nice, "best_candidate_skill": best_skill,
                "similarity": round(best_score, 4), "is_relevant": best_score >= nice_thresh
            })

        return {
            "score": match_result["final_score"],
            "required": {
                "strong": [{"skill": d["job_skill"], "match": d["best_candidate_skill"], "sim": d["similarity"]} for d in strong],
                "weak": [{"skill": d["job_skill"], "match": d["best_candidate_skill"], "sim": d["similarity"]} for d in weak],
                "missing": [{"skill": d["job_skill"], "closest": d["best_candidate_skill"], "sim": d["similarity"]} for d in missing]
            },
            "nice": [{"skill": n["nice_skill"], "match": n["best_candidate_skill"], "sim": n["similarity"], "ok": n["is_relevant"]} for n in nice_results],
            "summary": {
                "req_strong": len(strong), "req_weak": len(weak), "req_missing": len(missing),
                "nice_relevant": sum(n["is_relevant"] for n in nice_results)
            }
        }

    def __del__(self):
        # Ensure cache is saved when the object is destroyed.
        self._save_cache()


if __name__ == "__main__":
    matcher = SkillMatcher()
    
    # Example Case B
    candidate_skills = [
        "Content Strategy", "A/B Testing", "Market Research",
        "Google Analytics", "Copywriting", "SEO", "Communication Skills"
    ]

    job_required_skills = [
        "Python", "SQL", "Machine Learning", "Deep Learning",
        "Statistics", "Model Evaluation", "Data Wrangling"
    ]

    job_optional_skills = [
        "AWS", "MLOps", "Big Data Technologies", "Time Series Analysis"
    ]

    match_result = matcher.match_skills(candidate_skills, job_required_skills, job_optional_skills)
    report = matcher.generate_report(candidate_skills, job_required_skills, job_optional_skills, match_result)
    print(json.dumps(report, indent=2))