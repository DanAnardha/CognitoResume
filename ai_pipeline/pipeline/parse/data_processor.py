# ai_pipeline/pipeline/parse/data_processor.py

import json
import re
from typing import Any, Dict, List, Union, Set, Tuple
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from json_repair import repair_json
except ImportError:
    print("Error: 'json-repair' library not found. Please install it.")
    exit()

from ai_pipeline.pipeline.parse.file_utils import load_text, get_output_filename, save_json

class ResumeProcessor: 
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]: 
        if not response_text:
            return json.loads(json.dumps(self.schema))
        
        cleaned = response_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", cleaned)
        raw_json = match.group(0) if match else cleaned

        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(raw_json)
                return json.loads(repaired)
            except Exception:
                print("Failed to repair JSON. Returning default schema.")
                return json.loads(json.dumps(self.schema))
    
    def normalize_json_preserve_structure(self, data: Any, remove_empty: bool = True, remove_duplicates: bool = True, case_sensitive_duplicates: bool = True, preserve_order: bool = True, deep_copy: bool = True) -> Any:
        if deep_copy:
            data = json.loads(json.dumps(data))
        
        if isinstance(data, Mapping):
            return self._normalize_dict_preserve(data, remove_empty, remove_duplicates, case_sensitive_duplicates, preserve_order)
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return self._normalize_list_preserve(data, remove_empty, remove_duplicates, case_sensitive_duplicates, preserve_order)
        else:
            return self._normalize_primitive(data, remove_empty)
    
    def _normalize_dict_preserve(self, data: Dict[str, Any], remove_empty: bool, remove_duplicates: bool, case_sensitive_duplicates: bool, preserve_order: bool) -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            normalized_value = self.normalize_json_preserve_structure(value, remove_empty, remove_duplicates, case_sensitive_duplicates, preserve_order, deep_copy=False)
            result[key] = normalized_value
        if remove_empty and self._is_completely_empty(result):
            return {}
        return result
    
    def _normalize_list_preserve(self, data: List[Any], remove_empty: bool, remove_duplicates: bool, case_sensitive_duplicates: bool, preserve_order: bool) -> List[Any]:
        result = []
        seen = set()
        for item in data:
            normalized_item = self.normalize_json_preserve_structure(item, remove_empty, remove_duplicates, case_sensitive_duplicates, preserve_order, deep_copy=False)
            if remove_empty and self._is_completely_empty(normalized_item):
                continue
            if remove_duplicates:
                item_key = self._make_hashable(normalized_item, case_sensitive_duplicates)
                if item_key in seen:
                    continue
                seen.add(item_key)
            result.append(normalized_item)
        return result
    
    def _normalize_primitive(self, data: Any, remove_empty: bool) -> Any:
        if isinstance(data, str):
            return data.strip()
        return data
    
    def _is_completely_empty(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict)) and not value:
            return True
        if isinstance(value, dict):
            return all(self._is_completely_empty(v) for v in value.values())
        if isinstance(value, list):
            return all(self._is_completely_empty(item) for item in value)
        return False
    
    def _make_hashable(self, value: Any, case_sensitive: bool = True) -> Union[Tuple, str, int, float, bool, None]:
        if isinstance(value, Mapping):
            return tuple(sorted((k, self._make_hashable(v, case_sensitive)) for k, v in value.items()))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(self._make_hashable(item, case_sensitive) for item in value)
        elif isinstance(value, str):
            return value if case_sensitive else value.lower()
        return value
    
    def local_structural_cleaning(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_json_preserve_structure(resume_data)
        for key in ["education", "work_experience", "certifications", "projects"]:
            if key in normalized and isinstance(normalized[key], dict) and "items" in normalized[key]:
                non_empty_items = [
                    item for item in normalized[key]["items"] 
                    if any(str(item.get(field, "")).strip() for field in item if field != "tech_stack") # Khusus untuk tech_stack yang adalah list
                ]
                normalized[key]["items"] = non_empty_items
        
        if 'skills' in normalized and isinstance(normalized['skills'], dict) and 'items' in normalized['skills']:
            skills = [skill.strip() for skill in normalized['skills']['items'] if isinstance(skill, str) and skill.strip()]
            unique_skills = []
            seen_skills = set()
            for skill in skills:
                if skill.lower() not in seen_skills:
                    seen_skills.add(skill.lower())
                    unique_skills.append(skill)
            normalized['skills']['items'] = unique_skills
            
        return normalized
    
    def merge_results(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        if new.get("summary", {}).get("confidence", 0) > base.get("summary", {}).get("confidence", 0):
            base["summary"] = new.get("summary", base["summary"])
            
        for key in ["education", "work_experience", "certifications", "projects"]:
            if key in new and "items" in new[key] and isinstance(new[key]["items"], list):
                if new[key]["items"]:
                    base[key]["items"].extend(new[key]["items"])
                    if new[key].get("confidence", 0) > base[key].get("confidence", 0):
                        base[key]["confidence"] = new[key]["confidence"]

        if "skills" in new and "items" in new["skills"] and isinstance(new["skills"]["items"], list):
            base["skills"]["items"].extend(new["skills"]["items"])
            if new["skills"].get("confidence", 0) > base["skills"].get("confidence", 0):
                base["skills"]["confidence"] = new["skills"]["confidence"]
        return base
    
    def final_validation_and_cleaning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Starting Final Validation and Structural Cleaning ---")
        final_data = json.loads(json.dumps(self.schema))

        def validate_confidence(conf_val: Any) -> float:
            try:
                return round(float(conf_val), 2)
            except (ValueError, TypeError):
                return 0.00

        if "summary" in data and isinstance(data["summary"], dict):
            final_data["summary"]["value"] = data["summary"].get("value", "")
            final_data["summary"]["confidence"] = validate_confidence(data["summary"].get("confidence"))

        for key in ["education", "work_experience", "certifications", "projects"]:
            if key in data and isinstance(data[key], dict) and "items" in data[key]:
                non_empty_items = [item for item in data[key]["items"] if not self._is_completely_empty(item)]
                final_data[key]["items"] = non_empty_items
                final_data[key]["confidence"] = validate_confidence(data[key].get("confidence"))

        if "skills" in data and isinstance(data["skills"], dict):
            if isinstance(data["skills"].get("items"), list):
                skills_list = [str(skill).strip() for skill in data["skills"]["items"] if str(skill).strip()]
                unique_skills = []
                seen_skills = set()
                for skill in skills_list:
                    if skill.lower() not in seen_skills:
                        seen_skills.add(skill.lower())
                        unique_skills.append(skill)
                final_data["skills"]["items"] = unique_skills
            final_data["skills"]["confidence"] = validate_confidence(data["skills"].get("confidence"))
            
        return final_data
    
    def run_pipeline_step(self, llm_manager, step_config: Dict[str, Any], input_data: Any, prompt_root: str) -> Dict[str, Any]:
        step_name = step_config.get("name", "unknown")
        system_prompt_path = step_config.get("system_prompt", "")
        prompt_path = step_config.get("prompt", "")
        temperature = step_config.get("temperature", 0.0)
        max_tokens = step_config.get("max_tokens", 4096)
        print(f"\n--- Running Step: {step_name} ---")
        
        system_prompt = load_text(f"{prompt_root}/{system_prompt_path}")
        prompt_template = load_text(f"{prompt_root}/{prompt_path}")
        schema_string = json.dumps(self.schema, indent=2)
        
        if step_name == "parse":
            chunks = input_data
            merged_result = json.loads(json.dumps(self.schema))
            for i, chunk in enumerate(chunks):
                print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
                user_prompt = prompt_template.format(schema=schema_string, chunk=chunk)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                
                # Override temperature and max_tokens for this step
                for provider in llm_manager.providers:
                    provider.temperature = temperature
                    provider.max_tokens = max_tokens
                
                response_text, _ = llm_manager.get_response(messages)
                
                if not response_text:
                    print(f"Skipping chunk {i+1} due to API failure.")
                    continue
                
                partial_result = self.parse_llm_response(response_text)
                merged_result = self.merge_results(merged_result, partial_result)
            
            return merged_result
        else:
            input_json_string = json.dumps(input_data, indent=2)
            user_prompt = prompt_template.format(schema=schema_string, input_json=input_json_string)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            # Override temperature and max_tokens for this step
            for provider in llm_manager.providers:
                provider.temperature = temperature
                provider.max_tokens = max_tokens
            
            response_text, _ = llm_manager.get_response(messages)
            
            if not response_text:
                print("Validation step failed due to API issues. Returning input data.")
                return input_data
            
            return self.parse_llm_response(response_text)


    def process_resume(self, llm_manager, chunks: List[str], config) -> Dict[str, Any]:
        primary_provider_name = config.active_provider
        primary_pipeline = config.get_pipeline_config(primary_provider_name)
        steps = primary_pipeline.get("steps", [])
        
        if not steps:
            raise ValueError(f"No steps found for primary provider '{primary_provider_name}'.")
            
        first_step = steps[0]
        
        print("--- Determining active provider by running first step...")
        raw_data_after_parse = self.run_pipeline_step(
            llm_manager, first_step, chunks, config.prompt_root
        )
        
        actual_provider_name = llm_manager.get_used_provider()
        if not actual_provider_name:
            print("Warning: Could not determine used provider. Defaulting to primary.")
            actual_provider_name = primary_provider_name

        print(f"--- Active provider determined as: '{actual_provider_name}'. ---")
        print("--- Performing local structural cleaning on parsed data... ---")
        cleaned_data_after_parse = self.local_structural_cleaning(raw_data_after_parse)
        actual_pipeline_config = config.get_pipeline_config(actual_provider_name)
        all_steps = actual_pipeline_config.get("steps", [])
        current_data = cleaned_data_after_parse
        for step_config in all_steps[1:]:
            current_data = self.run_pipeline_step(
                llm_manager, step_config, current_data, config.prompt_root
            )
        final_result = self.final_validation_and_cleaning(current_data)
        return final_result