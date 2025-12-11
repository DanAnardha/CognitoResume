# ai_pipeline/pipeline/skill_match/config.py

import json
from typing import Any, Dict, Union

class Config:
    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        if isinstance(config_source, str):
            self._config_data = self._load_config_from_file(config_source)
        elif isinstance(config_source, dict):
            self._config_data = config_source
        else:
            raise TypeError("config_source must be a file path (str) or a dictionary.")
        
        if not self._config_data:
            raise ValueError("Configuration could not be loaded. Exiting.")
        
    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config from '{config_path}': {e}")
            return {}
    
    @property
    def version(self) -> str:
        return self._config_data.get("version", "1.0.0")
    
    @property
    def model_settings(self) -> Dict[str, Any]:
        return self._config_data.get("model_settings", {})
    
    @property
    def scoring_weights(self) -> Dict[str, Any]:
        return self._config_data.get("scoring_weights", {})
    
    @property
    def skill_thresholds(self) -> Dict[str, Any]:
        return self._config_data.get("skill_thresholds", {})
    
    @property
    def similarity_weights(self) -> Dict[str, Any]:
        return self._config_data.get("similarity_weights", {})
        
    @property
    def normalization(self) -> Dict[str, Any]:
        return self._config_data.get("normalization", {})
    
    @property
    def synonym_file(self) -> str:
        return self._config_data.get("synonym_file", "")
        
    @property
    def acronym_file(self) -> str:
        return self._config_data.get("acronym_file", "")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config_data.get(key, default)