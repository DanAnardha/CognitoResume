# ai_pipeline/pipeline/parse/config.py

import json as std_json  # Use alias to avoid potential name collisions
import os
from typing import Any, Dict, Union, List

class Config:
    def __init__(
        self,
        parse_config: Union[str, dict, None] = "ai_pipeline/pipeline/parse/config.json",
        global_providers: Union[str, dict, None] = "ai_pipeline/pipeline/config/global_providers.json",
        override_config: Union[dict, None] = None
    ):

        # Load base parse config
        if isinstance(parse_config, dict):
            parse_config_data = parse_config
        elif isinstance(parse_config, str):
            parse_config_data = self._load_config_from_file(parse_config)
        elif parse_config is None:
            parse_config_data = {}
        else:
            raise TypeError("Invalid type for parse_config")

        # Load global providers
        if isinstance(global_providers, dict):
            providers_config = global_providers
        elif isinstance(global_providers, str):
            providers_config = self._load_config_from_file(global_providers)
        elif global_providers is None:
            providers_config = {}
        else:
            raise TypeError("Invalid type for global_providers")

        # Base merge = parse config + global providers
        merged = self._merge_configs(parse_config_data, providers_config)

        # Apply override if exists (highest priority)
        if override_config:
            merged = self._deep_merge(merged, override_config)

        self._merged_config = merged
        
    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return std_json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found.")
            return {}
        except std_json.JSONDecodeError:
            print(f"Error: Configuration file '{config_path}' is invalid.")
            return {}
    
    def _get_env_var(self, env_var: str, default: Any = None) -> Any:
        return os.environ.get(env_var, default)
    
    def _merge_configs(self, parse_config: Dict[str, Any], global_providers_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "version": parse_config.get("version", "1.0.0"),
            "prompt_root": parse_config.get("prompt_root", "ai_pipeline/pipeline/parse/prompts"),
            "pipelines": parse_config.get("pipelines", {}),
            "active": global_providers_config.get("active", {}),
            "providers": global_providers_config.get("providers", {})
        }
        
        if "providers" in merged:
            for provider_name, provider_config in merged["providers"].items():
                keys_to_process = list(provider_config.keys())
                for key in keys_to_process:
                    if key.endswith("_env"):
                        env_key = key[:-4]
                        env_var_name = provider_config[key]
                        env_value = self._get_env_var(env_var_name)
                        if env_value is not None:
                            provider_config[env_key] = env_value
                        del provider_config[key]
        return merged

    def _deep_merge(self, source_dict: Dict[str, Any], overrides_dict: Dict[str, Any]) -> Dict[str, Any]:
        result = source_dict.copy()
        for key, value in overrides_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @property
    def prompt_root(self) -> str:
        return self._merged_config.get("prompt_root", "ai_pipeline/pipeline/parse/prompts")
    
    @property
    def active_provider(self) -> str:
        return self._merged_config.get("active", {}).get("primary", "openai")
    
    @property
    def fallback_providers(self) -> List[str]:
        return self._merged_config.get("active", {}).get("fallback", [])
    
    @property
    def use_fallback(self) -> bool:
        return self._merged_config.get("active", {}).get("use_fallback", True)
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        return self._merged_config.get("providers", {}).get(provider_name, {})
    
    def get_pipeline_config(self, provider_name: str) -> Dict[str, Any]:
        return self._merged_config.get("pipelines", {}).get(provider_name, {})
    
    def get_all_providers_config(self) -> List[Dict[str, Any]]:
        providers_config = []
        
        # Add primary provider
        primary_provider = self.active_provider
        primary_config = self.get_provider_config(primary_provider)
        if primary_config:
            primary_config["name"] = primary_provider
            providers_config.append(primary_config)
        
        # Add fallback providers if enabled
        if self.use_fallback:
            for fallback_provider in self.fallback_providers:
                fallback_config = self.get_provider_config(fallback_provider)
                if fallback_config:
                    fallback_config["name"] = fallback_provider
                    providers_config.append(fallback_config)
        return providers_config

    def get(self, key: str, default: Any = None) -> Any:
        return self._merged_config.get(key, default)