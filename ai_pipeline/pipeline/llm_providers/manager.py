# ai_pipeline/pipeline/llm_providers/manager.py

import os
from typing import List, Dict, Any, Tuple
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .cloud_provider import CloudProvider
from .llama_provider import LlamaCppProvider

class LLMManager:
    def __init__(self, providers_config: List[Dict[str, Any]]):
        self.providers: List[LLMProvider] = []
        self.configs: List[Dict[str, Any]] = []
        self.used_provider_index = -1
        
        for config in providers_config:
            provider_type = config.get("type", "").lower()
            provider_name = config.get("name", provider_type)
            provider_config = config.copy()
            
            provider = self._create_provider(provider_type, provider_config)
            if provider:
                self.providers.append(provider)
                self.configs.append(provider_config)

    def _create_provider(self, provider_type: str, config: Dict[str, Any]) -> LLMProvider | None:
        try:
            if provider_type == "openai":
                return OpenAIProvider(config)
            elif provider_type == "cloud":
                return CloudProvider(config)
            elif provider_type == "local":
                return LlamaCppProvider(config)
            else:
                print(f"Warning: Unknown API type '{provider_type}'. Skipping.")
                return None
        except ImportError as e:
            print(f"Warning: Library for API type '{provider_type}' is not installed. {e}. Skipping.")
            return None
        except Exception as e:
            print(f"Warning: Could not initialize provider for '{provider_type}'. Error: {e}. Skipping.")
            return None

    def get_response(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        for i, (provider, config) in enumerate(zip(self.providers, self.configs)):
            api_type = config.get("type", "unknown")
            print(f"\n--- Attempting to connect to API #{i+1} ({api_type}) ---")
            try:
                response_text = provider.call(messages, config)
                if response_text:
                    self.used_provider_index = i
                    return response_text, i
            except Exception as e:
                print(f"\nError connecting to API #{i+1} ({api_type}): {e}")
                continue
        
        print("\nError: All API connection attempts failed.")
        return "", -1
    
    def get_used_provider(self) -> str:
        if self.used_provider_index >= 0 and self.used_provider_index < len(self.configs):
            return self.configs[self.used_provider_index].get("name", "unknown")
        return ""