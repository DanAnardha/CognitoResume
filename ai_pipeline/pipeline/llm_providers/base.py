# ai_pipeline/pipeline/llm_providers/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class LLMProvider(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.temperature = self.config.get("default_temperature", 0.0)
        self.max_tokens = self.config.get("max_tokens", 4096)
    
    @abstractmethod
    def call(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        pass