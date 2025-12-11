# ai_pipeline/pipeline/llm_providers/openai_provider.py

from typing import Dict, Any, List
from .base import LLMProvider

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class OpenAIProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gpt-4-turbo") 
        self.supports_system_prompt = config.get("supports_system_prompt", True)
        self.supports_json_mode = config.get("supports_json_mode", True)
        self.stream = config.get("default_stream", True)
    
    def call(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        if not OpenAI:
            raise ImportError("OpenAI library is not installed. Please install it with 'pip install openai'.")
        
        client = OpenAI(api_key=self.api_key)
        api_arguments = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True
        }

        if self.model.endswith("5.1"):
            api_arguments["max_completion_tokens"] = self.max_tokens
        else:
            api_arguments["max_tokens"] = self.max_tokens
            
        if self.stream:
            api_arguments["stream"] = True
        else:
            api_arguments["stream"] = False

        response = client.chat.completions.create(**api_arguments)
        response_text = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                response_text += content
                print(content, end="", flush=True)
        
        return response_text