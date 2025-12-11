# ai_pipeline/pipeline/llm_providers/cloud_providers.py

import json
import requests
from typing import Dict, Any, List
from .base import LLMProvider

class CloudProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", "").rstrip('/')
        self.timeout = config.get("timeout", 60)
        self.supports_system_prompt = config.get("supports_system_prompt", True)
        self.supports_json_mode = config.get("supports_system_prompt", True)
        self.stream = config.get("default_stream", True)
    
    def call(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        full_url = f"{self.url}/chat/stream"
        
        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        if self.stream:
            payload["stream"] = True
            response_text = ""
            with requests.post(full_url, json=payload, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            json_data = json.loads(decoded_line[len("data: "):])
                            content = json_data.get('content', '')
                            if content:
                                response_text += content
                                print(content, end="", flush=True)
        else:
            payload["stream"] = False
            response = requests.post(full_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(response_text)
        
        return response_text