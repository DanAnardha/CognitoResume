# ai_pipeline/pipeline/llm_providers/llama_provider.py

import json
import threading
from pathlib import Path
from typing import Dict, Any, List
from .base import LLMProvider

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class LlamaCppProvider(LLMProvider):
    _llama_instance = None
    _init_lock = threading.Lock()
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get("model_path", "")
        self.n_ctx = config.get("n_ctx", 2048)
        self.n_threads = config.get("n_threads", 4)
        self.n_gpu_layers = config.get("n_gpu_layers", 0)
        self.verbose = config.get("verbose", False)
        self.supports_system_prompt = config.get("supports_system_prompt", True)
        self.supports_json_mode = config.get("supports_json_mode", True)
        self.stream = config.get("default_stream", True)
        self._init_model()
    
    def _init_model(self):
        if not Llama:
            raise ImportError("llama-cpp-python library is not installed. Please install it with 'pip install llama-cpp-python'.")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'")
        
        with LlamaCppProvider._init_lock:
            if LlamaCppProvider._llama_instance is None:
                print(f"--- Loading Llama model for the first time. This may take a while... ---")
                LlamaCppProvider._llama_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose
                )
                print("--- Llama model loaded and cached. ---")
        self.llm = LlamaCppProvider._llama_instance
    
    def call(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response_text = ""
        try:
            print("\n--- Calling local Llama model. This may take some time for complex prompts... ---")
            
            response_stream = self.llm(
                prompt, 
                max_tokens=self.max_tokens, 
                temperature=self.temperature, 
                stop=["</s>", "user:", "assistant:"],
                stream=self.stream
            )
            
            if self.stream:
                print("--- Model is generating response... ---")
                for chunk in response_stream:
                    choices = chunk.get('choices', [])
                    if choices:
                        content = choices[0].get('text', '')
                        if content:
                            print(content, end="", flush=True)
                            response_text += content
                print("\n--- Local Llama model finished generating. ---")
            else:
                response_text = response_stream["choices"][0]["text"]
                print(response_text)
            
        except Exception as e:
            print(f"\n--- ERROR during Llama call: {e} ---")
            raise
        
        return response_text