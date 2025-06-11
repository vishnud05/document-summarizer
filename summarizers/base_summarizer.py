from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from openai import OpenAI
from langsmith import wrappers, Client
from tiktoken import get_encoding
import os


class BaseSummarizer(ABC):
    def __init__(self, 
                 model_name: str = "llama3-8b-8192",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 **kwargs):
        self.model_name = "meta/llama-4-maverick-17b-128e-instruct" or model_name
        self.name = self.__class__.__name__
        
        self.model_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        api_key = api_key or os.getenv('NVIDIA_API_KEY')
        base_url = base_url or "https://integrate.api.nvidia.com/v1"
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        
        self.client = wrappers.wrap_openai(OpenAI(
            api_key=api_key,
            base_url=base_url,
        ))
        
        self.langsmith_client = Client()
    
    def calculate_metrics(self, input_text: str, summary: str) -> Dict[str, Any]:
        try:
            enc = get_encoding("cl100k_base")
            input_tokens = len(enc.encode(input_text))
            output_tokens = len(enc.encode(summary))
        except Exception:
            input_tokens = len(input_text.split())
            output_tokens = len(summary.split())
        
        total_tokens = input_tokens + output_tokens
        cost = total_tokens / 1000 * 0.001

        return {
            "technique": self.name,
            "model": self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "input_length": len(input_text),
            "output_length": len(summary),
            "model_params": self.model_params
        }
        
    @property
    def description(self) -> str:
        return "Base summarizer class."
    
    @abstractmethod
    def __call__(self, input_text: str, **kwargs) -> dict:
        pass
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"