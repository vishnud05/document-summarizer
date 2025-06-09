"""
Base class for document summarization techniques.
All specific summarization implementations should inherit from this class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from openai import OpenAI
from langsmith import wrappers, Client
from tiktoken import get_encoding
import os


class BaseSummarizer(ABC):
    """
    Abstract base class for all summarizers.
    
    This class provides:
    1. Centralized OpenAI client setup with LangSmith wrapping
    2. Common metric calculation utilities
    3. Interface definition for summarizer implementations
    
    Each subclass must implement its own __call__ method according to its specific technique.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 **kwargs):
        """
        Initialize a summarizer with common parameters.
        
        Args:
            model_name: The name of the language model to use
            api_key: API key (defaults to GROQ_API_KEY env var)
            base_url: Base URL for the API (defaults to Groq)
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum length of generated summary
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.name = self.__class__.__name__
        
        # Model parameters
        self.model_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Setup OpenAI client with LangSmith wrapper
        api_key = api_key or os.getenv('GROQ_API_KEY')
        base_url = base_url or "https://api.groq.com/openai/v1"
        
        if not api_key:
            raise ValueError("API key must be provided either directly or via GROQ_API_KEY environment variable")
        
        self.client = wrappers.wrap_openai(OpenAI(
            api_key=api_key,
            base_url=base_url,
        ))
        
        # Initialize LangSmith client for metrics logging
        self.langsmith_client = Client()
    
    def calculate_metrics(self, input_text: str, summary: str) -> Dict[str, Any]:
        """
        Calculate token usage and cost metrics.
        
        Args:
            input_text: Original input text
            summary: Generated summary
            
        Returns:
            Dictionary with metrics
        """
        # Token counting (using tiktoken or fallback to word count)
        try:
            enc = get_encoding("cl100k_base")
            input_tokens = len(enc.encode(input_text))
            output_tokens = len(enc.encode(summary))
        except Exception:
            input_tokens = len(input_text.split())
            output_tokens = len(summary.split())

        total_tokens = input_tokens + output_tokens
        # Example cost calculation (update with your model's pricing)
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
        """
        Returns a description of the summarization technique.
        This should be overridden by subclasses.
        """
        return "Base summarization technique"
    
    @abstractmethod
    def __call__(self, input_text: str, **kwargs) -> dict:
        """
        Main entry point for summarization.
        Each subclass must implement this method according to its specific technique.
        
        Args:
            input_text: The text to summarize
            **kwargs: Additional parameters
              Returns:
            Dictionary with summary and metadata
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the summarizer."""
        return f"{self.name}: {self.description}"