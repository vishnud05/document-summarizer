"""
Basic prompt summarization technique (Type 1).
This summarizer uses LangChain with Hugging Face Inference API for summarization.
Integrated with LangSmith for tracing and evaluation.
"""

from typing import Dict, Any, Optional, cast
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from .base_summarizer import BaseSummarizer


class BasicPromptSummarizer(BaseSummarizer):
    """
    Implements Type 1 summarization: Basic Prompt Summarization.
    
    This technique uses a simple prompt with LangChain and the Hugging Face
    Inference API to summarize text.
    """
    
    # Default prompt template for summarization
    SYSTEM_PROMPT = """
    You are a helpful assistant that summarizes text.
    You will be given a document and you need to provide a concise summary.
    Make sure that the summary captures the main points of the document.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 **kwargs):
        """
        Initialize the BasicPromptSummarizer.
        
        Args:
            model_name: Name of the pre-trained model to use for summarization
            api_key: API key (defaults to GROQ_API_KEY env var)
            base_url: Base URL for the API (defaults to Groq)
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum length of the generated summary
            **kwargs: Additional model parameters
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    @property
    def description(self) -> str:
        return "Basic summarization using LangChain with a simple prompt template"

    @traceable(name="Basic Prompt Summarizer", run_type="llm", tags=["summarization", "basic-prompt"])    
    def __call__(self, input_text: str, **kwargs) -> dict:
        """
        Perform basic prompt summarization.
        
        Args:
            input_text: The text to summarize
            **kwargs: Additional parameters (prompt_template can be overridden)
        
        Returns:
            Dictionary with summary and metadata
        """
        prompt = kwargs.get("prompt")
        if prompt is not None:
            prompt_template = prompt
        else:
            prompt_template = PromptTemplate.from_template("Summarize the following text:\n{text}")
        input_prompt = prompt_template.format(text=input_text)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": input_prompt}
        ]
        try:
            response = self.client.chat.completions.create(model=self.model_name, messages=cast(list, messages), temperature=self.model_params["temperature"], max_tokens=self.model_params["max_tokens"])
            summary = response.choices[0].message.content
            if summary is None:
                summary = ""
            else:
                summary = summary.strip()
        except Exception as e:
            summary = f"Error: {e}"
        metrics = self.calculate_metrics(input_text, summary)
        return {"summary": summary, "metadata": metrics}
