"""
Template-driven summarization technique (Type 2).
This summarizer uses a detailed template with specific instructions.
"""
from typing import Dict, Any, Optional, cast
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from .base_summarizer import BaseSummarizer


class TemplateDrivenSummarizer(BaseSummarizer):
    """
    Implements Type 2 summarization: Template-Driven Summarization.
    
    This technique uses a more detailed template with specific instructions
    on how to summarize the content.
    """
    
    SYSTEM_PROMPT = """
    You are a highly skilled document summarizer.
    You will be given a document and you need to provide a concise summary.
    Make sure that the summary captures the main points of the document.
    """
    
    # Default template for template-driven summarization
    DEFAULT_TEMPLATE = """
    You are an expert document summarizer.
    Summary should strictly follow the output format below:
    Output Format:
    
    1. Main Idea
    2. Key Points (in bullet points)
    3. Important Details
    4. Action Items (if any)
    
    Please ensure the summary is clear, concise, and covers all the above sections.
    Document : {text}
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 **kwargs):
        """
        Initialize the TemplateDrivenSummarizer.
        
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
        return "Summarization using a detailed prompt template with specific instructions"

    @traceable(name="Template Driven Summarizer", run_type="llm", tags=["summarization", "template-driven"])
    def __call__(self, input_text: str, **kwargs) -> dict:
        """
        Perform template-driven summarization.
        
        Args:
            input_text: The text to summarize
            **kwargs: Additional parameters (template can be overridden)
            
        Returns:
            Dictionary with summary and metadata
        """
        template = kwargs.get("prompt_template", self.DEFAULT_TEMPLATE)
        if not isinstance(template, PromptTemplate):
            prompt_template = PromptTemplate.from_template(template)
        else:
            prompt_template = template
        prompt_text = prompt_template.format(text=input_text)
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(list, messages),
                temperature=self.model_params["temperature"],
                max_tokens=self.model_params["max_tokens"]
            )
            summary = response.choices[0].message.content
            if summary is None:
                summary = ""
            else:
                summary = summary.strip()
        except Exception as e:
            summary = f"Error: {e}"
        metrics = self.calculate_metrics(input_text, summary)
        return {
            "summary": summary,
            "metadata": metrics
        }

