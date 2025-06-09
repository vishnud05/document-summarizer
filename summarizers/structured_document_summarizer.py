"""
Structured document summarization (Chain_Type = Stuff).
This summarizer uses LangChain's stuff approach to summarize a document.
"""
from typing import Dict, Any, Optional
from .base_summarizer import BaseSummarizer


class StructuredDocumentSummarizer(BaseSummarizer):
    """
    Implements Chain_Type = Stuff summarization.
    
    This technique uses LangChain's stuff approach to summarize the entire
    document in one go.
    """
    
    @property
    def description(self) -> str:
        return "Summarization using LangChain's 'stuff' approach for structured documents"
    