"""
Factory for creating instances of different summarizer types.
"""
from typing import Dict, Type, List

from .base_summarizer import BaseSummarizer
from .basic_prompt_summarizer import BasicPromptSummarizer
from .template_driven_summarizer import TemplateDrivenSummarizer
from .structured_document_summarizer import StructuredDocumentSummarizer
from .map_reduce_summarizer import MapReduceSummarizer
from .refinement_based_summarizer import RefinementBasedSummarizer


class Summarizer:
    """
    Factory class for creating and managing summarizer instances.
    """
    
    # Registry of available summarizer types
    _summarizer_classes: Dict[str, Type[BaseSummarizer]] = {
        'basic_prompt': BasicPromptSummarizer,
        'template_driven': TemplateDrivenSummarizer,
        'structured_document': StructuredDocumentSummarizer,
        'map_reduce': MapReduceSummarizer,
        'refinement_based': RefinementBasedSummarizer,
    }
    
    @classmethod
    def get_summarizer(cls, summarizer_type: str, **kwargs) -> BaseSummarizer:
        """
        Get an instance of the requested summarizer type.
        
        Args:
            summarizer_type: Key identifying the summarizer type
            **kwargs: Additional parameters to pass to the summarizer constructor
            
        Returns:
            An instance of the requested summarizer type
            
        Raises:
            ValueError: If the summarizer type is not recognized
        """
        summarizer_class = cls._summarizer_classes.get(summarizer_type)
        if not summarizer_class:
            available_types = list(cls._summarizer_classes.keys())
            raise ValueError(
                f"Unknown summarizer type: {summarizer_type}. "
                f"Available types are: {available_types}"
            )
        
        return summarizer_class(**kwargs)
    
    @classmethod
    def available_summarizers(cls) -> List[Dict[str, str]]:
        """
        Get a list of all available summarizer types with descriptions.
        
        Returns:
            List of dictionaries containing 'id', 'name', and 'description' for each summarizer
        """
        result = []
        for key, summarizer_class in cls._summarizer_classes.items():
            # Create a temporary instance to get the description
            instance = summarizer_class()
            result.append({
                'id': key,
                'name': instance.name,
                'description': instance.description
            })
        
        return result
