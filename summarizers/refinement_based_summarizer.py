"""
Refinement-based summarization (Chain_Type = Refine).
This summarizer iteratively refines the summary as it processes chunks of the document.
"""
from typing import Dict, Any, Optional, List, cast
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
import tiktoken
from .base_summarizer import BaseSummarizer

MODEL_CONTEXT_WINDOWS = {
    "llama3-8b-8192": 8192,
    "gemma2-9b-it": 8192,
    "mixtral-8x7b-32768": 32768,
    "llama3-70b-8192": 8192,
}

DEFAULT_MODEL_CONFIG = {
    "model_name": "llama3-8b-8192",
    "max_input_tokens": MODEL_CONTEXT_WINDOWS.get("llama3-8b-8192", 4096),
    "max_output_tokens": 800,
    "chunk_overlap": 150,  # Token overlap for better context preservation
    "temperature": 0.3,  # Moderate temperature for consistent but creative refinement
}


class RefinementBasedSummarizer(BaseSummarizer):
    """
    Implements Chain_Type = Refine summarization with iterative summary refinement.
    
    This technique processes the document in token-based chunks, starting with an initial 
    summary from the first chunk and then iteratively refining it by incorporating 
    information from subsequent chunks. Each refinement step improves clarity, accuracy, 
    and comprehensiveness of the summary.
    
    Key features:
    - Token-based chunking for optimal model performance
    - Sequential processing with iterative refinement
    - Specialized prompts for initial summary and refinement steps
    - Comprehensive progress tracking and error handling
    """
    
    # Initial summary template for the first chunk
    INITIAL_SUMMARY_TEMPLATE = """
    You are an expert document summarizer. Please provide a comprehensive initial summary of the following text chunk.
    This will be the foundation for further refinement as more content is processed.
    
    Focus on:
    - Main themes and key points
    - Important facts and details
    - Overall context and purpose
    
    Text Chunk:
    {text}
    
    Initial Summary:
    """
    
    # Refinement template for subsequent chunks
    REFINEMENT_TEMPLATE = """
    You are an expert document summarizer. You have an existing summary that needs to be refined and enhanced with new information.
    
    Your task:
    1. Review the existing summary and the new text chunk
    2. Integrate relevant new information from the text chunk
    3. Improve clarity, accuracy, and comprehensiveness
    4. Maintain coherence and flow in the refined summary
    5. Remove any redundancy while preserving important details
    
    Existing Summary:
    {existing_summary}
    
    New Text Chunk:
    {new_text}
    
    Please provide a refined summary that incorporates the new information while improving the overall quality:
    
    Refined Summary:
    """
    
    def __init__(self, 
                 model_name: str,
                 max_input_tokens: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: Optional[float] = None,  # Moderate temperature for refinement
                 **kwargs):
        """
        Initialize the RefinementBasedSummarizer with token-based chunking.
        
        Args:
            model_name: Name of the pre-trained model to use for summarization
            max_input_tokens: Maximum tokens per chunk (uses model context window if not specified)
            chunk_overlap: Number of tokens to overlap between chunks
            max_tokens: Maximum length of generated summary
            api_key: API key (defaults to GROQ_API_KEY env var)
            base_url: Base URL for the API (defaults to Groq)
            temperature: Controls randomness in the response (0.0 to 1.0)
            **kwargs: Additional model parameters
        """
        # Setup token-based chunking parameters
        self.max_input_tokens = max_input_tokens or MODEL_CONTEXT_WINDOWS.get(model_name, 4096)
        self.chunk_overlap = chunk_overlap or DEFAULT_MODEL_CONFIG["chunk_overlap"]
        
        # Reserve tokens for prompt overhead and existing summary in refinement
        self.initial_prompt_overhead = 150  # For initial summary prompt
        self.refinement_prompt_overhead = 300  # For refinement prompt + existing summary
        self.initial_chunk_size = self.max_input_tokens - self.initial_prompt_overhead
        self.temperature = temperature or DEFAULT_MODEL_CONFIG["temperature"]
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback encoding if cl100k_base is not available
            self.tokenizer = tiktoken.get_encoding("p50k_base")
            
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=self.temperature,
            max_tokens=max_tokens or DEFAULT_MODEL_CONFIG["max_output_tokens"],
            **kwargs
        )
        
    @property
    def description(self) -> str:
        return "Iterative summary refinement using token-based chunking for enhanced quality and coherence"
    
    def _split_document_by_tokens(self, document: str) -> List[str]:
        """
        Split the document into token-based chunks for sequential processing.
        
        Args:
            document: The document content to split
            
        Returns:
            List of text chunks based on token limits
        """
        # Encode the entire document
        tokens = self.tokenizer.encode(document)
        total_tokens = len(tokens)
        
        print(f"📊 Document contains {total_tokens} tokens")
        print(f"🧩 Using initial chunk size of {self.initial_chunk_size} tokens with {self.chunk_overlap} token overlap")
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # For the first chunk, use initial_chunk_size
            # For subsequent chunks, account for existing summary in refinement prompt
            if len(chunks) == 0:
                chunk_size = self.initial_chunk_size
            else:
                # Estimate existing summary tokens and adjust chunk size
                estimated_summary_tokens = min(800, len(chunks) * 100)  # Conservative estimate
                available_tokens = self.max_input_tokens - self.refinement_prompt_overhead - estimated_summary_tokens
                chunk_size = max(500, available_tokens)  # Minimum chunk size of 500 tokens
            
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start index forward, accounting for overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - self.chunk_overlap
            
        print(f"🧩 Split into {len(chunks)} token-based chunks for sequential refinement")
        
        # Log chunk sizes for debugging
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(self.tokenizer.encode(chunk))
            print(f"   Chunk {i+1}: {chunk_tokens} tokens")
        
        return chunks
    
    def _generate_initial_summary(self, first_chunk: str) -> str:
        """
        Generate the initial summary from the first chunk of the document.
        
        Args:
            first_chunk: The first text chunk to summarize
            
        Returns:
            Initial summary text
        """
        print(f"🚀 Generating initial summary from first chunk...")
        
        # Format the initial summary template
        prompt = self.INITIAL_SUMMARY_TEMPLATE.format(text=first_chunk)
        
        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": "You are an expert document summarizer specializing in creating comprehensive initial summaries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(list, messages),
                **self.model_params
            )
            
            # Extract summary from response
            summary = response.choices[0].message.content
            if summary is None:
                summary = ""
            
            print(f"✅ Initial summary generated ({len(summary)} characters)")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating initial summary: {e}")
            return f"Error generating initial summary: {str(e)}"
    
    def _refine_summary_with_chunk(self, existing_summary: str, new_chunk: str, chunk_index: int) -> str:
        """
        Refine the existing summary by incorporating information from a new chunk.
        
        Args:
            existing_summary: The current summary to be refined
            new_chunk: New text chunk to incorporate
            chunk_index: Index of the current chunk (for logging)
            
        Returns:
            Refined summary text
        """
        print(f"🔄 Refining summary with chunk {chunk_index + 1}...")
        
        # Format the refinement template
        prompt = self.REFINEMENT_TEMPLATE.format(
            existing_summary=existing_summary,
            new_text=new_chunk
        )
        
        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": "You are an expert document summarizer specializing in iterative summary refinement."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(list, messages),
                **self.model_params
            )
            
            # Extract refined summary from response
            refined_summary = response.choices[0].message.content
            if refined_summary is None:
                refined_summary = existing_summary  # Fallback to existing summary
            
            print(f"✅ Summary refined with chunk {chunk_index + 1} ({len(refined_summary)} characters)")
            return refined_summary
            
        except Exception as e:
            print(f"❌ Error refining summary with chunk {chunk_index + 1}: {e}")
            print(f"   Continuing with existing summary...")
            return existing_summary  # Return existing summary on error
    
    def _process_refinement_chain(self, chunks: List[str]) -> str:
        """
        Process the refinement chain by sequentially refining the summary with each chunk.
        
        Args:
            chunks: List of text chunks to process sequentially
            
        Returns:
            Final refined summary
        """
        if not chunks:
            return "No content to summarize."
        
        print(f"🔄 Starting refinement chain with {len(chunks)} chunks...")
        
        # Step 1: Generate initial summary from the first chunk
        current_summary = self._generate_initial_summary(chunks[0])
        
        # Step 2: Iteratively refine with remaining chunks
        if len(chunks) > 1:
            print(f"🔄 Beginning iterative refinement with {len(chunks) - 1} additional chunks...")
            
            for i, chunk in enumerate(chunks[1:], start=1):
                print(f"📝 Processing refinement iteration {i}/{len(chunks) - 1}")
                current_summary = self._refine_summary_with_chunk(current_summary, chunk, i)
                
                # Log progress
                summary_tokens = len(self.tokenizer.encode(current_summary))
                print(f"   Current summary: {summary_tokens} tokens, {len(current_summary)} characters")
        
        print(f"✅ Refinement chain complete - Final summary generated")
        return current_summary
    
    @traceable(name="Refinement-Based Summarizer", run_type="llm", tags=["summarization", "refinement-based"])
    def __call__(self, input_text: str, **kwargs) -> dict:
        """
        Perform refinement-based summarization with iterative improvement.
        
        Args:
            input_text: The text to summarize
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with summary and metadata (same format as BasicPromptSummarizer)
        """
        import time
        start_time = time.time()
        
        print(f"🚀 Starting Refinement-Based Summarization...")
        print(f"📄 Document length: {len(input_text)} characters")
        print(f"🪙 Model: {self.model_name} (Max tokens: {self.max_input_tokens})")
        print(f"🌡️  Temperature: {self.temperature}")
        
        # Step 1: Split the document into token-based chunks
        chunks = self._split_document_by_tokens(input_text)
        
        # Step 2: Process refinement chain
        final_summary = self._process_refinement_chain(chunks)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        
        # Calculate metrics using base class method
        metrics = self.calculate_metrics(input_text, final_summary)
        
        # Enhanced metadata (matching BasicPromptSummarizer format but with refinement-specific info)
        metadata = {
            "num_chunks": len(chunks),
            "max_input_tokens": self.max_input_tokens,
            "initial_chunk_size": self.initial_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "refinement_iterations": len(chunks) - 1 if len(chunks) > 1 else 0,
            "processing_time_seconds": processing_time,
            "chunking_method": "token-based",
            "tokenizer": "tiktoken",
            "refinement_approach": "sequential",
            **metrics
        }
        
        print(f"✅ Refinement-based summary generated")
        print(f"📊 Summary Length: {len(final_summary)} characters")
        print(f"🔄 Refinement Iterations: {metadata['refinement_iterations']}")
        
        return {
            "summary": final_summary,
            "metadata": metadata
        }

