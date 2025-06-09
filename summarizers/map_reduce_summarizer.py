"""
Map-reduce summarization (Chain_Type = Map_Reduce).
This summarizer splits the document using token-based chunking, summarizes each chunk, and combines the summaries.
"""
from typing import Optional, List, cast
from langsmith import traceable
import concurrent.futures
import tiktoken
from .base_summarizer import BaseSummarizer

MODEL_CONTEXT_WINDOWS = {
    "llama3-8b-8192": 8192,
    "compound-beta": 4000,
    "mixtral-8x7b-32768": 32768,
    "llama3-70b-8192": 8192,
}

DEFAULT_MODEL_CONFIG = {
    "model_name": "llama3-8b-8192",
    "max_input_tokens": MODEL_CONTEXT_WINDOWS.get("llama3-8b-8192", 4096),
    "max_output_tokens": 1000,
    "chunk_overlap": 100,  
    "max_workers": 5,
    "temperature": 0.1,  
    }

class MapReduceSummarizer(BaseSummarizer):
    """
    Implements Chain_Type = Map_Reduce summarization with parallel processing and token-based chunking.
    
    This technique splits the document into token-based chunks, summarizes each chunk separately
    in parallel using ThreadPoolExecutor, and then combines those summaries into a final summary.
    
    Key improvements:
    - Uses tiktoken for accurate token counting and chunking
    - Respects model context windows for optimal performance
    - Maintains chunk overlap for better context preservation
    """
    
    # Map phase prompt template for summarizing individual chunks
    MAP_TEMPLATE = """
    You are an expert document summarizer. Please provide a concise summary of the following text chunk:

    Text Chunk:
    {text}

    Summary:
    """
    
    # Reduce phase prompt template for combining summaries
    REDUCE_TEMPLATE = """
    You are an expert document summarizer. Please combine the following individual summaries into a single, comprehensive final summary:

    Individual Summaries:
    {summaries}

    Please create a cohesive final summary that captures all the main themes and key points from the individual summaries:
    """
    def __init__(self, 
                 model_name: str,
                 max_input_tokens: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 max_workers: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.1,  # Lower temperature for more consistent summaries
                 **kwargs):
        """
        Initialize the MapReduceSummarizer with token-based chunking.
        
        Args:
            model_name: Name of the pre-trained model to use for summarization
            max_input_tokens: Maximum tokens per chunk (uses model context window if not specified)
            chunk_overlap: Number of tokens to overlap between chunks
            max_workers: Maximum number of parallel workers for chunk processing
            api_key: API key (defaults to GROQ_API_KEY env var)
            base_url: Base URL for the API (defaults to Groq)
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum length of generated summary
            **kwargs: Additional model parameters
        """
        # Setup token-based chunking parameters
        self.max_input_tokens = max_input_tokens or DEFAULT_MODEL_CONFIG["max_input_tokens"]
        self.chunk_overlap = chunk_overlap or DEFAULT_MODEL_CONFIG["chunk_overlap"]
        self.max_workers = max_workers or DEFAULT_MODEL_CONFIG["max_workers"]
        
        # Reserve tokens for prompt overhead and response generation
        self.prompt_overhead = 200  # Estimated tokens for system/user prompts
        self.effective_chunk_size = self.max_input_tokens - self.prompt_overhead
        
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
            temperature=temperature,
            max_tokens=max_tokens or DEFAULT_MODEL_CONFIG["max_output_tokens"],
            **kwargs
        )
        
    @property
    def description(self) -> str:
        return "Summarization using LangChain's 'map_reduce' approach for handling longer documents"
    def _split_document_by_tokens(self, document: str) -> List[str]:
        """
        Split the document into token-based chunks using tiktoken.
        
        Args:
            document: The document content to split
            
        Returns:
            List of text chunks based on token limits
        """
        # Encode the entire document
        tokens = self.tokenizer.encode(document)
        total_tokens = len(tokens)
        
        print(f"📊 Document contains {total_tokens} tokens")
        print(f"🧩 Using chunk size of {self.effective_chunk_size} tokens with {self.chunk_overlap} token overlap")
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.effective_chunk_size, len(tokens))
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start index forward, accounting for overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - self.chunk_overlap
            
        print(f"🧩 Split into {len(chunks)} token-based chunks")
        
        # Log chunk sizes for debugging
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(self.tokenizer.encode(chunk))
            print(f"   Chunk {i+1}: {chunk_tokens} tokens")
        
        return chunks
    def _summarize_chunk(self, chunk: str, chunk_index: int = 0) -> str:
        """
        Summarize a single chunk of text (Map phase).
        
        Args:
            chunk: Text chunk to summarize
            chunk_index: Index of the chunk for logging purposes
            
        Returns:
            Summary of the chunk
        """
        # Format the map template with the chunk
        prompt = self.MAP_TEMPLATE.format(text=chunk)
        
        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text chunks concisely."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Use the centralized client with lower temperature for consistency
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(list, messages),
                temperature=0.1,
                max_tokens=300
            )
            
            # Extract summary from response
            summary = response.choices[0].message.content
            if summary is None:
                summary = ""
            return summary
        except Exception as e:
            print(f"❌ Error summarizing chunk {chunk_index + 1}: {e}")
            return f"Error summarizing chunk {chunk_index + 1}: {str(e)}"

    def _summarize_chunks_parallel(self, chunks: List[str], max_workers: int = 5) -> List[str]:
        """
        Summarize multiple chunks in parallel using ThreadPoolExecutor.
        
        Args:
            chunks: List of text chunks to summarize
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of chunk summaries in the same order as input chunks
        """
        print(f"🔄 Starting parallel processing with {max_workers} workers...")
        
        chunk_summaries = []
        
        # Use ThreadPoolExecutor for parallel API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:            # Submit all chunk summarization tasks
            future_to_index = {
                executor.submit(self._summarize_chunk, chunk, i): i 
                for i, chunk in enumerate(chunks)
            }
              # Initialize results list with empty strings
            results: List[str] = [""] * len(chunks)
            
            # Collect results as they complete
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_index):
                chunk_index = future_to_index[future]
                try:
                    summary = future.result()
                    results[chunk_index] = summary
                    completed_count += 1
                    print(f"✅ Chunk {chunk_index + 1}/{len(chunks)} completed ({completed_count}/{len(chunks)} total)")
                except Exception as e:
                    print(f"❌ Error processing chunk {chunk_index + 1}: {e}")
                    results[chunk_index] = f"Error processing chunk {chunk_index + 1}: {str(e)}"
                    completed_count += 1
          # Filter out empty values (shouldn't happen, but safety check)
        chunk_summaries = [summary for summary in results if summary != ""]
        
        print(f"🎉 Parallel processing complete - {len(chunk_summaries)} summaries generated")
        return chunk_summaries
    
    def _combine_summaries(self, chunk_summaries: List[str]) -> str:
        """
        Combine individual chunk summaries into a final summary (Reduce phase).
        
        Args:
            chunk_summaries: List of individual chunk summaries
            
        Returns:
            Final combined summary
        """
        # Join all summaries with newlines
        combined_summaries = "\n\n".join([f"Summary {i+1}: {summary}" 
                                        for i, summary in enumerate(chunk_summaries)])
        
        # Format the reduce template
        prompt = self.REDUCE_TEMPLATE.format(summaries=combined_summaries)
          # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that combines multiple summaries into a cohesive final summary."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Use the centralized client
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(list, messages),
                temperature=0.1,
                max_tokens=500
            )
            
            # Extract summary from response
            summary = response.choices[0].message.content
            if summary is None:
                summary = ""
            return summary
        except Exception as e:
            print(f"❌ Error combining summaries: {e}")
            return f"Error combining summaries: {str(e)}"
    @traceable(name="Map-Reduce Summarizer", run_type="llm", tags=["summarization", "map-reduce"])
    def __call__(self, input_text: str, **kwargs) -> dict:
        """
        Perform map-reduce summarization with parallel chunk processing.
        
        Args:
            input_text: The text to summarize
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with summary and metadata
        """
        import time
        start_time = time.time()
        print(f"🚀 Starting Map-Reduce Summarization...")
        print(f"📄 Document length: {len(input_text)} characters")
        print(f"🪙 Model: {self.model_name} (Max tokens: {self.max_input_tokens})")
        print(f"⚙️  Using {self.max_workers} parallel workers")
        
        # Step 1: Split the document into token-based chunks
        chunks = self._split_document_by_tokens(input_text)
        
        # Step 2: Map phase - summarize chunks in parallel
        print(f"🔄 Starting Map phase (parallel processing)...")
        chunk_summaries = self._summarize_chunks_parallel(chunks, self.max_workers)
        
        print(f"✅ Map phase complete - {len(chunk_summaries)} chunk summaries generated")
        
        # Step 3: Reduce phase - combine summaries
        print(f"🔄 Starting Reduce phase...")
        final_summary = self._combine_summaries(chunk_summaries)
        print(f"✅ Final summary generated")
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        
        # Calculate metrics using base class method
        metrics = self.calculate_metrics(input_text, final_summary)
          # Enhanced metadata with token-based chunking info
        metadata = {
            "num_chunks": len(chunks),
            "max_input_tokens": self.max_input_tokens,
            "effective_chunk_size": self.effective_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_workers": self.max_workers,
            "processing_time_seconds": processing_time,
            "individual_summaries": chunk_summaries,
            "chunking_method": "token-based",
            "tokenizer": "tiktoken",
            **metrics
        }
        
        return {
            "summary": final_summary,
            "metadata": metadata
        }



