#!/usr/bin/env python3
"""
Streamlit Web Interface for Document Summarization
"""

import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv

# Import summarizers
from summarizers.basic_prompt_summarizer import BasicPromptSummarizer
from summarizers.template_driven_summarizer import TemplateDrivenSummarizer
from summarizers.map_reduce_summarizer import MapReduceSummarizer
from summarizers.refinement_based_summarizer import RefinementBasedSummarizer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .technique-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .parameter-section {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-section {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file based on file type."""
    try:
        file_type = uploaded_file.type
        
        if file_type == "text/plain":
            # Handle .txt files
            content = uploaded_file.read().decode('utf-8')
            return content
            
        elif file_type == "application/pdf":
            # Handle .pdf files
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text.strip()
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Handle .docx files
            doc = Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
            
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def get_technique_description(technique):
    """Get description for each summarization technique."""
    descriptions = {
        "Basic Prompt": "Simple prompt-based summarization using a single LLM call. Best for short to medium documents.",
        "Template Driven": "Uses structured templates to guide the summarization process. Provides more consistent output format.",
        "Map-Reduce": "Divides large documents into chunks, summarizes each chunk separately, then combines results. Best for very long documents.",
        "Refinement Based": "Iteratively refines summaries by processing document chunks sequentially. Produces high-quality summaries through multiple passes."
    }
    return descriptions.get(technique, "")

def display_parameter_controls(technique):
    """Display parameter controls based on selected technique."""
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.subheader("🎛️ Parameter Settings")
    
    params = {}
    
    if technique == "Basic Prompt":
        col1, col2 = st.columns(2)
        with col1:
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                                            help="Controls randomness in output. Lower = more focused, Higher = more creative")
        with col2:
            params['max_tokens'] = st.number_input("Max Output Tokens", 50, 1000, 300, 50,
                                                 help="Maximum length of the generated summary")
    
    elif technique == "Template Driven":
        col1, col2 = st.columns(2)
        with col1:
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.6, 0.1)
        with col2:
            params['max_tokens'] = st.number_input("Max Output Tokens", 50, 1000, 400, 50)
    
    elif technique == "Map-Reduce":
        col1, col2 = st.columns(2)
        with col1:
            params['max_input_tokens'] = st.number_input("Max Input Tokens per Chunk", 1000, 8000, 5000, 500,
                                                       help="Size of each chunk for processing")
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
        with col2:
            params['max_tokens'] = st.number_input("Max Output Tokens", 50, 1000, 500, 50)
            params['max_workers'] = st.number_input("Max Workers", 1, 8, 4, 1,
                                                  help="Number of parallel workers for processing")
    
    elif technique == "Refinement Based":
        col1, col2 = st.columns(2)
        with col1:
            params['max_input_tokens'] = st.number_input("Max Input Tokens per Chunk", 1000, 8000, 4000, 500,
                                                       help="Size of each chunk for processing")
            params['chunk_overlap'] = st.number_input("Chunk Overlap", 0, 500, 150, 25,
                                                    help="Number of overlapping tokens between chunks")
        with col2:
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            params['max_tokens'] = st.number_input("Max Output Tokens", 50, 1000, 600, 50)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return params

def create_summarizer(technique, params):
    """Create and return the appropriate summarizer based on technique and parameters."""
    # Use a fixed model for consistency
    model_name = "llama3-8b-8192"  # You can change this to your preferred model
    
    if technique == "Basic Prompt":
        return BasicPromptSummarizer(
            model_name=model_name,
            temperature=params.get('temperature', 0.7),
            max_tokens=params.get('max_tokens', 300)
        )
    
    elif technique == "Template Driven":
        return TemplateDrivenSummarizer(
            model_name=model_name,
            temperature=params.get('temperature', 0.6),
            max_tokens=params.get('max_tokens', 400)
        )
    
    elif technique == "Map-Reduce":
        return MapReduceSummarizer(
            model_name=model_name,
            max_input_tokens=params.get('max_input_tokens', 5000),
            temperature=params.get('temperature', 0.5),
            max_tokens=params.get('max_tokens', 500),
            max_workers=params.get('max_workers', 4)
        )
    
    elif technique == "Refinement Based":
        return RefinementBasedSummarizer(
            model_name=model_name,
            max_input_tokens=params.get('max_input_tokens', 4000),
            chunk_overlap=params.get('chunk_overlap', 150),
            temperature=params.get('temperature', 0.3),
            max_tokens=params.get('max_tokens', 600)
        )
    else:
        raise ValueError(f"Unknown summarization technique: {technique}")

def display_results(result, technique):
    """Display summarization results."""
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.subheader("📄 Generated Summary")
    
    summary = result['summary']
    metadata = result['metadata']
    
    # Display summary
    st.text_area("Summary", summary, height=200, help="Your generated summary")
    
    # Display metadata in expandable section
    with st.expander("📊 Detailed Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Summary Length", f"{len(summary)} chars")
            st.metric("Input Tokens", metadata.get('input_tokens', 'N/A'))
            st.metric("Output Tokens", metadata.get('output_tokens', 'N/A'))
        
        with col2:
            st.metric("Cost", f"${metadata.get('cost', 0):.4f}")
            st.metric("Compression Ratio", f"{metadata.get('compression_ratio', 0):.2f}x")
            if 'processing_time_seconds' in metadata:
                st.metric("Processing Time", f"{metadata['processing_time_seconds']:.2f}s")
        
        with col3:
            if technique in ["Map-Reduce", "Refinement Based"]:
                st.metric("Number of Chunks", metadata.get('num_chunks', 'N/A'))
                if technique == "Refinement Based":
                    st.metric("Refinement Iterations", metadata.get('refinement_iterations', 'N/A'))
                if technique == "Map-Reduce":
                    st.metric("Max Workers", metadata.get('max_workers', 'N/A'))
    
    # Download button
    summary_filename = f"{technique.lower().replace(' ', '_')}_summary.txt"
    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name=summary_filename,
        mime="text/plain",
        help="Download the generated summary as a text file"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Document Summarizer</h1>
        <p>Upload your document and choose from multiple AI-powered summarization techniques</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not os.getenv('GROQ_API_KEY'):
        st.error("❌ GROQ_API_KEY environment variable not set! Please set your GROQ API key.")
        st.stop()
    
    # Sidebar for file upload and technique selection
    with st.sidebar:
        st.header("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx'],
            help="Upload a text file (.txt), PDF (.pdf), or Word document (.docx)"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size:,} bytes")
        
        st.header("🎯 Summarization Technique")
        technique = st.selectbox(
            "Choose technique:",
            ["Basic Prompt", "Template Driven", "Map-Reduce", "Refinement Based"],
            help="Select the summarization approach"
        )
        
        # Display technique description
        if technique:
            st.markdown(f'<div class="technique-card">', unsafe_allow_html=True)
            st.markdown(f"**{technique}**")
            st.markdown(get_technique_description(technique))
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if uploaded_file and technique:
        # Extract text from uploaded file
        with st.spinner("📖 Extracting text from document..."):
            document_text = extract_text_from_file(uploaded_file)
        
        if document_text:
            # Display document info
            st.subheader("📖 Document Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{len(document_text):,}")
            with col2:
                st.metric("Words", f"{len(document_text.split()):,}")
            with col3:
                st.metric("File Type", uploaded_file.type.split('/')[-1].upper())
            
            # Display document preview
            with st.expander("👀 Document Preview"):
                preview_text = document_text[:1000] + "..." if len(document_text) > 1000 else document_text
                st.text_area("Document Content", preview_text, height=200)
            
            # Parameter controls
            params = display_parameter_controls(technique)
            
            # Generate summary button
            if st.button("🚀 Generate Summary", type="primary"):
                try:
                    with st.spinner(f"🔄 Generating summary using {technique}..."):
                        # Create summarizer
                        summarizer = create_summarizer(technique, params)
                        
                        # Generate summary
                        start_time = time.time()
                        result = summarizer(document_text)
                        end_time = time.time()
                        
                        # Add processing time to metadata if not already present
                        if 'processing_time_seconds' not in result['metadata']:
                            result['metadata']['processing_time_seconds'] = end_time - start_time
                        
                        # Store result in session state
                        st.session_state['summary_result'] = result
                        st.session_state['technique_used'] = technique
                        
                        st.success(f"✅ Summary generated successfully using {technique}!")
                        
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
                    st.exception(e)
            
            # Display results if available
            if 'summary_result' in st.session_state and 'technique_used' in st.session_state:
                display_results(st.session_state['summary_result'], st.session_state['technique_used'])
        
        else:
            st.error("❌ Could not extract text from the uploaded file. Please try a different file.")
    
    elif not uploaded_file:
        st.info("👆 Please upload a document to get started.")
    
    else:
        st.info("👆 Please select a summarization technique.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Powered by Groq LLM API | Built with Streamlit</p>
        <p>Supports: Text (.txt), PDF (.pdf), Word (.docx) files</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
