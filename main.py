#!/usr/bin/env python3
"""
Command Line Interface for Document Summarization
"""

import sys
import os
from dotenv import load_dotenv
from summarizers.basic_prompt_summarizer import BasicPromptSummarizer
from summarizers.template_driven_summarizer import TemplateDrivenSummarizer
from summarizers.map_reduce_summarizer import MapReduceSummarizer
from summarizers.refinement_based_summarizer import RefinementBasedSummarizer

load_dotenv()

def main():
    if not os.getenv('GROQ_API_KEY'):
        print("GROQ_API_KEY environment variable not set! Please set your GROQ API key before running this demo.")
        sys.exit(1)
    
    doc_file = "doc.txt"
    if not os.path.exists(doc_file):
        print(f"Document file '{doc_file}' not found! Please create a 'doc.txt' file in the root directory.")
        sys.exit(1)
    
    print("Document Summarization CLI Demo\n" + "=" * 50)
    
    with open(doc_file, "r", encoding='utf-8') as file:
        document = file.read()
    
    print(f"Document loaded: {len(document)} characters\n" + "=" * 50)
    
    basic_summarizer = BasicPromptSummarizer(model_name="llama3-8b-8192")
    basic_result = basic_summarizer(document)
    print(f"Basic Summary:\n{basic_result['summary']}\nCost: ${basic_result['metadata']['cost']:.4f}\n" + "=" * 50)
    
    template_summarizer = TemplateDrivenSummarizer(model_name="llama3-8b-8192")
    template_result = template_summarizer(document)
    print(f"Template Summary:\n{template_result['summary']}\nCost: ${template_result['metadata']['cost']:.4f}\n" + "=" * 50)
    
    map_reduce_summarizer = MapReduceSummarizer(model_name="llama3-8b-8192", max_input_tokens=5000)
    map_reduce_result = map_reduce_summarizer(document)
    print(f"Map-Reduce Summary:\n{map_reduce_result['summary']}\nCost: ${map_reduce_result['metadata']['cost']:.4f}\nChunks: {map_reduce_result['metadata']['num_chunks']}\n" + "=" * 50)
    
    refinement_summarizer = RefinementBasedSummarizer(model_name="llama3-8b-8192", max_input_tokens=4000, chunk_overlap=150, temperature=0.3)
    refinement_result = refinement_summarizer(document)
    print(f"Refinement Summary:\n{refinement_result['summary']}\nCost: ${refinement_result['metadata']['cost']:.4f}\nIterations: {refinement_result['metadata']['refinement_iterations']}\n" + "=" * 50)
    
    total_cost = (
        basic_result['metadata']['cost'] +
        template_result['metadata']['cost'] +
        map_reduce_result['metadata']['cost'] +
        refinement_result['metadata']['cost']
    )
    
    print(f"All summarizers tested successfully!\nTotal cost: ${total_cost:.4f}\nFor a web interface, run: streamlit run app.py")

if __name__ == "__main__":
    main()