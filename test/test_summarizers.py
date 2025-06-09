#!/usr/bin/env python3

import os
from summarizers.basic_prompt_summarizer import BasicPromptSummarizer
from summarizers.template_driven_summarizer import TemplateDrivenSummarizer
from summarizers.map_reduce_summarizer import MapReduceSummarizer
from summarizers.refinement_based_summarizer import RefinementBasedSummarizer
import sys
import pathlib
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

def get_sample_document():
    documents_folder = pathlib.Path("documents")
    pdf_path = documents_folder / "Climate_change.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Sample PDF not found at {pdf_path}")
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def test_basic_prompt_summarizer():
    try:
        basic_summarizer = BasicPromptSummarizer(model_name="gemma2-9b-it", temperature=0.7, max_tokens=300)
        sample_document = get_sample_document()
        result = basic_summarizer(sample_document)
        print(f"BASIC | Len: {len(result['summary'])} | Cost: ${result['metadata']['cost']:.4f} | Technique: {result['metadata']['technique']} | In: {result['metadata']['input_tokens']} | Out: {result['metadata']['output_tokens']}\nSummary:\n{result['summary']}\n")
        return True, result
    except Exception as e:
        print(f"BASIC FAILED: {e}")
        return False, None

def test_template_driven_summarizer():
    try:
        template_summarizer = TemplateDrivenSummarizer(model_name="llama3-8b-8192", temperature=0.6, max_tokens=400)
        sample_document = get_sample_document()
        result = template_summarizer(sample_document)
        print(f"TEMPLATE | Len: {len(result['summary'])} | Cost: ${result['metadata']['cost']:.4f} | Technique: {result['metadata']['technique']} | In: {result['metadata']['input_tokens']} | Out: {result['metadata']['output_tokens']} | Custom: {result['metadata'].get('has_custom_template', False)}\nSummary:\n{result['summary']}\n")
        return True, result
    except Exception as e:
        print(f"TEMPLATE FAILED: {e}")
        return False, None

def test_map_reduce_summarizer():
    try:
        map_reduce_summarizer = MapReduceSummarizer(model_name="llama3-8b-8192", max_input_tokens=5000)
        sample_document = get_sample_document()
        result = map_reduce_summarizer(sample_document)
        print(f"MAP-REDUCE | Len: {len(result['summary'])} | Cost: ${result['metadata']['cost']:.4f} | Technique: {result['metadata']['technique']} | In: {result['metadata']['input_tokens']} | Out: {result['metadata']['output_tokens']} | Chunks: {result['metadata']['num_chunks']} | MaxIn: {result['metadata']['max_input_tokens']} | EffChunk: {result['metadata']['effective_chunk_size']} | Overlap: {result['metadata']['chunk_overlap']} | Chunking: {result['metadata']['chunking_method']} | Workers: {result['metadata']['max_workers']} | Time: {result['metadata']['processing_time_seconds']:.2f}s\nSummary:\n{result['summary']}\n")
        return True, result
    except Exception as e:
        print(f"MAP-REDUCE FAILED: {e}")
        return False, None

def test_refinement_based_summarizer():
    try:
        refinement_summarizer = RefinementBasedSummarizer(model_name="llama3-8b-8192", max_input_tokens=4000, chunk_overlap=150, temperature=0.3, max_tokens=600)
        sample_document = get_sample_document()
        result = refinement_summarizer(sample_document)
        print(f"REFINEMENT | Len: {len(result['summary'])} | Cost: ${result['metadata']['cost']:.4f} | Technique: {result['metadata']['technique']} | In: {result['metadata']['input_tokens']} | Out: {result['metadata']['output_tokens']} | Chunks: {result['metadata']['num_chunks']} | MaxIn: {result['metadata']['max_input_tokens']} | InitChunk: {result['metadata']['initial_chunk_size']} | Overlap: {result['metadata']['chunk_overlap']} | Iter: {result['metadata']['refinement_iterations']} | Chunking: {result['metadata']['chunking_method']} | Approach: {result['metadata']['refinement_approach']} | Time: {result['metadata']['processing_time_seconds']:.2f}s\nSummary:\n{result['summary']}\n")
        return True, result
    except Exception as e:
        print(f"REFINEMENT FAILED: {e}")
        return False, None

def test_all_summarizers():
    test_results = []
    
    basic_success, basic_result = test_basic_prompt_summarizer()
    test_results.append(("Basic Prompt Summarizer", basic_success, basic_result))
    
    template_success, template_result = test_template_driven_summarizer()
    test_results.append(("Template Driven Summarizer", template_success, template_result))
    
    map_reduce_success, map_reduce_result = test_map_reduce_summarizer()
    test_results.append(("Map-Reduce Summarizer", map_reduce_success, map_reduce_result))
    
    refinement_success, refinement_result = test_refinement_based_summarizer()
    test_results.append(("Refinement-Based Summarizer", refinement_success, refinement_result))
    
    successful_tests = 0
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    for test_name, success, result in test_results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:<30} {status}")
        if success and result:
            successful_tests += 1
            total_cost += result['metadata']['cost']
            total_input_tokens += result['metadata']['input_tokens']
            total_output_tokens += result['metadata']['output_tokens']
    print(f"Tests Passed: {successful_tests}/{len(test_results)} | Total Cost: ${total_cost:.4f} | In: {total_input_tokens} | Out: {total_output_tokens} | Total: {total_input_tokens + total_output_tokens}")
    if successful_tests == len(test_results):
        print("All tests completed successfully!")
    else:
        print(f"{len(test_results) - successful_tests} test(s) failed.")
    return successful_tests == len(test_results)

def run_individual_test(test_name: str):
    test_functions = {
        "basic": test_basic_prompt_summarizer,
        "template": test_template_driven_summarizer,  
        "map-reduce": test_map_reduce_summarizer,
        "refinement": test_refinement_based_summarizer
    }
    if test_name.lower() in test_functions:
        print(f"Running {test_name} test only...")
        success, result = test_functions[test_name.lower()]()
        print(f"{test_name} test {'completed successfully!' if success else 'failed!'}")
        return success
    else:
        print(f"Unknown test: {test_name}\nAvailable tests: basic, template, map-reduce, refinement")
        return False

if __name__ == "__main__":
    if not os.getenv('GROQ_API_KEY'):
        print("GROQ_API_KEY environment variable not set! Please set your GROQ API key before running this test.")
        exit(1)
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        run_individual_test(test_name)
    else:
        test_all_summarizers()
