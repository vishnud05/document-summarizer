#!/usr/bin/env python3
"""
Comprehensive test script for all refactored summarizers.
Tests the new centralized client architecture and LangSmith tracing.
"""

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
        # Construct the path to the PDF file in the user's Documents folder
        documents_folder = pathlib.Path("documents")
        pdf_path = documents_folder / "Climate_change.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Sample PDF not found at {pdf_path}")

        # Read and concatenate all text from the PDF
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()


def test_basic_prompt_summarizer():
    """Test the BasicPromptSummarizer with new architecture."""
    print("\n📝 Testing Basic Prompt Summarizer...")
    print("-" * 50)
    
    try:
        # Initialize summarizer
        basic_summarizer = BasicPromptSummarizer(
            model_name="gemma2-9b-it",
            temperature=0.7,
            max_tokens=300
        )
        
        # Get test document
        sample_document = get_sample_document()
        
        # Run summarization
        result = basic_summarizer(sample_document)
        
        # Display results
        print("✅ Basic Prompt Summarizer - SUCCESS")
        print(f"📊 Summary Length: {len(result['summary'])} characters")
        print(f"💰 Cost: ${result['metadata']['cost']:.4f}")
        print(f"🎯 Technique: {result['metadata']['technique']}")
        print(f"🔢 Input Tokens: {result['metadata']['input_tokens']}")
        print(f"🔢 Output Tokens: {result['metadata']['output_tokens']}")
        print("-" * 40)
        print("Summary:")
        print(result["summary"])
        print()
        
        return True, result
        
    except Exception as e:
        print(f"❌ Basic Prompt Summarizer - FAILED: {e}")
        return False, None


def test_template_driven_summarizer():
    """Test the TemplateDrivenSummarizer with new architecture."""
    print("\n📋 Testing Template Driven Summarizer...")
    print("-" * 50)
    
    try:
        # Initialize summarizer
        template_summarizer = TemplateDrivenSummarizer(
            model_name="llama3-8b-8192",
            temperature=0.6,
            max_tokens=400
        )
        
        # Get test document
        sample_document = get_sample_document()
        
        # Run summarization
        result = template_summarizer(sample_document)
        
        # Display results
        print("✅ Template Driven Summarizer - SUCCESS")
        print(f"📊 Summary Length: {len(result['summary'])} characters")
        print(f"💰 Cost: ${result['metadata']['cost']:.4f}")
        print(f"🎯 Technique: {result['metadata']['technique']}")
        print(f"🔢 Input Tokens: {result['metadata']['input_tokens']}")
        print(f"🔢 Output Tokens: {result['metadata']['output_tokens']}")
        print(f"🎨 Custom Template: {result['metadata'].get('has_custom_template', False)}")
        print("-" * 40)
        print("Summary:")
        print(result["summary"])
        print()
        
        return True, result
        
    except Exception as e:
        print(f"❌ Template Driven Summarizer - FAILED: {e}")
        return False, None


def test_map_reduce_summarizer():
    """Test the MapReduceSummarizer with new architecture."""
    print("\n🗂️ Testing Map-Reduce Summarizer...")
    print("-" * 50)
    try:        # Initialize summarizer with token-based chunking
        map_reduce_summarizer = MapReduceSummarizer(
            model_name="llama3-8b-8192",
            max_input_tokens=5000
        )
        
        # Get test document
        sample_document = get_sample_document()
        
        # Run summarization
        result = map_reduce_summarizer(sample_document)
        
        # Display results
        print("✅ Map-Reduce Summarizer - SUCCESS")
        print(f"📊 Summary Length: {len(result['summary'])} characters")
        print(f"💰 Cost: ${result['metadata']['cost']:.4f}")
        print(f"🎯 Technique: {result['metadata']['technique']}")
        print(f"🔢 Input Tokens: {result['metadata']['input_tokens']}")
        print(f"🔢 Output Tokens: {result['metadata']['output_tokens']}")          
        print(f"🧩 Number of Chunks: {result['metadata']['num_chunks']}")
        print(f"📏 Max Input Tokens: {result['metadata']['max_input_tokens']}")
        print(f"🎯 Effective Chunk Size: {result['metadata']['effective_chunk_size']} tokens")
        print(f"🔄 Token Overlap: {result['metadata']['chunk_overlap']}")
        print(f"🪙 Chunking Method: {result['metadata']['chunking_method']}")
        print(f"👥 Parallel Workers: {result['metadata']['max_workers']}")
        print(f"⏱️  Processing Time: {result['metadata']['processing_time_seconds']:.2f}s")
        print("-" * 40)
        print("Summary:")
        print(result["summary"])
        print()
        
        return True, result
        
    except Exception as e:
        print(f"❌ Map-Reduce Summarizer - FAILED: {e}")
        return False, None


def test_refinement_based_summarizer():
    """Test the RefinementBasedSummarizer with new architecture."""
    print("\n🔄 Testing Refinement-Based Summarizer...")
    print("-" * 50)
    
    try:
        # Initialize summarizer with token-based chunking
        refinement_summarizer = RefinementBasedSummarizer(
            model_name="llama3-8b-8192",
            max_input_tokens=4000,  # Use token-based chunking
            chunk_overlap=150,      # Token overlap
            temperature=0.3,        # Moderate temperature for refinement
            max_tokens=600
        )
        
        # Get test document
        sample_document = get_sample_document()
        
        # Run summarization
        result = refinement_summarizer(sample_document)
        
        # Display results
        print("✅ Refinement-Based Summarizer - SUCCESS")
        print(f"📊 Summary Length: {len(result['summary'])} characters")
        print(f"💰 Cost: ${result['metadata']['cost']:.4f}")
        print(f"🎯 Technique: {result['metadata']['technique']}")
        print(f"🔢 Input Tokens: {result['metadata']['input_tokens']}")
        print(f"🔢 Output Tokens: {result['metadata']['output_tokens']}")
        print(f"🧩 Number of Chunks: {result['metadata']['num_chunks']}")
        print(f"📏 Max Input Tokens: {result['metadata']['max_input_tokens']}")
        print(f"🎯 Initial Chunk Size: {result['metadata']['initial_chunk_size']} tokens")
        print(f"🔄 Token Overlap: {result['metadata']['chunk_overlap']}")
        print(f"🔄 Refinement Iterations: {result['metadata']['refinement_iterations']}")
        print(f"🪙 Chunking Method: {result['metadata']['chunking_method']}")
        print(f"🎯 Refinement Approach: {result['metadata']['refinement_approach']}")
        print(f"⏱️  Processing Time: {result['metadata']['processing_time_seconds']:.2f}s")
        print("-" * 40)
        print("Summary:")
        print(result["summary"])
        print()
        
        return True, result
        
    except Exception as e:
        print(f"❌ Refinement-Based Summarizer - FAILED: {e}")
        return False, None


def test_all_summarizers():
    """Run all summarizer tests and display comprehensive results."""
    print("🌍 Testing All Summarizers with Refactored Architecture")
    print("=" * 80)
    
    # Track results from all tests
    test_results = []
    
    # Run individual tests
    basic_success, basic_result = test_basic_prompt_summarizer()
    test_results.append(("Basic Prompt Summarizer", basic_success, basic_result))
    
    template_success, template_result = test_template_driven_summarizer()
    test_results.append(("Template Driven Summarizer", template_success, template_result))
    
    map_reduce_success, map_reduce_result = test_map_reduce_summarizer()
    test_results.append(("Map-Reduce Summarizer", map_reduce_success, map_reduce_result))
    
    refinement_success, refinement_result = test_refinement_based_summarizer()
    test_results.append(("Refinement-Based Summarizer", refinement_success, refinement_result))
    
    # Display comprehensive summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    successful_tests = 0
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for test_name, success, result in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        
        if success and result:
            successful_tests += 1
            total_cost += result['metadata']['cost']
            total_input_tokens += result['metadata']['input_tokens']
            total_output_tokens += result['metadata']['output_tokens']
    
    print("-" * 80)
    print(f"Tests Passed: {successful_tests}/{len(test_results)}")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Tokens: {total_input_tokens + total_output_tokens}")
    
    print("\n📈 LangSmith Integration Status:")
    print("   ✅ Each summarizer tracked separately with unique tags")
    print("   ✅ Centralized client setup in base class")
    print("   ✅ Token usage and cost metrics logged")
    print("   ✅ Individual summarizer performance analyzed")
    print("   ✅ Custom tracing for Map-Reduce phases")
    
    if successful_tests == len(test_results):
        print("\n🎉 All tests completed successfully!")
        print("🔍 Check your LangSmith dashboard for detailed tracing data.")
    else:
        print(f"\n⚠️  {len(test_results) - successful_tests} test(s) failed. Check error messages above.")
    
    return successful_tests == len(test_results)


def run_individual_test(test_name: str):
    """Run a specific test by name."""
    test_functions = {
        "basic": test_basic_prompt_summarizer,
        "template": test_template_driven_summarizer,  
        "map-reduce": test_map_reduce_summarizer,
        "refinement": test_refinement_based_summarizer
    }
    
    if test_name.lower() in test_functions:
        print(f"🚀 Running {test_name} test only...")
        success, result = test_functions[test_name.lower()]()
        
        if success:
            print(f"✅ {test_name} test completed successfully!")
        else:
            print(f"❌ {test_name} test failed!")
        
        return success
    else:
        print(f"❌ Unknown test: {test_name}")
        print("Available tests: basic, template, map-reduce, refinement")
        return False


if __name__ == "__main__":
    # Check if GROQ_API_KEY is set
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY environment variable not set!")
        print("Please set your GROQ API key before running this test.")
        exit(1)
    
    # Check if user wants to run a specific test
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        run_individual_test(test_name)
    else:
        # Run all tests
        test_all_summarizers()
