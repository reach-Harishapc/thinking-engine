#!/usr/bin/env python3
"""
Thinking Engine — PDF Processing Test Script
Author: Harish
Purpose:
    Test PDF processing capabilities and demonstrate dataset preparation for training.
    This script shows how to extract text from PDFs and prepare training data.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.utils import (
    extract_text_from_pdf,
    process_pdf_for_training,
    create_training_dataset_from_directory,
    encode_text_to_sparse_representation
)

def test_pdf_extraction():
    """Test basic PDF text extraction."""
    print("🧠 Thinking Engine - PDF Processing Test")
    print("=" * 50)

    # Create a sample PDF path (you can replace this with your actual PDF)
    sample_pdf = "sample_document.pdf"

    # For demo purposes, let's create a simple text file to simulate PDF content
    if not os.path.exists(sample_pdf):
        print(f"[INFO] Sample PDF not found. Creating demo text file: {sample_pdf}")
        demo_content = """
        Thinking Engine: A Cognitive AI Framework

        Introduction

        Artificial Intelligence has evolved significantly over the past decade. Machine learning models,
        particularly deep learning architectures, have achieved remarkable success in various domains
        including computer vision, natural language processing, and autonomous systems.

        However, traditional AI approaches often operate as "black boxes" with limited transparency
        and user control. This opacity creates challenges in understanding model behavior, debugging
        issues, and ensuring ethical AI deployment.

        The Thinking Engine framework addresses these limitations by providing a transparent,
        user-controllable AI system built on cognitive principles inspired by biological neural systems.

        Key Features

        1. JSON-Based Model Persistence
        Unlike binary formats used by PyTorch and TensorFlow, Thinking Engine stores models in
        human-readable JSON format. This enables direct inspection and modification of AI behavior.

        2. Multi-Agent Architecture
        The system employs specialized agents for different cognitive tasks:
        - Web Agent: Internet research and content analysis
        - Code Agent: Python execution and debugging
        - File Agent: Secure file system operations
        - Reasoning Agent: Logical analysis and planning

        3. Experience-Based Learning
        The framework learns from interactions and experiences, continuously improving
        its responses and decision-making capabilities.

        4. Direct Model Surgery
        Users can directly edit the AI's "brain" by modifying JSON structures, enabling
        real-time customization without retraining.

        Technical Implementation

        The Thinking Engine uses sparse synaptic representations for efficient computation
        while maintaining cognitive flexibility. The multi-agent coordination is managed
        through a central cortex that routes queries to appropriate specialized agents.

        Future Directions

        Future work will focus on expanding agent capabilities, improving learning algorithms,
        and scaling the system for enterprise applications while maintaining the core
        principles of transparency and user empowerment.
        """

        # Save as text file for demo (you can replace with actual PDF)
        with open(sample_pdf.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
            f.write(demo_content.strip())
        print(f"[SUCCESS] Created demo text file: {sample_pdf.replace('.pdf', '.txt')}")

    # Test PDF processing (if PDF exists)
    pdf_path = sample_pdf
    txt_path = sample_pdf.replace('.pdf', '.txt')

    if os.path.exists(pdf_path):
        print(f"\n[TEST] Testing PDF extraction: {pdf_path}")
        try:
            text = extract_text_from_pdf(pdf_path)
            print(f"[SUCCESS] Extracted {len(text)} characters from PDF")
            print(f"[PREVIEW] First 200 characters:\n{text[:200]}...")
        except Exception as e:
            print(f"[ERROR] PDF extraction failed: {e}")
    else:
        print(f"[INFO] PDF not found: {pdf_path}")

    # Test text file processing
    if os.path.exists(txt_path):
        print(f"\n[TEST] Testing text file processing: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        print(f"[INFO] Text file contains {len(text_content)} characters")

        # Test chunking (for text files, we'll simulate PDF processing)
        # Split text into chunks manually for demo
        chunk_size = 500
        chunks = []
        for i in range(0, len(text_content), chunk_size):
            chunk = text_content[i:i + chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())

        print(f"[SUCCESS] Split into {len(chunks)} training chunks")

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"[CHUNK {i+1}] {len(chunk)} chars: {chunk[:100]}...")

    # Test directory processing
    print("\n[TEST] Testing directory dataset creation")
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)

    # Create sample files
    sample_files = {
        "sample1.txt": "This is a sample text for training. It contains information about machine learning.",
        "sample2.txt": "Another training sample discussing artificial intelligence and neural networks.",
        "data.json": ["Sample JSON data", "More training examples", "AI concepts"]
    }

    for filename, content in sample_files.items():
        filepath = os.path.join(test_dir, filename)
        if isinstance(content, list):
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

    print(f"[INFO] Created test dataset in: {test_dir}")

    # Process directory
    training_data = create_training_dataset_from_directory(test_dir)
    print(f"[SUCCESS] Created training dataset with {len(training_data)} samples")

    # Show sample data
    for i, sample in enumerate(training_data[:3]):
        print(f"[SAMPLE {i+1}] {len(sample)} chars: {sample[:80]}...")

    # Test encoding
    print("\n[TEST] Testing sparse encoding")
    if training_data:
        sample_text = training_data[0]
        sparse_vector = encode_text_to_sparse_representation(sample_text)
        print(f"[SUCCESS] Encoded text to {len(sparse_vector)}-dimensional sparse vector")
        print(f"[STATS] Non-zero elements: {np.count_nonzero(sparse_vector)}")
        print(f"[STATS] Sparsity: {(1 - np.count_nonzero(sparse_vector)/len(sparse_vector)):.2%}")

    print("\n🎉 PDF Processing Test Complete!")
    print("You can now use PDFs for training your Thinking Engine model!")
    print(f"Example usage: python run_model.py --train {test_dir} --save")

    # Cleanup
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"[CLEANUP] Removed test directory: {test_dir}")


if __name__ == "__main__":
    import numpy as np
    test_pdf_extraction()
