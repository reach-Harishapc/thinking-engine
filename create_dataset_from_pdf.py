#!/usr/bin/env python3
"""
Thinking Engine — Create Dataset from PDF
Author: Harish
Purpose:
    Extract text from PDF files and create training datasets for the Thinking Engine.
    This script processes real PDF documents and prepares them for model training.
"""

import os
import sys
from pathlib import Path
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.utils import extract_text_from_pdf, process_pdf_for_training

def create_dataset_from_pdf(pdf_path: str, output_dir: str = "pdf_dataset", chunk_size: int = 800):
    """
    Create a training dataset from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the dataset
        chunk_size: Size of text chunks for training
    """
    print("🧠 Thinking Engine - PDF Dataset Creation")
    print("=" * 50)

    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None

    print(f"📄 Processing PDF: {pdf_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract text from PDF
    print("📖 Extracting text from PDF...")
    try:
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            print("❌ No text extracted from PDF")
            return None

        print(f"✅ Extracted {len(full_text)} characters of text")

        # Save full extracted text
        full_text_path = os.path.join(output_dir, "full_extracted_text.txt")
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"💾 Saved full text to: {full_text_path}")

    except Exception as e:
        print(f"❌ Failed to extract text: {e}")
        return None

    # Create training samples (full content, not limited chunks)
    print("📚 Creating training samples from full PDF content...")

    # Split by natural sections (double newlines for paragraphs)
    paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p.strip()) > 50]

    # If we have too few paragraphs, split by single newlines
    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip() and len(p.strip()) > 100]

    # Filter out very short paragraphs and create meaningful training samples
    training_samples = []
    current_sample = ""

    for para in paragraphs:
        if len(current_sample) + len(para) < 2000:  # Allow larger samples
            current_sample += para + "\n\n"
        else:
            if current_sample.strip():
                training_samples.append(current_sample.strip())
            current_sample = para + "\n\n"

    # Add the last sample
    if current_sample.strip():
        training_samples.append(current_sample.strip())

    # If still too few samples, use the full text as one sample
    if len(training_samples) < 2:
        training_samples = [full_text]

    chunks = training_samples  # Use samples instead of chunks

    if not chunks:
        print("❌ No training samples created")
        return None

    print(f"✅ Created {len(chunks)} training samples")

    # Save chunks as individual files
    chunks_dir = os.path.join(output_dir, "training_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{i+1:02d}.txt"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk.strip())
        print(f"📄 Saved chunk {i+1:2d}: {len(chunk)} chars")

    # Create JSON dataset for training
    json_dataset = {
        "metadata": {
            "source_pdf": os.path.basename(pdf_path),
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "total_characters": len(full_text),
            "created_by": "Thinking Engine PDF Processor"
        },
        "training_data": chunks
    }

    json_path = os.path.join(output_dir, "training_dataset.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dataset, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved JSON dataset to: {json_path}")

    # Create summary
    summary = f"""
PDF Dataset Summary
===================
Source PDF: {os.path.basename(pdf_path)}
Total Characters: {len(full_text):,}
Training Chunks: {len(chunks)}
Chunk Size: {chunk_size} characters
Output Directory: {output_dir}

Files Created:
- full_extracted_text.txt: Complete extracted text
- training_dataset.json: JSON format for training
- training_chunks/: Individual chunk files (chunk_01.txt, chunk_02.txt, ...)

Ready for training with:
python run_model.py --train {output_dir} --save
"""

    summary_path = os.path.join(output_dir, "DATASET_SUMMARY.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary.strip())

    print(f"📋 Saved summary to: {summary_path}")

    print("\n🎉 Dataset creation complete!")
    print(f"📂 Dataset saved in: {output_dir}")
    print(f"📊 Ready to train with {len(chunks)} chunks")

    return {
        "output_dir": output_dir,
        "chunks": len(chunks),
        "characters": len(full_text),
        "json_path": json_path,
        "chunks_dir": chunks_dir
    }

def main():
    """Main function to create dataset from available PDFs."""
    print("🔍 Looking for PDF files to process...")

    # Check for PDFs in current directory and sample files
    pdf_candidates = [
        "arxiv_paper.pdf",
        "sample files/Introduction_to_Quantum_Computers.pdf"
    ]

    available_pdfs = []
    for pdf_path in pdf_candidates:
        if os.path.exists(pdf_path):
            available_pdfs.append(pdf_path)
            print(f"📄 Found PDF: {pdf_path}")

    if not available_pdfs:
        print("❌ No PDF files found in expected locations")
        print("Please place your PDF file in the thinking-engine directory")
        return

    # Process the first available PDF (or let user choose)
    if len(available_pdfs) == 1:
        selected_pdf = available_pdfs[0]
    else:
        print("\n📋 Available PDFs:")
        for i, pdf in enumerate(available_pdfs):
            print(f"{i+1}. {pdf}")
        choice = input("Select PDF to process (1-{}): ".format(len(available_pdfs)))
        try:
            selected_pdf = available_pdfs[int(choice) - 1]
        except (ValueError, IndexError):
            print("❌ Invalid choice, using first PDF")
            selected_pdf = available_pdfs[0]

    # Create dataset name based on PDF
    pdf_name = Path(selected_pdf).stem
    dataset_name = f"{pdf_name}_dataset"

    # Create the dataset
    result = create_dataset_from_pdf(selected_pdf, dataset_name)

    if result:
        print("\n🚀 Ready to train!")
        print(f"Run: python run_model.py --train {result['output_dir']} --save")
        print(f"Then: python run_model.py --load --chat")

if __name__ == "__main__":
    main()
