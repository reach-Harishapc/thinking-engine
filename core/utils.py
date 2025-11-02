"""
Thinking Engine — utils.py
Author: Harish
Purpose:
    Utility functions for data processing, compression, and file handling.
"""

import os
import json
import gzip
import hashlib
from typing import Dict, Any, List
import numpy as np
import logging
from datetime import datetime

try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARNING] PyPDF2 not installed. PDF processing disabled.")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content as string
    """
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")

    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract text from PDF {pdf_path}: {e}")
        return ""


def process_pdf_for_training(pdf_path: str, chunk_size: int = 1000) -> List[str]:
    """
    Process a PDF file and split it into training chunks.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks for training
    """
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []

    # Split text into chunks
    chunks = []
    current_chunk = ""

    for paragraph in text.split('\n\n'):  # Split by paragraphs
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Paragraph itself is too long, split it
                words = paragraph.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                if temp_chunk:
                    current_chunk = temp_chunk
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def compress_data(data: Dict[str, Any]) -> bytes:
    """
    Compress data using gzip.

    Args:
        data: Dictionary to compress

    Returns:
        Compressed data as bytes
    """
    json_str = json.dumps(data, indent=2)
    return gzip.compress(json_str.encode('utf-8'))


def decompress_data(compressed_data: bytes) -> Dict[str, Any]:
    """
    Decompress gzipped data.

    Args:
        compressed_data: Compressed data bytes

    Returns:
        Decompressed dictionary
    """
    json_str = gzip.decompress(compressed_data).decode('utf-8')
    return json.loads(json_str)


def calculate_integrity_hash(data: Dict[str, Any]) -> str:
    """
    Calculate SHA256 integrity hash for data.

    Args:
        data: Dictionary to hash

    Returns:
        SHA256 hash as hex string
    """
    # Create a normalized JSON string for consistent hashing
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def validate_file_integrity(filepath: str, expected_hash: str) -> bool:
    """
    Validate file integrity against expected hash.

    Args:
        filepath: Path to file to check
        expected_hash: Expected SHA256 hash

    Returns:
        True if integrity is valid
    """
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash == expected_hash


def create_training_dataset_from_directory(directory_path: str) -> List[str]:
    """
    Create a training dataset from all files in a directory.
    Supports .txt, .json, and .pdf files.

    Args:
        directory_path: Path to directory containing training files

    Returns:
        List of training samples
    """
    training_data = []

    if not os.path.exists(directory_path):
        print(f"[WARNING] Directory not found: {directory_path}")
        return training_data

    print(f"[INFO] Processing training data from: {directory_path}")

    for root, _, files in os.walk(directory_path):
        for filename in files:
            filepath = os.path.join(root, filename)

            try:
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        training_data.append(content)
                        print(f"[INFO] Loaded text file: {filename}")

                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            training_data.extend(data)
                        else:
                            training_data.append(json.dumps(data))
                        print(f"[INFO] Loaded JSON file: {filename}")

                elif filename.endswith('.pdf'):
                    if PDF_SUPPORT:
                        chunks = process_pdf_for_training(filepath)
                        training_data.extend(chunks)
                        print(f"[INFO] Processed PDF file: {filename} ({len(chunks)} chunks)")
                    else:
                        print(f"[WARNING] Skipping PDF file (PyPDF2 not available): {filename}")

            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")

    print(f"[SUCCESS] Created training dataset with {len(training_data)} samples")
    return training_data


def encode_text_to_sparse_representation(text: str, vocab_size: int = 10000) -> np.ndarray:
    """
    Encode text to sparse synaptic representation.

    Args:
        text: Input text to encode
        vocab_size: Size of vocabulary for hashing

    Returns:
        Sparse numpy array representation
    """
    # Simple hash-based encoding (can be improved with proper tokenization)
    words = text.lower().split()
    indices = [hash(word) % vocab_size for word in words]

    # Create sparse representation
    sparse_vector = np.zeros(vocab_size, dtype=np.float32)
    for idx in indices:
        sparse_vector[idx] += 1.0

    # Normalize
    if np.sum(sparse_vector) > 0:
        sparse_vector = sparse_vector / np.sum(sparse_vector)

    return sparse_vector


def save_sparse_dataset(dataset: List[np.ndarray], filepath: str):
    """
    Save sparse dataset to compressed file.

    Args:
        dataset: List of sparse vectors
        filepath: Output file path
    """
    data_dict = {
        'dataset': [vec.tolist() for vec in dataset],
        'metadata': {
            'samples': len(dataset),
            'vocab_size': len(dataset[0]) if dataset else 0,
            'created': str(np.datetime64('now'))
        }
    }

    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data_dict, f)

    print(f"[SUCCESS] Saved sparse dataset to {filepath}")


def load_sparse_dataset(filepath: str) -> List[np.ndarray]:
    """
    Load sparse dataset from compressed file.

    Args:
        filepath: Input file path

    Returns:
        List of sparse vectors
    """
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        data_dict = json.load(f)

    dataset = [np.array(vec) for vec in data_dict['dataset']]
    metadata = data_dict.get('metadata', {})

    print(f"[INFO] Loaded dataset with {metadata.get('samples', len(dataset))} samples")
    return dataset


class Logger:
    """
    Simple logging utility for Thinking Engine components.
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level.upper()
        self.levels = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40
        }

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp and component name."""
        level = level.upper()
        if self.levels.get(level, 0) >= self.levels.get(self.level, 20):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{self.name}] [{level}] {message}")

    def debug(self, message: str):
        """Log debug message."""
        self.log(message, "DEBUG")

    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")

    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")

    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")
