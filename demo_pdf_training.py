#!/usr/bin/env python3
"""
Thinking Engine — PDF Training Demo
Author: Harish
Purpose:
    Demonstrate how to train Thinking Engine with PDF documents.
    This script shows the complete workflow from PDF processing to model training.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from run_model import ThinkingModelInterface

def demo_pdf_training():
    """Demonstrate PDF-based training workflow."""
    print("🧠 Thinking Engine - PDF Training Demo")
    print("=" * 50)

    # Step 1: Check for PDF processing capability
    try:
        from PyPDF2 import PdfReader
        print("✅ PyPDF2 available - PDF processing enabled")
        pdf_available = True
    except ImportError:
        print("⚠️  PyPDF2 not installed. Install with: pip install PyPDF2")
        print("For demo purposes, we'll use text files instead.")
        pdf_available = False

    # Step 2: Create sample training data directory
    training_dir = "demo_training_data"
    os.makedirs(training_dir, exist_ok=True)

    print(f"\n📁 Created training directory: {training_dir}")

    # Step 3: Create sample documents (in real usage, you'd place your PDFs here)
    sample_docs = {
        "ai_research.txt": """
        Artificial Intelligence Research Trends 2025

        The field of AI continues to evolve rapidly with several key trends emerging:

        1. Multimodal Learning: Combining text, vision, and audio processing
        2. Edge Computing: Running AI models on resource-constrained devices
        3. Federated Learning: Privacy-preserving distributed training
        4. Explainable AI: Making AI decisions interpretable to humans
        5. Cognitive Architectures: Building AI systems inspired by human cognition

        These trends represent the future direction of AI research and development.
        """,

        "machine_learning_basics.txt": """
        Machine Learning Fundamentals

        Machine learning is a subset of artificial intelligence that enables systems
        to learn from data without being explicitly programmed.

        Key Concepts:
        - Supervised Learning: Learning from labeled examples
        - Unsupervised Learning: Finding patterns in unlabeled data
        - Reinforcement Learning: Learning through interaction and rewards
        - Deep Learning: Neural networks with multiple layers

        Applications include image recognition, natural language processing,
        recommendation systems, and autonomous vehicles.
        """,

        "neural_networks.txt": """
        Neural Network Architectures

        Neural networks are computing systems inspired by biological neural networks.

        Common Architectures:
        - Feedforward Neural Networks: Information flows in one direction
        - Convolutional Neural Networks (CNNs): Excellent for image processing
        - Recurrent Neural Networks (RNNs): Handle sequential data
        - Transformer Networks: Revolutionized natural language processing

        Training involves forward propagation, loss calculation, and backpropagation
        to update network weights and minimize prediction errors.
        """
    }

    # Create sample files
    for filename, content in sample_docs.items():
        filepath = os.path.join(training_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"📄 Created sample document: {filename}")

    print(f"\n📚 Training dataset ready with {len(sample_docs)} documents")

    # Step 4: Initialize Thinking Engine
    print("\n🤖 Initializing Thinking Engine...")
    model = ThinkingModelInterface()
    print("✅ Model initialized successfully")

    # Step 5: Train the model
    print(f"\n🎓 Starting training with dataset: {training_dir}")
    print("This will process text files and create sparse synaptic representations...")

    try:
        model.train(training_dir)
        print("✅ Training completed successfully!")

        # Step 6: Save the trained model
        model_name = "demo_trained_model.think"
        model.save_model(model_name, compressed=True)
        print(f"💾 Model saved as: {model_name}")

        # Step 7: Test the trained model
        print("\n🧪 Testing trained model with sample queries...")

        test_queries = [
            "What are the main trends in AI research?",
            "Explain machine learning basics",
            "What are neural network architectures?"
        ]

        for query in test_queries:
            print(f"\n❓ Query: {query}")
            response = model.think(query)
            print(f"🤖 Response: {response[:200]}..." if len(response) > 200 else f"🤖 Response: {response}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # Step 8: Cleanup
    print("\n🧹 Cleaning up demo files...")
    import shutil
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
        print(f"🗑️  Removed training directory: {training_dir}")

    print("\n🎉 PDF Training Demo Complete!")
    print("You can now train Thinking Engine with your own PDF documents!")
    print("\nExample usage with real PDFs:")
    print("1. Place your PDF files in a directory")
    print("2. Run: python run_model.py --train /path/to/pdf/directory --save")
    print("3. The system will automatically extract text and train the model")
    print("\nFor real PDF files, ensure PyPDF2 is installed: pip install PyPDF2")
if __name__ == "__main__":
    demo_pdf_training()
