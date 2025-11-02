"""
Thinking Engine — run_model.py
Author: Harish
Purpose:
    Main entry point to train, save, load, and chat with the Thinking Engine model.
"""

import os
import json
import argparse
import numpy as np
import sys
from datetime import datetime
from core.cortex import Cortex
from core.memory import MemoryManager as Memory
from core.learning_manager import LearningManager
from core.execution_engine import CognitiveExecutionEngine as ExecutionEngine
from core.utils import create_training_dataset_from_directory, encode_text_to_sparse_representation

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class ThinkingModelInterface:
    """
    High-level interface for the Thinking Engine model.
    Provides training, saving, loading, and interactive thinking capabilities.
    """

    def __init__(self, system_prompt_file=None):
        self.memory = Memory()
        self.cortex = Cortex(memory=self.memory)
        self.executor = ExecutionEngine()
        self.learning_manager = LearningManager()

        # Load system prompt if specified
        if system_prompt_file:
            self.cortex.load_system_prompt(prompt_file=system_prompt_file)

    def train(self, dataset_path: str):
        """
        Train the model using data from the dataset path.
        Supports .txt, .json, and .pdf files with intelligent chunking.
        """
        print(f"[TRAIN] Starting training with dataset: {dataset_path}")

        # Check if this is a PDF-generated dataset directory
        training_samples = []

        # First, try to load from JSON dataset if it exists (PDF-processed data)
        json_dataset_path = os.path.join(dataset_path, "training_dataset.json")
        if os.path.exists(json_dataset_path):
            print("[INFO] Found PDF-processed dataset, loading training chunks...")
            try:
                with open(json_dataset_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if "training_data" in json_data:
                        training_samples = json_data["training_data"]
                        print(f"[SUCCESS] Loaded {len(training_samples)} training chunks from PDF dataset")
            except Exception as e:
                print(f"[WARNING] Could not load JSON dataset: {e}")

        # If no JSON dataset found, fall back to directory processing
        if not training_samples:
            print("[INFO] No PDF dataset found, processing directory files...")
            training_samples = create_training_dataset_from_directory(dataset_path)

        if not training_samples:
            print("[ERROR] No training data found!")
            return

        print(f"[INFO] Processing {len(training_samples)} training samples...")

        # Store training samples in memory for future retrieval
        print("[MEMORY] Storing training data in memory for future queries...")
        for i, sample in enumerate(training_samples):
            if i % 50 == 0:  # Progress indicator for memory storage
                print(f"[MEMORY] Storing sample {i+1}/{len(training_samples)}")
            # Store each training sample in memory with searchable content
            self.memory.store_experience("training_sample", sample[:200], {"full_content": sample, "index": i})

        # Convert to sparse synaptic representations
        print("[INFO] Encoding dataset to sparse synaptic format...")
        encoded_samples = []
        for i, sample in enumerate(training_samples):
            if i % 100 == 0:  # Progress indicator
                print(f"[ENCODE] Processing sample {i+1}/{len(training_samples)}")
            sparse_vector = encode_text_to_sparse_representation(sample)
            encoded_samples.append(sparse_vector)

        # Convert to numpy array for learning
        encoded_array = np.array(encoded_samples)

        # Train the learning manager
        print("[LEARNING] Updating synaptic weights...")
        self.learning_manager.learn(encoded_array)

        # Store training metadata in memory
        training_metadata = {
            "samples_processed": len(training_samples),
            "encoding_method": "sparse_synaptic",
            "vocab_size": 10000,
            "dataset_path": dataset_path,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.store_experience("training", "completed", training_metadata)

        print("[SUCCESS] Training completed successfully!")
        print(f"[STATS] Processed {len(training_samples)} samples")
        print(f"[STATS] Created {len(encoded_samples)} sparse representations")
        print(f"[STATS] Stored {len(training_samples)} samples in memory for future queries")

    def save_model(self, model_name="thinking_model.think", compressed=False, encrypted=False):
        """
        Save the model state into a .think file.
        Options:
        - compressed: Use gzip compression for smaller files
        - encrypted: Add basic integrity check (not full encryption)
        """
        print("[SAVE] Saving model...")
        state = {
            "cortex": self.cortex.export_state(),
            "memory": self.memory.export_state(),
            "learning": self.learning_manager.export_state(),
            "metadata": {
                "version": "1.0.1",
                "timestamp": datetime.now().isoformat(),
                "compressed": compressed,
                "encrypted": encrypted
            }
        }

        # Add integrity hash for tamper detection
        import hashlib
        state_str = json.dumps(state, sort_keys=True)
        integrity_hash = hashlib.sha256(state_str.encode()).hexdigest()
        state["integrity"] = integrity_hash

        path = os.path.join(MODEL_DIR, model_name)

        if compressed:
            import gzip
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            print(f"[SUCCESS] Compressed model saved to {path}")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            print(f"[SUCCESS] Model saved to {path}")

        if encrypted:
            print("[INFO] Integrity check enabled - model tampering will be detected")

    def load_model(self, model_name="thinking_model.think", verify_integrity=True):
        """
        Load the model state from a .think file.
        Options:
        - verify_integrity: Check if model has been tampered with
        """
        path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Model file not found: {path}")

        print("[LOAD] Loading model from storage...")

        # Handle compressed files
        if path.endswith('.gz') or model_name.endswith('.gz'):
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as f:
                state = json.load(f)
            print("[INFO] Loaded compressed model")
        else:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

        # Verify integrity if enabled
        if verify_integrity and "integrity" in state:
            import hashlib
            original_hash = state.pop("integrity")  # Remove hash from state
            state_str = json.dumps(state, sort_keys=True)
            current_hash = hashlib.sha256(state_str.encode()).hexdigest()

            if original_hash != current_hash:
                print("[WARNING] Model integrity check failed - file may have been tampered with!")
                print(f"[DEBUG] Expected: {original_hash[:16]}...")
                print(f"[DEBUG] Got:      {current_hash[:16]}...")
                user_input = input("Continue loading anyway? (y/N): ")
                if user_input.lower() != 'y':
                    raise ValueError("Model integrity check failed - aborting load")
            else:
                print("[INFO] Model integrity verified ✓")

        # Load model components
        self.cortex.import_state(state["cortex"])
        self.memory.import_state(state["memory"])
        self.learning_manager.import_state(state["learning"])

        # Show metadata if available
        if "metadata" in state:
            meta = state["metadata"]
            print(f"[INFO] Model version: {meta.get('version', 'unknown')}")
            print(f"[INFO] Saved: {meta.get('timestamp', 'unknown')}")

        print("[SUCCESS] Model loaded successfully.")

    def think(self, query: str):
        """
        Generate a response based on current model state.
        """
        print(f"[THINK] Processing query: {query}")
        # Use Cortex to reason
        response = self.cortex.reason(query)
        # Store in memory
        self.memory.store_experience("query", query, {})
        self.memory.store_experience("response", response, {})
        return response


def main():
    parser = argparse.ArgumentParser(description="Thinking Engine: Train, Save, Load, Chat")
    parser.add_argument("--train", type=str, help="Path to dataset folder for training")
    parser.add_argument("--chat", action="store_true", help="Chat interactively with model")
    parser.add_argument("--save", action="store_true", help="Save model after training")
    parser.add_argument("--load", action="store_true", help="Load previously saved model")
    parser.add_argument("--system-prompt", type=str, help="Path to system prompt JSON file")
    parser.add_argument("--query", type=str, help="Process a single query and exit")
    parser.add_argument("--file", type=str, help="Process queries from a file (one per line)")
    args = parser.parse_args()

    engine = ThinkingModelInterface(system_prompt_file=args.system_prompt)

    if args.train:
        engine.train(args.train)
        if args.save:
            engine.save_model()

    elif args.load:
        engine.load_model()

    # Handle file input
    if args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        
        with open(args.file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"[FILE] Processing {len(queries)} queries from {args.file}")
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}/{len(queries)} ---")
            print(f"You: {query}")
            response = engine.think(query)
            print(f"Model: {response}")
        return

    # Handle single query from argument
    if args.query:
        print(f"You: {args.query}")
        response = engine.think(args.query)
        print(f"Model: {response}")
        return

    # Handle piped input (check if stdin has content)
    if args.chat and not sys.stdin.isatty():
        # Stdin is being piped, process available input
        queries = []
        try:
            for line in sys.stdin:
                query = line.strip()
                if query:
                    queries.append(query)
        except EOFError:
            pass
        
        if queries:
            print(f"[PIPED] Processing {len(queries)} query(s) from stdin")
            for i, query in enumerate(queries, 1):
                print(f"\n--- Query {i}/{len(queries)} ---")
                print(f"You: {query}")
                response = engine.think(query)
                print(f"Model: {response}")
            return
    
    # Interactive chat mode (original behavior)
    if args.chat:
        print("Thinking Engine Interactive Chat")
        print("Type 'exit' to quit.\n")
        while True:
            try:
                query = input("You: ")
                if query.lower() in ["exit", "quit"]:
                    print("Exiting chat.")
                    break
                response = engine.think(query)
                print(f"Model: {response}\n")
            except EOFError:
                print("\nExiting chat (EOF detected).")
                break
            except KeyboardInterrupt:
                print("\nExiting chat (Ctrl+C).")
                break


if __name__ == "__main__":
    main()
