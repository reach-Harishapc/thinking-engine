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
from datetime import datetime
from core.cortex import Cortex
from core.memory import MemoryManager as Memory
from core.learning_manager import LearningManager
from core.execution_engine import CognitiveExecutionEngine as ExecutionEngine
# from core.utils import compress_data, decompress_data

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
        Supports .txt, .json, and .pdf (through pre-processing).
        """
        print(f"[TRAIN] Starting training with dataset: {dataset_path}")
        data = []

        for root, _, files in os.walk(dataset_path):
            for f in files:
                file_path = os.path.join(root, f)
                if f.endswith(".json"):
                    with open(file_path, "r") as fp:
                        data.extend(json.load(fp))
                elif f.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as fp:
                        data.append(fp.read())
                elif f.endswith(".pdf"):
                    # Convert PDF -> text if pdfminer or similar is available later
                    print(f"[INFO] Skipping {f} (PDF handler not integrated yet)")
                    continue

        # Convert to numpy sparse representation
        print("[INFO] Encoding dataset to sparse synaptic format...")
        encoded = np.array([hash(x) % 10_000 for x in data])

        # Simulate training in learning manager
        print("[LEARNING] Updating synaptic weights...")
        self.learning_manager.learn(encoded)
        self.memory.store_experience("training", "data", {"encoded": encoded.tolist()})

        print("[SUCCESS] Training completed successfully.")

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
    args = parser.parse_args()

    engine = ThinkingModelInterface(system_prompt_file=args.system_prompt)

    if args.train:
        engine.train(args.train)
        if args.save:
            engine.save_model()

    elif args.load:
        engine.load_model()

    if args.chat:
        print("Thinking Engine Interactive Chat")
        print("Type 'exit' to quit.\n")
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break
            response = engine.think(query)
            print(f"Model: {response}\n")


if __name__ == "__main__":
    main()
