#!/usr/bin/env python3
"""
Thinking Engine — Multi-Platform Test Runner
Author: Harish
Purpose:
    Simple script to run multi-platform tests across CPU, GPU, MPS, and Quantum backends.
    This is a convenience script for testing without diving into the full test framework.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_basic_tests():
    """Run basic functionality tests."""
    print("Running Basic Thinking Engine Tests")
    print("=" * 50)

    # Test 1: Platform detection
    print("1. Testing platform detection...")
    try:
        from tests.test_multiplatform import MultiPlatformTester
        tester = MultiPlatformTester()
        print("   [PASS] Platform detection successful")
        print(f"   Available backends: {', '.join(tester.platform_info.available_backends)}")
        print(f"   Hardware: {tester.platform_info.hardware_info}")
    except Exception as e:
        print(f"   [FAIL] Platform detection failed: {e}")
        return False

    # Test 2: Sparse encoding
    print("\n2. Testing sparse encoding...")
    try:
        from core.utils import encode_text_to_sparse_representation
        test_text = "Quantum computing uses quantum mechanics principles."
        vector = encode_text_to_sparse_representation(test_text)
        print("   [PASS] Sparse encoding successful")
        print(f"   Vector size: {len(vector)}")
    except Exception as e:
        print(f"   [FAIL] Sparse encoding failed: {e}")
        return False

    # Test 3: Model training
    print("\n3. Testing model training...")
    try:
        from core.learning_manager import LearningManager
        import numpy as np

        lm = LearningManager()
        # Create dummy training data
        dummy_data = np.random.rand(10, 100)
        lm.learn(dummy_data)
        print("   [PASS] Model training successful")
    except Exception as e:
        print(f"   [FAIL] Model training failed: {e}")
        return False

    # Test 4: Model inference
    print("\n4. Testing model inference...")
    try:
        from run_model import ThinkingModelInterface

        engine = ThinkingModelInterface()
        response = engine.think("What is quantum computing?")
        print("   [PASS] Model inference successful")
        print(f"   Response length: {len(response)} characters")
    except Exception as e:
        print(f"   [FAIL] Model inference failed: {e}")
        return False

    print("\n[SUCCESS] All basic tests passed!")
    return True

def run_platform_detection():
    """Run platform detection and show available backends."""
    print("Detecting Available Computing Platforms")
    print("=" * 50)

    try:
        from tests.test_multiplatform import MultiPlatformTester
        tester = MultiPlatformTester()

        print(f"Operating System: {tester.platform_info.os}")
        print(f"Architecture: {tester.platform_info.architecture}")
        print(f"Python Version: {tester.platform_info.python_version}")
        print()

        print("Available Backends:")
        for backend in tester.platform_info.available_backends:
            status = "Available" if backend in ['cpu'] else "Detected"
            print(f"  - {backend.upper()}: {status}")

        print()
        print("Hardware Information:")
        for key, value in tester.platform_info.hardware_info.items():
            print(f"  - {key}: {value}")

        return True

    except Exception as e:
        print(f"[ERROR] Platform detection failed: {e}")
        return False

def run_comprehensive_tests():
    """Run the full multi-platform test suite."""
    print("Running Comprehensive Multi-Platform Tests")
    print("=" * 50)

    try:
        from tests.test_multiplatform import main
        main()
        return True
    except Exception as e:
        print(f"[ERROR] Comprehensive tests failed: {e}")
        return False

def main():
    """Main function with menu-driven interface."""
    print("Thinking Engine - Multi-Platform Test Runner")
    print("=" * 60)

    while True:
        print("\nSelect test option:")
        print("1. Run basic functionality tests")
        print("2. Detect available platforms")
        print("3. Run comprehensive multi-platform tests")
        print("4. Exit")

        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == "1":
                success = run_basic_tests()
                if success:
                    print("\n[SUCCESS] Basic tests completed successfully!")
                else:
                    print("\n[ERROR] Basic tests failed!")

            elif choice == "2":
                success = run_platform_detection()
                if success:
                    print("\n[SUCCESS] Platform detection completed!")
                else:
                    print("\n[ERROR] Platform detection failed!")

            elif choice == "3":
                print("\n[WARNING] Note: Comprehensive tests may take several minutes...")
                confirm = input("Continue? (y/N): ").strip().lower()
                if confirm == 'y':
                    success = run_comprehensive_tests()
                    if success:
                        print("\n[SUCCESS] Comprehensive tests completed!")
                    else:
                        print("\n[ERROR] Comprehensive tests failed!")
                else:
                    print("Test cancelled.")

            elif choice == "4":
                print("\nGoodbye!")
                break

            else:
                print("[ERROR] Invalid choice. Please enter 1-4.")

        except KeyboardInterrupt:
            print("\n\nTest runner interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
