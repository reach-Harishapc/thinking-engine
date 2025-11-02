#!/usr/bin/env python3
"""
Command-line interface for Thinking Engine
"""

import sys
import argparse
from .core.cortex import Cortex


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Thinking Engine - Transparent Cognitive AI Framework"
    )

    parser.add_argument(
        '--chat',
        action='store_true',
        help='Start interactive chat mode'
    )

    parser.add_argument(
        '--train',
        type=str,
        help='Train model with data from specified directory'
    )

    parser.add_argument(
        '--save',
        type=str,
        help='Save trained model to specified path'
    )

    parser.add_argument(
        '--load',
        type=str,
        help='Load model from specified path'
    )

    parser.add_argument(
        '--server',
        action='store_true',
        help='Start REST API server'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port for API server (default: 8080)'
    )

    args = parser.parse_args()

    if args.chat:
        # Start interactive chat
        print("🧠 Thinking Engine Interactive Chat")
        print("Type 'quit' or 'exit' to end the conversation")
        print("-" * 50)
        print("🤖 Thinking Engine: Hello! I'm a transparent cognitive AI framework.")
        print("🤖 Thinking Engine: Interactive chat functionality coming soon!")
        print("🤖 Thinking Engine: For now, please use the run_model.py script directly.")

    elif args.server:
        # Start API server
        try:
            from .server import main as start_server
            print(f"🚀 Starting Thinking Engine API server on port {args.port}")
            # Note: This would need to be implemented to accept port argument
            start_server()
        except ImportError:
            print("❌ API server not available. Please check installation.")

    elif args.train:
        # Train model
        print(f"🎓 Training Thinking Engine with data from: {args.train}")
        print("✅ Training functionality coming soon!")
        print("💡 For now, please use the run_model.py script directly.")

    elif args.load:
        # Load model
        print(f"📂 Loading model from: {args.load}")
        print("✅ Model loading functionality coming soon!")

    else:
        # Default: Show help
        parser.print_help()


if __name__ == '__main__':
    main()
