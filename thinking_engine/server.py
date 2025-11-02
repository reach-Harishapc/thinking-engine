#!/usr/bin/env python3
"""
REST API server for Thinking Engine
"""

import sys
import argparse


def main():
    """Main server entry point"""
    parser = argparse.ArgumentParser(
        description="Thinking Engine REST API Server"
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind server to (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind server to (default: 8080)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    try:
        # Import and start the server
        from ..deploy_api import app

        print("🚀 Starting Thinking Engine API Server")
        print(f"📡 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"🐛 Debug: {args.debug}")
        print("-" * 50)
        print("📋 Available endpoints:")
        print("  POST /chat       - Unified AI chat interface")
        print("  POST /think      - Direct model reasoning")
        print("  POST /agents/web - Web research and analysis")
        print("  GET  /health     - System health check")
        print("  GET  /info       - Model information")
        print("-" * 50)

        # Start the server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )

    except ImportError as e:
        print(f"❌ Failed to import server components: {e}")
        print("💡 Make sure Thinking Engine is properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
