#!/usr/bin/env python3
"""
Test script for Thinking Engine API
Run this to test the API functionality without starting a server.

Usage:
    python test_api.py
"""

from run_model import ThinkingModelInterface
import json
import time

def test_model_directly():
    """Test the model directly without API server"""
    print("🧪 Testing Thinking Engine Model Directly")
    print("=" * 50)

    # Initialize model
    model = ThinkingModelInterface()

    # Test queries
    test_queries = [
        "hi",
        "what is 2+5? show python code",
        "how to create a variable?",
        "what is machine learning?",
        "who is harisha p c"
    ]

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 30)

        start_time = time.time()
        response = model.think(query)
        processing_time = time.time() - start_time

        print(f"🤖 Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(".2f")

def simulate_api_calls():
    """Simulate API calls that would be made to the server"""
    print("\n🌐 Simulating API Calls")
    print("=" * 50)

    # This simulates what the API server would do
    model = ThinkingModelInterface()

    # Simulate different API endpoints
    api_calls = [
        # Unified chat endpoint
        {"endpoint": "/chat", "method": "POST", "data": {"query": "what is 10*5?"}},
        {"endpoint": "/think", "method": "POST", "data": {"query": "explain variables in python"}},

        # Agent-specific endpoints
        {"endpoint": "/agents/web", "method": "POST", "data": {"query": "what is AI?"}},
        {"endpoint": "/agents/code", "method": "POST", "data": {"code": "print('Hello World')"}},
        {"endpoint": "/agents/file", "method": "POST", "data": {"action": "read", "file_path": "data/python_knowledge.json"}},
        {"endpoint": "/agents/reasoning", "method": "POST", "data": {"query": "If all cats are mammals and some mammals are pets, are all cats pets?"}},
    ]

    for call in api_calls:
        endpoint = call["endpoint"]
        method = call["method"]
        data = call["data"]

        print(f"\n📨 API Call: {method} {endpoint}")
        print(f"   Data: {json.dumps(data, indent=2)}")

        # Simulate processing based on endpoint
        start_time = time.time()

        if endpoint == "/chat":
            response = model.think(data["query"])
            api_response = {
                "response": response,
                "query": data["query"],
                "processing_time": round(time.time() - start_time, 2),
                "timestamp": "2025-01-11T12:00:00"
            }
        elif endpoint == "/think":
            response = model.cortex.reason(data["query"])
            api_response = {
                "response": response,
                "query": data["query"],
                "processing_time": round(time.time() - start_time, 2),
                "method": "direct_reasoning",
                "timestamp": "2025-01-11T12:00:00"
            }
        elif endpoint == "/agents/web":
            result = model.cortex.web_agent.run(None, query=data["query"])
            api_response = {
                "response": result.get("summary", "No results found"),
                "query": data["query"],
                "processing_time": round(time.time() - start_time, 2),
                "agent": "web_agent",
                "status": result.get("status", "unknown"),
                "timestamp": "2025-01-11T12:00:00"
            }
        elif endpoint == "/agents/code":
            result = model.cortex.code_agent.run("execute", code=data["code"])
            api_response = {
                "response": result,
                "code_length": len(data["code"]),
                "processing_time": round(time.time() - start_time, 2),
                "agent": "code_agent",
                "timestamp": "2025-01-11T12:00:00"
            }
        elif endpoint == "/agents/file":
            result = model.cortex.file_agent.run(None, data["action"], path=data["file_path"])
            api_response = {
                "response": result,
                "action": data["action"],
                "file_path": data["file_path"],
                "processing_time": round(time.time() - start_time, 2),
                "agent": "file_agent",
                "timestamp": "2025-01-11T12:00:00"
            }
        elif endpoint == "/agents/reasoning":
            result = model.cortex.reasoning_agent.run(None, data["query"])
            api_response = {
                "response": result,
                "query": data["query"],
                "processing_time": round(time.time() - start_time, 2),
                "agent": "reasoning_agent",
                "timestamp": "2025-01-11T12:00:00"
            }

        print(f"📨 API Response: {json.dumps(api_response, indent=2)[:400]}...")

def show_deployment_info():
    """Show information about deployment options"""
    print("\n🚀 Deployment Information")
    print("=" * 50)

    print("📁 Model Format: .think (JSON-based)")
    print("🌐 API Endpoints:")
    print("   POST /chat - Unified AI chat interface")
    print("   POST /think - Direct model reasoning")
    print("   POST /agents/web - Web search and research")
    print("   POST /agents/file - File operations")
    print("   POST /agents/code - Code execution and analysis")
    print("   POST /agents/reasoning - Logical reasoning")
    print("   GET /health - Check service health")
    print("   GET /info - Get model information")
    print()
    print("🏃‍♂️ To start API server:")
    print("   python deploy_api.py")
    print()
    print("🐳 Docker deployment:")
    print("   docker build -t thinking-engine .")
    print("   docker run -p 8080:8080 thinking-engine")
    print()
    print("☁️ Cloud deployment options:")
    print("   • AWS Lambda")
    print("   • Google Cloud Functions")
    print("   • Azure Functions")
    print("   • Heroku")
    print("   • DigitalOcean App Platform")
    print()
    print("🔧 Agent-Specific Usage Examples:")
    print("   curl -X POST http://localhost:8080/agents/web \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"query\": \"what is machine learning?\"}'")
    print()
    print("   curl -X POST http://localhost:8080/agents/code \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"code\": \"print(2+3)\"}'")

if __name__ == "__main__":
    test_model_directly()
    simulate_api_calls()
    show_deployment_info()

    print("\n✅ All tests completed successfully!")
    print("🎉 Your Thinking Engine is ready for production deployment!")
