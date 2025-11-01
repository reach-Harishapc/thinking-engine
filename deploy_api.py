#!/usr/bin/env python3
"""
Thinking Engine Production API Server
Deploy your trained Thinking Engine model as a REST API service.

Usage:
    python deploy_api.py

The server will start on http://localhost:8080
"""

from flask import Flask, request, jsonify
from datetime import datetime
import time
import logging
import os
from run_model import ThinkingModelInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global model instance (lazy loading)
model = None

def get_model():
    """Lazy load the model on first request"""
    global model
    if model is None:
        logger.info("Loading Thinking Engine model...")
        model = ThinkingModelInterface()

        # Try to load production model, fallback to fresh instance
        model_path = os.getenv('THINKING_ENGINE_MODEL', 'models/thinking_model.think')
        try:
            model.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model file not found at {model_path}, using fresh instance")

    return model

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Test model responsiveness
        test_model = get_model()
        test_response = test_model.think("Hello")
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.1"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    try:
        # Validate input
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request"}), 400

        query = data['query'].strip()
        if not query or len(query) > 2000:  # Reasonable limit
            return jsonify({"error": "Query must be 1-2000 characters"}), 400

        # Get model and process query
        ai_model = get_model()

        # Log request
        logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")

        # Time the response
        start_time = time.time()
        response = ai_model.think(query)
        processing_time = time.time() - start_time

        # Log response time
        logger.info(".2f")

        # Return response
        return jsonify({
            "response": response,
            "query": query,
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e) if app.debug else "Please try again later"
        }), 500

@app.route("/info", methods=["GET"])
def model_info():
    """Get information about the loaded model"""
    try:
        ai_model = get_model()
        return jsonify({
            "model_type": "Thinking Engine v1.0.1",
            "capabilities": [
                "Conversational AI",
                "Python Code Education",
                "Mathematical Calculations",
                "Web Research & Analysis",
                "Professional Profile Analysis"
            ],
            "supported_queries": [
                "General conversation",
                "Python programming questions",
                "Math calculations (2+5, 10*3, etc.)",
                "Research questions",
                "Professional inquiries"
            ],
            "endpoints": {
                "/chat": "POST - Unified AI chat interface",
                "/think": "POST - Direct model reasoning",
                "/agents/web": "POST - Web search and research",
                "/agents/file": "POST - File operations",
                "/agents/code": "POST - Code execution and analysis",
                "/agents/reasoning": "POST - Logical reasoning",
                "/health": "GET - Check service health",
                "/info": "GET - Get model information"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/think", methods=["POST"])
def think_endpoint():
    """Direct model reasoning endpoint - bypasses agent routing"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request"}), 400

        query = data['query'].strip()
        if not query or len(query) > 2000:
            return jsonify({"error": "Query must be 1-2000 characters"}), 400

        ai_model = get_model()

        logger.info(f"Direct reasoning query: {query[:100]}{'...' if len(query) > 100 else ''}")

        start_time = time.time()
        # Direct cortex reasoning without agent routing
        response = ai_model.cortex.reason(query)
        processing_time = time.time() - start_time

        logger.info(".2f")

        return jsonify({
            "response": response,
            "query": query,
            "processing_time": round(processing_time, 2),
            "method": "direct_reasoning",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Think request failed: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e) if app.debug else "Please try again later"
        }), 500

@app.route("/agents/web", methods=["POST"])
def web_agent_endpoint():
    """Dedicated web agent API for search and research"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request"}), 400

        query = data['query'].strip()
        if not query or len(query) > 2000:
            return jsonify({"error": "Query must be 1-2000 characters"}), 400

        ai_model = get_model()

        logger.info(f"Web agent query: {query[:100]}{'...' if len(query) > 100 else ''}")

        start_time = time.time()
        # Direct web agent call
        result = ai_model.cortex.web_agent.run(None, query=query)
        processing_time = time.time() - start_time

        logger.info(".2f")

        return jsonify({
            "response": result.get("summary", "No results found"),
            "query": query,
            "processing_time": round(processing_time, 2),
            "agent": "web_agent",
            "status": result.get("status", "unknown"),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Web agent request failed: {e}")
        return jsonify({
            "error": "Web agent error",
            "details": str(e) if app.debug else "Please try again later"
        }), 500

@app.route("/agents/file", methods=["POST"])
def file_agent_endpoint():
    """Dedicated file agent API for file operations"""
    try:
        data = request.get_json()
        if not data or 'action' not in data:
            return jsonify({"error": "Missing 'action' field in request"}), 400

        action = data['action']
        file_path = data.get('file_path', '')
        content = data.get('content', '')

        ai_model = get_model()

        logger.info(f"File agent action: {action} on {file_path[:50]}{'...' if len(file_path) > 50 else ''}")

        start_time = time.time()
        # Direct file agent call
        result = ai_model.cortex.file_agent.run(None, action, path=file_path, content=content.encode() if content else None)
        processing_time = time.time() - start_time

        logger.info(".2f")

        return jsonify({
            "response": result,
            "action": action,
            "file_path": file_path,
            "processing_time": round(processing_time, 2),
            "agent": "file_agent",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"File agent request failed: {e}")
        return jsonify({
            "error": "File agent error",
            "details": str(e) if app.debug else "Please try again later"
        }), 500

@app.route("/agents/code", methods=["POST"])
def code_agent_endpoint():
    """Dedicated code agent API for code execution and analysis"""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({"error": "Missing 'code' field in request"}), 400

        code = data['code'].strip()
        if not code or len(code) > 10000:  # Allow larger code blocks
            return jsonify({"error": "Code must be 1-10000 characters"}), 400

        ai_model = get_model()

        logger.info(f"Code agent execution: {len(code)} characters")

        start_time = time.time()
        # Direct code agent call
        result = ai_model.cortex.code_agent.run("execute", code=code)
        processing_time = time.time() - start_time

        logger.info(".2f")

        return jsonify({
            "response": result,
            "code_length": len(code),
            "processing_time": round(processing_time, 2),
            "agent": "code_agent",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Code agent request failed: {e}")
        return jsonify({
            "error": "Code agent error",
            "details": str(e) if app.debug else "Please try again later"
        }), 500

@app.route("/agents/reasoning", methods=["POST"])
def reasoning_agent_endpoint():
    """Dedicated reasoning agent API for logical analysis"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request"}), 400

        query = data['query'].strip()
        if not query or len(query) > 2000:
            return jsonify({"error": "Query must be 1-2000 characters"}), 400

        ai_model = get_model()

        logger.info(f"Reasoning agent query: {query[:100]}{'...' if len(query) > 100 else ''}")

        start_time = time.time()
        # Direct reasoning agent call
        result = ai_model.cortex.reasoning_agent.run(None, query)
        processing_time = time.time() - start_time

        logger.info(".2f")

        return jsonify({
            "response": result,
            "query": query,
            "processing_time": round(processing_time, 2),
            "agent": "reasoning_agent",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Reasoning agent request failed: {e}")
        return jsonify({
            "error": "Reasoning agent error",
            "details": str(e) if app.debug else "Please try again later"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting Thinking Engine API server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")

    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Disable reloader in production
    )
