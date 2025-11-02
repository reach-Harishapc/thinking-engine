# API Reference

Complete API documentation for Thinking Engine's REST API and Python interfaces.

## 🚀 REST API Overview

Thinking Engine provides a comprehensive REST API for production deployment and integration.

### Base URL
```
http://localhost:8080
```

### Authentication
Currently no authentication required (development mode). Production deployments should implement API key authentication.

### Response Format
All responses are JSON formatted:
```json
{
  "status": "success|error",
  "data": {...},
  "message": "Optional message",
  "timestamp": "ISO 8601 timestamp"
}
```

## 📡 API Endpoints

### Core Endpoints

#### POST /chat
**Unified AI chat interface**

**Request:**
```json
{
  "message": "What is machine learning?",
  "context": "educational",
  "max_tokens": 500
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "response": "Machine learning is a subset of artificial intelligence...",
    "confidence": 0.92,
    "processing_time": 0.8,
    "agent_used": "reasoning"
  },
  "timestamp": "2025-01-11T19:30:00Z"
}
```

**Parameters:**
- `message` (string, required): User query or message
- `context` (string, optional): Query context (educational, technical, creative)
- `max_tokens` (integer, optional): Maximum response length (default: 500)

#### POST /think
**Direct model reasoning**

**Request:**
```json
{
  "query": "Solve 2x + 3 = 7",
  "reasoning_type": "mathematical"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "solution": "x = 2",
    "steps": ["2x + 3 = 7", "2x = 4", "x = 2"],
    "confidence": 0.95
  }
}
```

### Agent-Specific Endpoints

#### POST /agents/web
**Web research and analysis**

**Request:**
```json
{
  "query": "latest developments in quantum computing",
  "max_sources": 5,
  "include_summary": true
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "sources": [
      {
        "url": "https://example.com/quantum-news",
        "title": "Quantum Computing Breakthrough",
        "relevance": 0.89,
        "summary": "Recent advances in quantum error correction..."
      }
    ],
    "synthesized_answer": "Recent developments include...",
    "credibility_score": 0.85
  }
}
```

#### POST /agents/code
**Code execution and analysis**

**Request:**
```json
{
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "language": "python",
  "action": "execute|analyze|optimize"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "result": "Function executed successfully",
    "output": "fibonacci(10) = 55",
    "analysis": {
      "complexity": "O(2^n)",
      "issues": ["Inefficient recursive implementation"],
      "suggestions": ["Use iterative approach or memoization"]
    }
  }
}
```

#### POST /agents/file
**File operations and analysis**

**Request:**
```json
{
  "operation": "read|write|analyze",
  "filepath": "/path/to/file.txt",
  "content": "Optional content for write operations"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "operation": "read",
    "content": "File contents...",
    "metadata": {
      "size": 1024,
      "type": "text/plain",
      "encoding": "utf-8"
    }
  }
}
```

#### POST /agents/reasoning
**Logical analysis and planning**

**Request:**
```json
{
  "problem": "Plan a machine learning project",
  "constraints": ["time: 2 weeks", "budget: $5000"],
  "objectives": ["accuracy > 90%", "deployable model"]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "plan": [
      "Step 1: Data collection and preprocessing",
      "Step 2: Model selection and training",
      "Step 3: Evaluation and optimization",
      "Step 4: Deployment and monitoring"
    ],
    "estimated_time": "12 days",
    "risk_assessment": "Medium risk due to time constraints"
  }
}
```

### System Endpoints

#### GET /health
**System health check**

**Response:**
```json
{
  "status": "success",
  "data": {
    "system_status": "healthy",
    "uptime": "2h 30m",
    "memory_usage": "256MB",
    "active_agents": 4,
    "pending_requests": 0
  }
}
```

#### GET /info
**Model and system information**

**Response:**
```json
{
  "status": "success",
  "data": {
    "version": "1.0.0",
    "model_size": "1.2MB",
    "training_epochs": 1000,
    "last_updated": "2025-01-11T19:30:00Z",
    "capabilities": [
      "text_processing",
      "code_execution",
      "web_research",
      "file_operations",
      "logical_reasoning"
    ]
  }
}
```

## 🐍 Python API

### Core Classes

#### ThinkingModelInterface

Main interface for interacting with Thinking Engine.

```python
from run_model import ThinkingModelInterface

class ThinkingModelInterface:
    def __init__(self, model_path: str = None):
        """Initialize the AI model interface.

        Args:
            model_path: Path to saved model file (.think format)
        """

    def think(self, query: str, context: str = None) -> str:
        """Process a query and return response.

        Args:
            query: User query or message
            context: Optional context (educational, technical, creative)

        Returns:
            AI-generated response
        """

    def load_model(self, filepath: str) -> bool:
        """Load a saved model from JSON file.

        Args:
            filepath: Path to .think model file

        Returns:
            True if loaded successfully
        """

    def save_model(self, filepath: str, compressed: bool = False) -> bool:
        """Save current model state to JSON file.

        Args:
            filepath: Output file path
            compressed: Whether to use gzip compression

        Returns:
            True if saved successfully
        """

    def modify_personality(self, personality_config: dict) -> bool:
        """Modify AI personality settings.

        Args:
            personality_config: Dictionary with personality settings

        Returns:
            True if modified successfully
        """

    def add_knowledge(self, knowledge_data: dict) -> bool:
        """Add new knowledge to the model.

        Args:
            knowledge_data: Knowledge to inject

        Returns:
            True if added successfully
        """
```

### Agent Classes

#### WebAgent

```python
from interfaces.native_agents.web_agent import WebAgent

class WebAgent:
    def __init__(self):
        """Initialize web research agent."""

    def research(self, query: str, max_sources: int = 5) -> dict:
        """Perform web research on a topic.

        Args:
            query: Research query
            max_sources: Maximum sources to analyze

        Returns:
            Research results with sources and analysis
        """
```

#### CodeAgent

```python
from interfaces.native_agents.code_agent import CodeAgent

class CodeAgent:
    def __init__(self):
        """Initialize code analysis agent."""

    def execute_code(self, code: str, language: str = "python") -> dict:
        """Execute code and return results.

        Args:
            code: Code to execute
            language: Programming language

        Returns:
            Execution results and analysis
        """

    def analyze_code(self, code: str) -> dict:
        """Analyze code for issues and improvements.

        Args:
            code: Code to analyze

        Returns:
            Analysis results with suggestions
        """
```

#### FileAgent

```python
from interfaces.native_agents.file_agent import FileAgent

class FileAgent:
    def __init__(self):
        """Initialize file operations agent."""

    def read_file(self, filepath: str) -> dict:
        """Read and analyze a file.

        Args:
            filepath: Path to file

        Returns:
            File contents and metadata
        """

    def write_file(self, filepath: str, content: str) -> bool:
        """Write content to a file.

        Args:
            filepath: Target file path
            content: Content to write

        Returns:
            True if written successfully
        """
```

#### ReasoningAgent

```python
from interfaces.native_agents.reasoning_agent import ReasoningAgent

class ReasoningAgent:
    def __init__(self):
        """Initialize logical reasoning agent."""

    def analyze_problem(self, problem: str, constraints: list = None) -> dict:
        """Analyze a problem and provide solution approach.

        Args:
            problem: Problem description
            constraints: List of constraints

        Returns:
            Analysis and solution approach
        """

    def plan_solution(self, objectives: list, resources: dict) -> dict:
        """Create a solution plan.

        Args:
            objectives: List of objectives
            resources: Available resources

        Returns:
            Detailed solution plan
        """
```

### Core Components

#### Cortex

```python
from core.cortex import Cortex

class Cortex:
    def __init__(self):
        """Initialize the central reasoning system."""

    def reason(self, query: str) -> str:
        """Process query through agent routing and response integration.

        Args:
            query: User query

        Returns:
            Integrated response from appropriate agents
        """

    def classify_intent(self, query: str) -> str:
        """Classify query intent for agent routing.

        Args:
            query: User query

        Returns:
            Intent classification (web, code, file, reasoning)
        """

    def select_agent(self, intent: str) -> object:
        """Select appropriate agent for intent.

        Args:
            intent: Classified intent

        Returns:
            Agent instance
        """
```

#### MemoryManager

```python
from core.memory import MemoryManager

class MemoryManager:
    def __init__(self):
        """Initialize memory management system."""

    def store_experience(self, input_text: str, output_text: str,
                        metadata: dict) -> bool:
        """Store interaction experience.

        Args:
            input_text: User input
            output_text: AI response
            metadata: Additional context data

        Returns:
            True if stored successfully
        """

    def retrieve_relevant(self, query: str, limit: int = 5) -> list:
        """Retrieve relevant past experiences.

        Args:
            query: Current query
            limit: Maximum results

        Returns:
            List of relevant experiences
        """
```

#### LearningManager

```python
from core.learning_manager import LearningManager

class LearningManager:
    def __init__(self):
        """Initialize learning management system."""

    def update_weights(self, patterns: dict, feedback: float) -> bool:
        """Update synaptic weights based on learning patterns.

        Args:
            patterns: Learned patterns
            feedback: Learning feedback signal

        Returns:
            True if updated successfully
        """

    def consolidate_learning(self) -> bool:
        """Consolidate recent learning into long-term memory.

        Returns:
            True if consolidated successfully
        """
```

## 📊 Data Formats

### Model Persistence (.think files)

```json
{
  "cortex": {
    "system_prompt": {
      "personality": "helpful and analytical",
      "communication_style": "clear and concise"
    },
    "learned_patterns": {
      "python_concepts": ["variables", "functions", "classes"],
      "mathematical_operations": ["addition", "multiplication"]
    }
  },
  "memory": {
    "experiences": [
      {
        "input": "What is machine learning?",
        "output": "Machine learning is a subset of AI...",
        "confidence": 0.92,
        "timestamp": "2025-01-11T19:30:00Z"
      }
    ]
  },
  "integrity": "sha256_hash_for_verification"
}
```

### Compressed Models (.think.gz)

Same JSON structure but gzip compressed for efficient storage and transfer.

## 🔒 Security & Best Practices

### API Security
- Implement API key authentication for production
- Use HTTPS for all communications
- Rate limit requests to prevent abuse
- Validate all input parameters

### Model Security
- Verify model integrity with SHA256 hashes
- Encrypt sensitive model sections if needed
- Implement access controls for model modifications
- Regular security audits of model files

### Data Privacy
- Don't store sensitive user data in model files
- Implement data anonymization where required
- Comply with relevant data protection regulations
- Clear user data upon request

## 🧪 Testing & Validation

### Unit Tests

```bash
# Run all unit tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_multiplatform.py

# Run with coverage
python -m pytest --cov=thinking_engine tests/
```

### Integration Tests

```bash
# Test API endpoints
python test_api.py

# Test multi-platform functionality
python run_multiplatform_tests.py
```

### Performance Benchmarks

```bash
# Run comprehensive benchmarks
python run_multiplatform_tests.py  # Select option 3

# Generate performance reports
python thinking_engine_benchmark_report.md
```

## 🚨 Error Handling

### Common Error Responses

```json
{
  "status": "error",
  "message": "Invalid request parameters",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2025-01-11T19:30:00Z"
}
```

### Error Codes

- `VALIDATION_ERROR`: Invalid request parameters
- `MODEL_NOT_FOUND`: Specified model file not found
- `AGENT_UNAVAILABLE`: Requested agent not available
- `PROCESSING_ERROR`: Internal processing error
- `RATE_LIMITED`: Too many requests
- `AUTHENTICATION_FAILED`: Invalid API key

## 📈 Rate Limits

### Default Limits
- 100 requests per minute per IP
- 1000 requests per hour per IP
- 10000 requests per day per IP

### Premium Limits (configurable)
- 1000 requests per minute
- 10000 requests per hour
- Unlimited daily requests

## 🔄 Versioning

### API Versioning
- Current version: v1
- URL format: `/v1/chat`, `/v1/agents/web`
- Backward compatibility maintained

### Model Versioning
- Models include version metadata
- Automatic migration for compatible versions
- Version validation on load

---

*This API reference provides comprehensive documentation for integrating Thinking Engine into your applications. For additional support, visit our [GitHub repository](https://github.com/reach-Harishapc/thinking-engine) or join our [community Discord](https://discord.gg/EK9A4QGtG).* 🚀
