# Getting Started with Thinking Engine

This guide will help you get up and running with Thinking Engine quickly.

## 📋 Prerequisites

- **Python 3.8+** - Required for all features
- **Git** - For cloning the repository
- **pip** - Python package manager

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/reach-Harishapc/thinking-engine.git
cd thinking-engine
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import thinking_engine; print('Installation successful!')"
```

## 🎯 Basic Usage

### Hello World Example

```python
from run_model import ThinkingModelInterface

# Initialize the AI
model = ThinkingModelInterface()

# Ask a simple question
response = model.think("What is 2 + 5?")
print(response)
# Output: The addition of 2 + 5 equals 7...
```

### Interactive Chat Mode

```bash
python run_model.py --chat
```

This starts an interactive chat session where you can converse with the AI.

### Training with Custom Data

```bash
# Train with PDF documents
python run_model.py --train /path/to/pdf/folder --save

# The system automatically:
# - Extracts text from PDF files
# - Chunks content for optimal training
# - Encodes to sparse synaptic representations
# - Updates learning weights
```

## 🧪 Testing Your Installation

### Run Basic Tests

```bash
python test_api.py
```

### Multi-Platform Testing

```bash
# Run comprehensive platform tests
python run_multiplatform_tests.py

# Test specific components:
# Option 1: Basic functionality tests
# Option 2: Platform detection
# Option 3: Full benchmarking suite
```

### PDF Processing Tests

```bash
# Test PDF processing capabilities
python test_pdf_processing.py
```

## 🚀 Production Deployment

### Start API Server

```bash
python deploy_api.py
```

The server starts on `http://localhost:8080` with the following endpoints:

- `POST /chat` - Unified AI chat interface
- `POST /think` - Direct model reasoning
- `POST /agents/web` - Web search and research
- `POST /agents/code` - Code execution and analysis
- `POST /agents/file` - File operations
- `POST /agents/reasoning` - Logical reasoning
- `GET /health` - Service health check
- `GET /info` - Model information

### API Usage Example

```python
import requests

# Chat with the AI
response = requests.post('http://localhost:8080/chat',
    json={'message': 'What is machine learning?'}
)
print(response.json())
```

## 🎛️ Model Customization

### Direct Model Surgery

```python
# Load and modify model personality
model = ThinkingModelInterface()
model.load_model('models/demo_model.think')

# Change personality
model.modify_personality({
    'identity': 'You are a creative writing assistant',
    'personality': 'imaginative and encouraging',
    'communication_style': 'engaging and inspirational'
})

# Save modified model
model.save_model('models/creative_writer.think')
```

### Knowledge Injection

```python
# Add domain expertise
model.add_knowledge({
    'domain': 'quantum_physics',
    'facts': [
        'Quantum mechanics describes nature at atomic scales',
        'Wave-particle duality is a fundamental concept',
        'Uncertainty principle limits measurement precision'
    ]
})
```

## 📊 Performance Monitoring

### Biological Learning Visualization

```bash
# Run neuron evolution demo
python neuron_evolution_demo.py

# This generates visualizations showing:
# - Real-time weight evolution
# - Neural population dynamics
# - Learning curve analysis
```

### Benchmarking

```bash
# Run comprehensive benchmarks
python run_multiplatform_tests.py  # Select option 3

# Results include:
# - Accuracy metrics across backends
# - Training time analysis
# - Hardware utilization statistics
```

## 🐛 Troubleshooting

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'core.agent_runtime'`
- **Solution**: Run `python run_model.py` to generate missing components

**Memory Error**: Training fails with out-of-memory
- **Solution**: Reduce batch size or use CPU backend
- **Alternative**: Use sparse synaptic representations

**PDF Processing Fails**: `PyPDF2` import error
- **Solution**: Install PDF dependencies: `pip install PyPDF2`

### Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/reach-Harishapc/thinking-engine/issues)
- **Documentation**: [Complete guides](https://github.com/reach-Harishapc/thinking-engine/tree/main/docs)
- **Community**: [Discord server](https://discord.gg/EK9A4QGtG)

## 🎓 Next Steps

Now that you have Thinking Engine running, explore:

1. **[Architecture Guide](Architecture.md)** - Understand system design
2. **[Biological Learning](Biological-Learning.md)** - Learn about advanced learning mechanisms
3. **[API Reference](API-Reference.md)** - Complete API documentation
4. **[Contributing](https://github.com/reach-Harishapc/thinking-engine/blob/main/CONTRIBUTING.md)** - Join the development

## 📞 Support

- **Documentation**: [docs/index.html](https://github.com/reach-Harishapc/thinking-engine/blob/main/docs/index.html)
- **Research Paper**: [arxiv_submission/](https://github.com/reach-Harishapc/thinking-engine/tree/main/arxiv_submission)
- **Community**: [Discord](https://discord.gg/EK9A4QGtG)

---

*Ready to build transparent, ethical AI? Let's get started!* 🚀
