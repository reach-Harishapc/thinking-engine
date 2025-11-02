"""
Thinking Engine - Transparent Cognitive AI Framework

A revolutionary AI framework that combines biological learning mechanisms
with transparent model persistence and multi-agent intelligence.
"""

__version__ = "1.0.0"
__author__ = "Harisha P C"
__email__ = "reach.harishapc@gmail.com"
__license__ = "Apache 2.0"

from .core.cortex import Cortex
from .core.memory import MemoryManager
from .core.learning_manager import LearningManager
from .interfaces import WebAgent, CodeAgent, FileAgent, ReasoningAgent

__all__ = [
    'Cortex',
    'MemoryManager',
    'LearningManager',
    'WebAgent',
    'CodeAgent',
    'FileAgent',
    'ReasoningAgent',
]
