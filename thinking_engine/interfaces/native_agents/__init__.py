"""
Native agent implementations for Thinking Engine
"""

from .web_agent import Agent as WebAgent
from .code_agent import Agent as CodeAgent
from .file_agent import Agent as FileAgent
from .reasoning_agent import Agent as ReasoningAgent

__all__ = [
    'WebAgent',
    'CodeAgent',
    'FileAgent',
    'ReasoningAgent',
]
