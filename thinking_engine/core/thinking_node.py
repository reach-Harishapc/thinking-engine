# core/thinking_node.py
"""
High-level ThinkingNode: coordinates cortex, execution engine and scheduler.
"""

from .cortex import Cortex
from .execution_engine import CognitiveExecutionEngine
from .cognitive_scheduler import CognitiveScheduler

class ThinkingNode:
    def __init__(self):
        self.cortex = Cortex()
        self.engine = CognitiveExecutionEngine()
        self.scheduler = CognitiveScheduler()

    def handle_request(self, prompt: str):
        # Cortex decides which agent/plan to propose
        decision = self.cortex.think(prompt)
        # schedule a task vector derived from decision
        task_vec = self.cortex.vectorize(prompt)
        self.scheduler.add_task(task_vec, priority=1.0)
        # execute immediate if needed
        task = self.scheduler.next_task()
        if task is not None:
            return self.engine.execute(task, lambda v: decision)
        return {"status":"queued"}
