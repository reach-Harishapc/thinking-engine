# core/execution_engine.py
import numpy as np
from .memory import MemoryManager
from .utils import Logger

class CognitiveExecutionEngine:
    def __init__(self):
        self.memory = MemoryManager()
        self.logger = Logger("CognitiveExecutionEngine")
        self.active_context = {}

    def load_context(self, context_id: str):
        # placeholder: loads context if saved
        self.active_context = {"id": context_id}
        self.logger.log(f"Loaded context: {context_id}")

    def execute(self, task_vector, reasoning_fn):
        """Execute reasoning function using the task vector."""
        try:
            self.logger.log("Starting cognitive execution pipeline...")
            arr = np.array(task_vector, dtype=float)
            norm = arr / (np.linalg.norm(arr) + 1e-12)
            output = reasoning_fn(norm)
            # store a small trace
            self.memory.store_experience("task", str(output), meta={"vec": arr.tolist()})
            return {"status": "ok", "output": output}
        except Exception as e:
            self.logger.log(f"Execution failed: {e}", level="error")
            return {"status": "error", "error": str(e)}
