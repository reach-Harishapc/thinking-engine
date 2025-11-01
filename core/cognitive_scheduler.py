# core/cognitive_scheduler.py
import numpy as np
from .utils import Logger

class CognitiveScheduler:
    def __init__(self):
        self.queue = []
        self.logger = Logger("CognitiveScheduler")

    def add_task(self, task_vector, priority: float = 1.0):
        self.queue.append((priority, task_vector))
        self.queue.sort(key=lambda x: -x[0])

    def next_task(self):
        if not self.queue:
            self.logger.log("No pending tasks.")
            return None
        pr, task = self.queue.pop(0)
        self.logger.log(f"Dispatching task with priority {pr}")
        return task
