# core/agent_node.py
import numpy as np
from threading import Thread
from .execution_engine import CognitiveExecutionEngine
import time

class AgentNode(Thread):
    def __init__(self, agent_id: str, bus, reasoning_fn):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.bus = bus
        self.engine = CognitiveExecutionEngine()
        self.reasoning_fn = reasoning_fn
        self.running = True

    def run(self):
        while self.running:
            msg = self.bus.receive()
            if msg:
                sender, data = msg
                if sender == self.agent_id:
                    continue
                # execute reasoning and broadcast result
                result = self.engine.execute(data, self.reasoning_fn)
                # send back result
                self.bus.broadcast(self.agent_id, result)
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False
