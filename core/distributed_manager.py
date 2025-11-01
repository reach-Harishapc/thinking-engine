# core/distributed_manager.py
from .communication_bus import CommunicationBus
from .agent_node import AgentNode

class DistributedManager:
    def __init__(self, num_agents: int, reasoning_fn):
        self.bus = CommunicationBus()
        self.agents = [AgentNode(f"Agent-{i}", self.bus, reasoning_fn) for i in range(num_agents)]

    def start_network(self):
        for a in self.agents:
            a.start()

    def broadcast_task(self, data):
        self.bus.broadcast("Controller", data)

    def stop_network(self):
        for a in self.agents:
            a.stop()
            a.join(timeout=1.0)
