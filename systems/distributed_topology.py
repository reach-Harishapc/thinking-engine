# systems/distributed_topology.py
class DistributedTopology:
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.connections = {f"Agent-{i}": [f"Agent-{(i+1)%num_agents}"] for i in range(num_agents)}

    def describe(self):
        return self.connections
