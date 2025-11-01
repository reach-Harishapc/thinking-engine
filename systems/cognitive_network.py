# systems/cognitive_network.py
"""
High-level graph describing connections between internal modules.
"""

class CognitiveNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, name):
        self.nodes[name] = {}

    def connect(self, a, b):
        self.edges.append((a, b))
