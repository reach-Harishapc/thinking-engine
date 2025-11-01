# systems/energy_flow.py
"""
Placeholder: track energy usage among components.
"""

class EnergyFlow:
    def __init__(self):
        self.usage = {}

    def record(self, component: str, amount: float):
        self.usage[component] = self.usage.get(component, 0.0) + amount

    def summary(self):
        return self.usage
