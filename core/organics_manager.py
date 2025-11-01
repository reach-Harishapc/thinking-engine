# core/organics_manager.py
import time
from typing import Dict, Any
import math

class OrganicsManager:
    """
    Tracks energy/cost estimates per agent invocation and computes reward signals.
    Reward = task_success_score - normalized_energy_cost
    """

    def __init__(self, base_costs: Dict[str, float] = None):
        # base energy cost per agent type (arbitrary units)
        self.base_costs = base_costs or {}
        self.default_cost = 1.0
        self.history = []  # list of events

    def cost_of(self, agent_name: str, complexity: float = 1.0) -> float:
        base = float(self.base_costs.get(agent_name, self.default_cost))
        return base * (1.0 + 0.5 * complexity)

    def compute_reward(self, success_score: float, agent_name: str, complexity: float = 1.0) -> float:
        c = self.cost_of(agent_name, complexity)
        # reward between -1..1
        reward = success_score - (c / (c + 1.0))
        # clamp
        reward = max(-1.0, min(1.0, reward))
        # log
        self.history.append({"ts": time.time(), "agent": agent_name, "success": success_score, "cost": c, "reward": reward})
        return reward

    def last_summary(self, n: int = 10):
        return self.history[-n:]
