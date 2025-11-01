# core/topology_optimizer.py
import time
from typing import Any, List, Dict
from .agent_router import AgentRouter
from .organics_manager import OrganicsManager
import numpy as np
import os

class TopologyOptimizer:
    """
    Periodically samples the router connections and applies small gradient updates
    based on recent feedback from OrganicsManager. This is a thin wrapper around
    AgentRouter.update_weights to implement scheduled optimization.
    """

    def __init__(self, router: AgentRouter, organics: OrganicsManager, window: int = 50):
        self.router = router
        self.organics = organics
        self.window = window

    def optimize_step(self, events: List[Dict[str, Any]], lr: float = 0.03):
        """
        events: list of {"origin":..., "target":..., "success": float, "complexity": float}
        Convert events into reward updates for router.
        """
        for ev in events[-self.window:]:
            origin = ev.get("origin")
            target = ev.get("target")
            success = float(ev.get("success", 1.0))
            complexity = float(ev.get("complexity", 1.0))
            reward = self.organics.compute_reward(success, target, complexity)
            self.router.update_weights(origin, target, reward, lr=lr)
        # persist the router
        self.router.save()
