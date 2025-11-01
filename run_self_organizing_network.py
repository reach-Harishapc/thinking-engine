# run_self_organizing_network.py
from core.agent_router import AgentRouter
from core.organics_manager import OrganicsManager
from core.topology_optimizer import TopologyOptimizer
import numpy as np
import time

def demo():
    agents = ["Agent-0","Agent-1","Agent-2"]
    router = AgentRouter(agents)
    organics = OrganicsManager(base_costs={"Agent-0":1.0,"Agent-1":0.7,"Agent-2":1.2})
    optimizer = TopologyOptimizer(router, organics)
    for t in range(20):
        origin = agents[t % len(agents)]
        targets = router.sample_targets(origin, top_k=1)
        target = targets[0] if targets else agents[(t+1)%3]
        success = 0.9 if target == "Agent-1" else 0.5
        complexity = float(np.random.rand()*2.0)
        event = {"origin": origin, "target": target, "success": success, "complexity": complexity}
        optimizer.optimize_step([event], lr=0.1)
        if t % 5 == 0:
            print(f"[t={t}] probs {origin}: {router.get_probs(origin)}")
        time.sleep(0.05)
    router.save()
    print("Self-organizing demo finished.")

if __name__ == "__main__":
    demo()
