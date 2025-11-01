# tests/test_topology_optimizer.py
from thinking_engine.core.agent_router import AgentRouter
from thinking_engine.core.organics_manager import OrganicsManager
from thinking_engine.core.topology_optimizer import TopologyOptimizer

def test_optimizer_updates():
    agents = ["A","B","C"]
    r = AgentRouter(agents)
    org = OrganicsManager(base_costs={"A":1.0,"B":2.0,"C":0.5})
    opt = TopologyOptimizer(r, org)
    events = [{"origin":"A","target":"B","success":1.0,"complexity":1.0}]*5
    opt.optimize_step(events, lr=0.1)
    # ensure weight updated
    probs = r.get_probs("A")
    assert isinstance(probs, dict)
