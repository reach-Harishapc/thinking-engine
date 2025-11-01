# tests/test_router.py
from thinking_engine.core.agent_router import AgentRouter
def test_router_basic():
    agents = ["A","B","C"]
    r = AgentRouter(agents)
    probs = r.get_probs("A")
    assert set(probs.keys()) == set(agents)
    targ = r.sample_targets("A", top_k=2)
    assert len(targ) == 2
    # update a weight and ensure it changes distribution
    prior = r.get_probs("A")["B"]
    r.update_weights("A","B", reward=1.0, lr=0.5)
    post = r.get_probs("A")["B"]
    assert post >= prior
