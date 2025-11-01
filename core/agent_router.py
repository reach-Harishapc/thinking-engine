# core/agent_router.py
import os
import json
import tempfile
import numpy as np

TOPOLOGY_PATH = os.path.join("storage", "policies", "topology_weights.json")
os.makedirs(os.path.dirname(TOPOLOGY_PATH), exist_ok=True)

def _atomic_write_json(path: str, obj):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

class AgentRouter:
    def __init__(self, agents):
        self.agents = list(agents)
        n = len(self.agents)
        self.w = np.ones((n,n), dtype=np.float32) * 0.1
        for i in range(n):
            self.w[i,i] = 0.05
        self.agent_to_idx = {a:i for i,a in enumerate(self.agents)}

    def save(self, path: str = None):
        path = path or TOPOLOGY_PATH
        payload = {"agents": self.agents, "weights": self.w.tolist()}
        _atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str = None):
        path = path or TOPOLOGY_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = cls(data["agents"])
        inst.w = np.array(data["weights"], dtype=np.float32)
        inst.agent_to_idx = {a:i for i,a in enumerate(inst.agents)}
        return inst

    def sample_targets(self, origin: str, top_k: int = 1, temperature: float = 1.0):
        if origin not in self.agent_to_idx:
            return []
        i = self.agent_to_idx[origin]
        scores = self.w[i] / max(1e-6, float(temperature))
        ex = np.exp(scores - np.max(scores))
        probs = ex / (np.sum(ex) + 1e-12)
        idxs = list(np.argsort(-probs)[:top_k])
        return [self.agents[j] for j in idxs]

    def get_probs(self, origin: str):
        if origin not in self.agent_to_idx:
            return {}
        i = self.agent_to_idx[origin]
        ex = np.exp(self.w[i] - np.max(self.w[i]))
        probs = ex / (np.sum(ex) + 1e-12)
        return {self.agents[j]: float(probs[j]) for j in range(len(self.agents))}

    def update_weights(self, origin: str, target: str, reward: float, lr: float = 0.05):
        if origin not in self.agent_to_idx or target not in self.agent_to_idx:
            return
        oi = self.agent_to_idx[origin]
        ti = self.agent_to_idx[target]
        ex = np.exp(self.w[oi] - np.max(self.w[oi]))
        probs = ex / (np.sum(ex) + 1e-12)
        grad = np.zeros_like(self.w[oi])
        grad[ti] = 1.0 - probs[ti]
        self.w[oi] += lr * reward * grad
        self.w = np.clip(self.w, 1e-6, None)

    def add_agent(self, agent_name: str):
        if agent_name in self.agent_to_idx:
            return
        self.agents.append(agent_name)
        n = len(self.agents)
        new_w = np.ones((n,n), dtype=np.float32) * 0.01
        new_w[:n-1,:n-1] = self.w
        new_w[n-1,n-1] = 0.01
        self.w = new_w
        self.agent_to_idx = {a:i for i,a in enumerate(self.agents)}
