# core/memory.py
"""
Very small MemoryManager storing experiences in a JSONL file.
"""

import os, json, time
from typing import List, Dict

MEM_DIR = "memory_store"
os.makedirs(MEM_DIR, exist_ok=True)
MEM_FILE = os.path.join(MEM_DIR, "experiences.jsonl")

class MemoryManager:
    def __init__(self, path: str = MEM_FILE):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def store_experience(self, input_text: str, output_text: str, meta: Dict = None):
        rec = {"ts": time.time(), "input": input_text, "output": output_text, "meta": meta or {}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def recall(self, query: str, limit: int = 10) -> List[Dict]:
        # naive: return last N experiences containing any token
        toks = set(query.split())
        res = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                try:
                    r = json.loads(line)
                    if toks & set(r.get("input","").split()):
                        res.append(r)
                        if len(res) >= limit:
                            break
                except Exception:
                    continue
        return res

    def export_state(self):
        # Export memory state as dict
        return {"path": self.path}

    def import_state(self, state):
        # Import memory state
        self.path = state.get("path", MEM_FILE)
