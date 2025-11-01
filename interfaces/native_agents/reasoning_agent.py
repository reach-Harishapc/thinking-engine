# interfaces/native_agents/reasoning_agent.py
"""
Deterministic ReasoningAgent which proposes plans and uses any available learned selector when present.
"""

from typing import Dict, Any, List, Optional
import time, math
try:
    from core.learned_selector import LearnedSelector
    import numpy as np
except Exception:
    LearnedSelector = None
    np = None

def _simple_tokens(text: str):
    return [t.lower() for t in text.strip().split() if t]

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / (len(union) + 1e-9)

class Agent:
    def __init__(self, memory=None, max_trace_steps: int = 6):
        self.name = "reasoning_agent"
        self.memory = memory
        self.max_trace_steps = max_trace_steps

    def run(self, sandbox, prompt: str, context: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        t0 = time.time()
        ctx = context or {}
        trace = []
        plan = []

        tokens = _simple_tokens(prompt)
        trace.append(f"Parsed tokens ({len(tokens)})")
        intent = self._intent_hint(tokens)
        trace.append(f"Intent hint: {intent}")

        recalled = []
        if self.memory and tokens:
            mem_results = self.memory.recall(" ".join(tokens[:3]))
            for e in mem_results:
                sim = _jaccard(_simple_tokens(e.get("input","")), tokens)
                if sim > 0.05:
                    recalled.append({"exp": e, "sim": float(sim)})
            trace.append(f"Recalled {len(recalled)} items")
        else:
            trace.append("No memory or no tokens")

        # chain steps
        for i in range(min(self.max_trace_steps,6)):
            if i == 0:
                step = f"Understand: interpret as {intent}"
            else:
                step = f"Refine {i}: check memory ({len(recalled)})"
            trace.append(step)

        # basic plan candidates
        if intent == "fetch_and_summarize":
            plan = [
                {"agent":"web_agent","op":"run","args":{"query": prompt},"importance":0.9},
                {"agent":"reasoning_agent","op":"summarize","args":{"source_key":"web_fetch"},"importance":0.6},
                {"agent":"file_agent","op":"run","args":{"action":"write","path":"/home/user/summaries/auto.txt","content_key":"summary"},"importance":0.2}
            ]
        elif intent == "code_run":
            plan = [{"agent":"code_agent","op":"run","args":{"code":prompt},"importance":0.95}]
        elif intent == "edit_file":
            plan = [{"agent":"file_agent","op":"read","args":{"path":ctx.get("target_path")},"importance":0.8},
                    {"agent":"file_agent","op":"append","args":{"path":ctx.get("target_path"),"content":"\n# patched"},"importance":0.6}]
        else:
            plan = [{"agent":"reasoning_agent","op":"explain","args":{"prompt":prompt},"importance":0.4}]

        conf_score = self._compute_confidence(tokens, recalled, plan)

        # try to use LearnedSelector if present
        try:
            if LearnedSelector is not None:
                try:
                    sel = LearnedSelector.load_online_cache() if hasattr(LearnedSelector, "load_online_cache") else None
                except Exception:
                    sel = None
                if sel is None:
                    try:
                        sel = LearnedSelector.load()
                    except Exception:
                        sel = None
                if sel is not None and len(plan) > 0:
                    prompt_toks = tokens
                    probs = sel.score_candidates(plan, prompt_toks)
                    for i,p in enumerate(plan):
                        p["selector_score"] = float(probs[i])
                    plan.sort(key=lambda x: x.get("selector_score",0.0), reverse=True)
                    selector_conf = float(max(probs))
                    conf_score = min(1.0, conf_score*0.6 + selector_conf*0.4)
        except Exception:
            pass

        duration = time.time() - t0
        return {"trace": trace, "plan": plan, "confidence": round(conf_score,3), "meta":{"duration_s": round(duration,3)}}

    def _intent_hint(self, tokens: List[str]) -> str:
        s = " ".join(tokens)
        if any(w in s for w in ("search","find","summarize","look up")):
            return "fetch_and_summarize"
        if any(w in s for w in ("run code","execute","python","script")):
            return "code_run"
        if any(w in s for w in ("edit file","update file","modify")):
            return "edit_file"
        return "explain"

    def _compute_confidence(self, tokens, recalled, plan):
        recall_strength = max((r["sim"] for r in recalled), default=0.0)
        plan_strength = sum(p.get("importance",0.5) for p in plan) / max(1,len(plan))
        raw = 0.4 * recall_strength + 0.6 * plan_strength
        return 1.0 / (1.0 + math.exp(-10*(raw - 0.5)))
