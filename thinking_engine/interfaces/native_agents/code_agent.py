# interfaces/native_agents/code_agent.py
import tempfile
import subprocess
import os
from datetime import datetime

class Agent:
    def __init__(self):
        self.name = "code_agent"

    def run(self, sandbox, code: str = None, path: str = None, mode: str = "run"):
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return {"status":"error","error": str(e)}

        if not code:
            return {"status":"error","error":"no_code"}

        if mode == "run":
            # execute in a temp file via subprocess
            tmp = None
            try:
                fd, tmp = tempfile.mkstemp(suffix=".py")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(code)
                res = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=5)
                return {"status":"ok","stdout": res.stdout, "stderr": res.stderr, "rc": res.returncode, "ts": datetime.utcnow().isoformat()}
            except subprocess.TimeoutExpired:
                return {"status":"error","error":"timeout"}
            except Exception as e:
                return {"status":"error","error": str(e)}
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
        elif mode == "analyze":
            # quick static summary
            lines = code.splitlines()
            funcs = [l for l in lines if l.strip().startswith("def ")]
            imports = [l for l in lines if l.strip().startswith("import ")]
            return {"status":"ok","analysis": {"functions": funcs, "imports": imports, "lines": len(lines)}}
        else:
            return {"status":"error","error":"unknown_mode"}
