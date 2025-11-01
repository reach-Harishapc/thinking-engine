# interfaces/native_agents/file_agent.py
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import logging

class Agent:
    def __init__(self):
        self.name = "file_agent"

    def run(self, sandbox, action: str, path: str = None, content: bytes = None):
        safe_root = "/home/"
        if path and not Path(path).resolve().as_posix().startswith(Path(safe_root).resolve().as_posix()):
            return {"status":"error","error":"access_denied"}

        if action == "read":
            try:
                with open(path, "rb") as f:
                    data = f.read()
                return {"status":"ok","content_bytes": data}
            except Exception as e:
                return {"status":"error","error": str(e)}
        elif action == "write":
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_")
                with os.fdopen(fd, "wb") as f:
                    f.write(content or b"")
                os.replace(tmp, path)
                return {"status":"ok","path": path}
            except Exception as e:
                return {"status":"error","error": str(e)}
        elif action == "append":
            try:
                with open(path, "ab") as f:
                    f.write(content or b"")
                return {"status":"ok","path": path}
            except Exception as e:
                return {"status":"error","error": str(e)}
        elif action == "delete":
            try:
                os.remove(path)
                return {"status":"ok","path": path}
            except Exception as e:
                return {"status":"error","error": str(e)}
        else:
            return {"status":"error","error":"unknown_action"}
