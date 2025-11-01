# core/utils.py
import logging
import json
import tempfile
import os

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

class Logger:
    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def log(self, msg: str, level: str = "info"):
        if level == "info":
            self._log.info(msg)
        elif level == "debug":
            self._log.debug(msg)
        elif level == "error":
            self._log.error(msg)
        else:
            self._log.warning(msg)
