# core/backend.py
"""
Minimal backend utilities and entry points for the Thinking Engine.
"""

import logging
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, "engine.log")),
            logging.StreamHandler()
        ]
    )
