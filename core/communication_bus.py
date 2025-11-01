# core/communication_bus.py
from queue import Queue
from threading import Lock

class CommunicationBus:
    def __init__(self):
        self._q = Queue()
        self._lock = Lock()

    def broadcast(self, sender_id, data):
        with self._lock:
            self._q.put((sender_id, data))

    def receive(self):
        try:
            return self._q.get_nowait()
        except Exception:
            return None
