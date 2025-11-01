# run_distributed_network.py
from thinking_engine.core.distributed_manager import DistributedManager
import numpy as np
import time

def reasoning_process(task_vector):
    # Mock reasoning
    return np.mean(task_vector) + np.random.rand() * 0.01

if __name__ == "__main__":
    manager = DistributedManager(num_agents=3, reasoning_fn=reasoning_process)
    manager.start_network()

    for i in range(3):
        manager.broadcast_task(np.random.rand(5))
        time.sleep(0.5)

    manager.stop_network()
    print("Distributed cognition session completed.")
