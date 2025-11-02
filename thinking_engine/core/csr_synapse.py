# core/csr_synapse.py
"""
Lightweight CSR synapse placeholder for sparse structures.
This is a small utility used by thinking_node or learning_manager.
"""

import numpy as np
from typing import Tuple

def build_csr(indices: np.ndarray, values: np.ndarray, shape: Tuple[int,int]):
    """
    Build a very simple CSR-like structure (row_ptr, col_idx, data)
    Expects indices shape (N,2) rows: [r, c]
    """
    if indices.size == 0:
        return {"row_ptr": np.zeros(shape[0]+1, dtype=int).tolist(), "col_idx": [], "data": []}
    rows = indices[:,0].astype(int)
    cols = indices[:,1].astype(int)
    order = np.argsort(rows)
    rows = rows[order]
    cols = cols[order]
    data = values[order]
    row_ptr = [0]
    current = 0
    for r in range(shape[0]):
        count = int((rows == r).sum())
        current += count
        row_ptr.append(current)
    return {"row_ptr": row_ptr, "col_idx": cols.tolist(), "data": data.tolist()}
