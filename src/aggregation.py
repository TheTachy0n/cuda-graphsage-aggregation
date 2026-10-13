"""CSR graph helpers and reference GraphSAGE mean aggregation implementations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CSRGraph:
    row_offsets: np.ndarray  # int64, shape [num_nodes + 1]
    col_indices: np.ndarray  # int64, shape [num_edges]

    @property
    def num_nodes(self) -> int:
        return len(self.row_offsets) - 1

    @property
    def num_edges(self) -> int:
        return len(self.col_indices)


def make_uniform_csr(num_nodes: int, avg_degree: int, seed: int = 7) -> CSRGraph:
    """Create a deterministic directed CSR graph with fixed out-degree."""
    if num_nodes <= 0 or avg_degree <= 0:
        raise ValueError("num_nodes and avg_degree must be positive")
    rng = np.random.default_rng(seed)
    row_offsets = np.arange(num_nodes + 1, dtype=np.int64) * avg_degree
    col_indices = rng.integers(0, num_nodes, size=num_nodes * avg_degree, dtype=np.int64)
    return CSRGraph(row_offsets=row_offsets, col_indices=col_indices)


def csr_mean_cpu(graph: CSRGraph, features: np.ndarray) -> np.ndarray:
    """Straightforward, correctness-first NumPy reference."""
    if features.shape[0] != graph.num_nodes:
        raise ValueError("features must have one row per graph node")
    out = np.zeros_like(features)
    for node in range(graph.num_nodes):
        start, end = graph.row_offsets[node], graph.row_offsets[node + 1]
        if end > start:
            out[node] = features[graph.col_indices[start:end]].mean(axis=0)
    return out


def csr_mean_pytorch(row_offsets: torch.Tensor, col_indices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch baseline using indexed reduction."""
    degrees = row_offsets[1:] - row_offsets[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(features.shape[0], device=features.device, dtype=torch.long), degrees
    )
    out = torch.zeros_like(features)
    out.index_add_(0, row_ids, features.index_select(0, col_indices))
    return out / degrees.clamp_min(1).to(features.dtype).unsqueeze(1)
