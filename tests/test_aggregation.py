import numpy as np
import torch

from src.aggregation import csr_mean_cpu, csr_mean_pytorch, make_uniform_csr


def test_cpu_and_pytorch_agree_on_csr_mean():
    graph = make_uniform_csr(11, 3, seed=11)
    features = np.random.default_rng(4).normal(size=(11, 5)).astype(np.float32)
    expected = csr_mean_cpu(graph, features)
    actual = csr_mean_pytorch(torch.from_numpy(graph.row_offsets), torch.from_numpy(graph.col_indices), torch.from_numpy(features))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_empty_neighbor_row_returns_zero():
    offsets = torch.tensor([0, 2, 2], dtype=torch.int64)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    features = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    actual = csr_mean_pytorch(offsets, indices, features)
    torch.testing.assert_close(actual, torch.tensor([[4.0, 6.0], [0.0, 0.0]]))
