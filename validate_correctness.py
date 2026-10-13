"""Reproduce CSR edge-case comparisons; pass --cuda to include both kernels."""
import argparse

import numpy as np
import torch

from src.aggregation import CSRGraph, csr_mean_cpu, csr_mean_pytorch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requires an available NVIDIA GPU")
    device = "cuda" if args.cuda else "cpu"
    implementations = {"PyTorch": csr_mean_pytorch}
    if args.cuda:
        from src.cuda_extension import load_extension
        extension = load_extension()
        implementations.update({
            "CUDA naive": extension.mean_aggregate_naive,
            "CUDA coalesced": extension.mean_aggregate_coalesced,
        })
    rng = np.random.default_rng(123)
    cases = {
        "all_empty": [0] * 8,
        "mixed": [0, 1, 2, 0, 7, 3, 1, 12],
        "high_degree": [0, 1, 1000, 2, 31, 0, 64, 3],
    }
    comparisons = 0
    for case, degrees in cases.items():
        offsets_np = np.concatenate(([0], np.cumsum(degrees))).astype(np.int64)
        indices_np = rng.integers(0, len(degrees), size=int(offsets_np[-1]), dtype=np.int64)
        graph = CSRGraph(offsets_np, indices_np)
        for width in [1, 7, 33, 128, 257]:
            features_np = rng.standard_normal((len(degrees), width)).astype(np.float32)
            offsets = torch.from_numpy(offsets_np).to(device)
            indices = torch.from_numpy(indices_np).to(device)
            features = torch.from_numpy(features_np).to(device)
            expected = torch.from_numpy(csr_mean_cpu(graph, features_np)).to(device)
            for name, fn in implementations.items():
                torch.testing.assert_close(
                    fn(offsets, indices, features), expected,
                    rtol=1e-4, atol=1e-4, msg=f"{name}: {case}, width={width}",
                )
                comparisons += 1
        print(f"PASS: {case}, all 5 feature widths")
    print(f"All {comparisons} correctness comparisons passed.")


if __name__ == "__main__":
    main()
