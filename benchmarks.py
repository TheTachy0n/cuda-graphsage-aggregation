"""Validate and benchmark CSR GraphSAGE mean aggregation implementations."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.aggregation import csr_mean_cpu, csr_mean_pytorch, make_uniform_csr


def median_ms_cuda(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return statistics.median(elapsed)


def median_ms_cpu(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup): fn()
    samples = []
    for _ in range(runs):
        begin = time.perf_counter(); fn(); samples.append((time.perf_counter() - begin) * 1_000)
    return statistics.median(samples)


def write_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    columns = list(rows[0])
    with (output_dir / "benchmark_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    labels = [row["implementation"] for row in rows]
    values = [float(row["median_ms"]) for row in rows]
    plt.figure(figsize=(8, 4)); plt.bar(labels, values, color=["#777777", "#ee9b00", "#76b900", "#2a9d8f"][:len(rows)])
    plt.ylabel("Median latency (ms)"); plt.title("CSR GraphSAGE mean aggregation")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.savefig(output_dir / "benchmark_plot.png", dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--avg-degree", type=int, default=20)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    graph = make_uniform_csr(args.nodes, args.avg_degree)
    features_np = np.random.default_rng(42).standard_normal((args.nodes, args.features)).astype(np.float32)
    cpu_ms = median_ms_cpu(lambda: csr_mean_cpu(graph, features_np), max(1, args.warmup // 2), min(args.runs, 10))
    rows = [{"implementation": "NumPy CPU", "median_ms": f"{cpu_ms:.4f}", "edges_per_second": f"{graph.num_edges / (cpu_ms / 1_000):.2f}"}]
    print(f"NumPy CPU: {cpu_ms:.3f} ms")

    if args.cpu_only:
        write_results(rows, Path("results")); return
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Run with --cpu-only locally or enable a Kaggle GPU.")

    from src.cuda_extension import load_extension
    device = torch.device("cuda")
    offsets = torch.from_numpy(graph.row_offsets).to(device)
    indices = torch.from_numpy(graph.col_indices).to(device)
    features = torch.from_numpy(features_np).to(device)
    extension = load_extension()
    reference = csr_mean_pytorch(offsets, indices, features)
    implementations = [
        ("PyTorch index_add", lambda: csr_mean_pytorch(offsets, indices, features)),
        ("CUDA naive", lambda: extension.mean_aggregate_naive(offsets, indices, features)),
        ("CUDA coalesced", lambda: extension.mean_aggregate_coalesced(offsets, indices, features)),
    ]
    for name, fn in implementations:
        result = fn()
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4, msg=f"{name} failed correctness")
        latency = median_ms_cuda(fn, args.warmup, args.runs)
        throughput = graph.num_edges / (latency / 1_000)
        rows.append({"implementation": name, "median_ms": f"{latency:.4f}", "edges_per_second": f"{throughput:.2f}"})
        print(f"{name}: {latency:.3f} ms, {throughput / 1e6:.2f} M edges/s")
    print("GPU:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    write_results(rows, Path("results"))


if __name__ == "__main__":
    main()
