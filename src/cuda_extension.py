"""Build and expose the custom CUDA extension on CUDA-capable machines."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to build the custom extension.")
    source_dir = Path(__file__).parent / "csrc"
    return load(
        name="graphsage_aggregation_cuda",
        sources=[str(source_dir / "bindings.cpp"), str(source_dir / "aggregation_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=True,
    )
