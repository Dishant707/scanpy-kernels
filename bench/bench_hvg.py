"""Benchmark the Rust `pp.highly_variable_genes` kernel (Seurat flavor).

Usage:
    python bench/bench_hvg.py

Compares `scanpy_kernels.hvg_seurat` against
`scanpy.pp.highly_variable_genes(flavor="seurat")` on dense log1p matrices, and
verifies the highly-variable mask is identical on every run.

Build the extension in release mode first:
    maturin develop --release
"""

from __future__ import annotations

import time
from importlib.metadata import version

import numpy as np
import scanpy as sc
from anndata import AnnData

import scanpy_kernels


def timeit(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_case(name: str, x: np.ndarray) -> None:
    adata = AnnData(x)

    # warm-up
    sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False)
    scanpy_kernels.hvg_seurat(x)

    t_ref = timeit(lambda: sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False))
    t_kern = timeit(lambda: scanpy_kernels.hvg_seurat(x))

    ref = sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False)
    got = scanpy_kernels.hvg_seurat(x)
    np.testing.assert_array_equal(
        ref["highly_variable"].to_numpy(), got["highly_variable"].to_numpy()
    )

    speedup = t_ref / t_kern
    print(
        f"{name:24s} shape={x.shape} dtype={x.dtype.name} "
        f"ref={t_ref*1e3:8.2f}ms kern={t_kern*1e3:8.2f}ms speedup={speedup:6.2f}x"
    )


def main() -> None:
    print(f"scanpy-kernels v{scanpy_kernels.__version__}")
    print(f"numpy {np.__version__}, scanpy {version('scanpy')}\n")

    rng = np.random.default_rng(0)
    for dtype in (np.float64, np.float32):
        for shape in [(20_000, 2_000), (50_000, 1_000)]:
            counts = rng.poisson(lam=np.exp(rng.uniform(-1, 2, shape[1])), size=shape)
            x = np.log1p(counts).astype(dtype)
            bench_case(f"hvg dense {dtype.__name__}", x)


if __name__ == "__main__":
    main()
