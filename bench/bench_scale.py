"""Benchmark the Rust `pp.scale` kernel against reference Scanpy/NumPy.

Usage:
    python bench/bench_scale.py

Runs on dense float64 and float32 matrices, reports wall-clock time and speedup,
and verifies numerical equivalence on every run.

Build the extension in release mode first:
    maturin develop --release
"""

from __future__ import annotations

import time

import numpy as np
from scanpy.preprocessing._scale import scale_array as reference_scale

import scanpy_kernels


def timeit(fn, repeats: int = 3) -> float:
    """Return the best (minimum) wall-clock time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_case(
    name: str,
    x: np.ndarray,
    zero_center: bool = True,
    max_value: float | None = None,
) -> None:
    # warm-up (also triggers any JIT compilation in the reference)
    reference_scale(x.copy(), zero_center=zero_center, max_value=max_value)
    scanpy_kernels.scale(x, zero_center=zero_center, max_value=max_value)

    t_ref = timeit(lambda: reference_scale(x.copy(), zero_center=zero_center, max_value=max_value))
    t_kern = timeit(lambda: scanpy_kernels.scale(x, zero_center=zero_center, max_value=max_value))

    ref = reference_scale(x.copy(), zero_center=zero_center, max_value=max_value)
    got = scanpy_kernels.scale(x, zero_center=zero_center, max_value=max_value)
    tol = 1e-4 if x.dtype == np.float32 else 1e-7
    np.testing.assert_allclose(got, ref, rtol=tol, atol=1e-6, equal_nan=True)

    speedup = t_ref / t_kern
    print(
        f"{name:24s} shape={x.shape} dtype={x.dtype.name} "
        f"ref={t_ref*1e3:8.2f}ms kern={t_kern*1e3:8.2f}ms speedup={speedup:6.2f}x"
    )


def main() -> None:
    print(f"scanpy-kernels v{scanpy_kernels.__version__}  (extension: {scanpy_kernels.__file__})")
    print(f"numpy {np.__version__}\n")

    rng = np.random.default_rng(0)
    cases: list[tuple[str, np.ndarray]] = [
        ("dense f64 20k x 2k", rng.standard_normal((20_000, 2_000))),
        ("dense f64 100k x 1k", rng.standard_normal((100_000, 1_000))),
        ("dense f32 20k x 2k", rng.standard_normal((20_000, 2_000)).astype(np.float32)),
        ("dense f32 100k x 1k", rng.standard_normal((100_000, 1_000)).astype(np.float32)),
    ]

    for name, x in cases:
        bench_case(name, x)

    # Clipping adds a pass; check it doesn't regress correctness or speed badly.
    x = rng.standard_normal((20_000, 2_000))
    bench_case("dense f64 + clip", x, max_value=10.0)


if __name__ == "__main__":
    main()
