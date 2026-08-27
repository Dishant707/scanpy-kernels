# scanpy-kernels

Rust-accelerated, numerically-verified kernels for [Scanpy](https://scanpy.readthedocs.io/) — single-cell RNA-seq analysis.

**Correctness-first**: every kernel must reproduce the reference Scanpy/NumPy
output within a strict tolerance (`rtol=1e-7` for `float64`, `rtol=1e-4` for
`float32`), be deterministic, and be validated against both synthetic and real
data.

## Implemented kernels

| Kernel | Status |
|--------|--------|
| `pp.scale` (per-gene standardization) | ✅ dense `float32`/`float64`/integer |

## Install (development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

## Usage

```python
import numpy as np
import scanpy_kernels

X = np.random.default_rng(0).standard_normal((1000, 100))
X_scaled = scanpy_kernels.scale(X, zero_center=True, max_value=10.0)
```

The public function mirrors `scanpy.preprocessing._scale.scale_array`:

```
scanpy_kernels.scale(x, *, zero_center=True, max_value=None, return_mean_std=False)
```

## Verify

```bash
pytest tests/
```

## Benchmark

```bash
python bench/bench_scale.py
```

First-milestone results (Apple Silicon, release build) — see
[`BENCHMARKS.md`](BENCHMARKS.md) for full details:

- `float64` dense: **~5×** speedup
- `float32` dense: **~8–10×** speedup

## Correctness notes

The kernel reproduces the reference *exactly*, including two non-obvious
behaviours of Scanpy 1.12.3's `scale` (documented and covered by tests):

1. Variance uses the naive `E[X^2] - E[X]^2` formula (not two-pass), which can
   round to a tiny negative number for constant genes with non-exactly
   representable values — the reference then emits `NaN`, and so does this
   kernel.
2. For `float32` input, `x**2` is computed in `float32` before being summed in
   `float64`; the kernel matches this dtype behaviour.
