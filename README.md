# scanpy-kernels

Rust-accelerated, numerically-verified kernels for [Scanpy](https://scanpy.readthedocs.io/) — single-cell RNA-seq analysis.

**Correctness-first**: every kernel must reproduce the reference Scanpy/NumPy
output within a strict tolerance (`rtol=1e-7` for `float64`, `rtol=1e-4` for
`float32`), be deterministic, and be validated against both synthetic and real
data.

## Implemented kernels

| Kernel | Status |
|--------|--------|
| `pp.scale` (per-gene standardization) | ✅ dense + sparse (CSR/CSC), `float32`/`float64`/integer |
| `pp.highly_variable_genes` (Seurat flavor) | ✅ dense + sparse (CSR/CSC), `float32`/`float64`/integer |

## Install

### For researchers (prebuilt wheels, from PyPI)

```bash
pip install scanpy-kernels
```

No Rust or compiler needed. Prebuilt wheels are built for Linux, macOS
(Intel + Apple Silicon) and Windows via GitHub Actions on every tagged release.

Then add two lines to any existing Scanpy script:

```python
import scanpy_kernels
scanpy_kernels.install()   # pp.scale & pp.highly_variable_genes now use Rust
```

### For development (from source)

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

# Highly variable genes (Seurat flavor) on log1p data
hvg = scanpy_kernels.hvg_seurat(np.log1p(X), n_bins=20)
```

The public functions mirror Scanpy and accept dense NumPy arrays or scipy sparse
(CSR/CSC) matrices:

```
scanpy_kernels.scale(x, *, zero_center=True, max_value=None, return_mean_std=False)
scanpy_kernels.hvg_seurat(x, *, n_bins=20, n_top_genes=None, min_disp=0.5,
                          max_disp=inf, min_mean=0.0125, max_mean=3.0, log_base=None)
```

### Drop-in integration

Make existing Scanpy pipelines use the kernels transparently (with automatic
fallback to Scanpy for anything unsupported):

```python
import scanpy_kernels

scanpy_kernels.install()   # patches scanpy.pp.scale and scanpy.pp.highly_variable_genes
# ... your normal scanpy.pp.scale(...) / scanpy.pp.highly_variable_genes(...) calls ...
scanpy_kernels.uninstall()
```

## Verify

```bash
pytest tests/
```

## Real-data validation

`bench/validate_pbmc3k.py` runs the kernels end-to-end on the published PBMC 3k
dataset and confirms identical results: same HVG selection, bit-identical
scaled matrix, and identical Leiden clusters (13 clusters).

## Benchmark

```bash
python bench/bench_scale.py
python bench/bench_hvg.py
```

First-milestone results (Apple Silicon, release build) — see
[`BENCHMARKS.md`](BENCHMARKS.md) for full details:

- `pp.scale` dense: **~5×** (float64), **~8–10×** (float32)
- `pp.highly_variable_genes` dense: **~7–9×**

## Correctness notes

The kernel reproduces the reference *exactly*, including two non-obvious
behaviours of Scanpy 1.12.3's `scale` (documented and covered by tests):

1. Variance uses the naive `E[X^2] - E[X]^2` formula (not two-pass), which can
   round to a tiny negative number for constant genes with non-exactly
   representable values — the reference then emits `NaN`, and so does this
   kernel.
2. For `float32` input, `x**2` is computed in `float32` before being summed in
   `float64`; the kernel matches this dtype behaviour.
