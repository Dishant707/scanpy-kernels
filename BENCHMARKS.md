# Benchmark results — `pp.scale`

Machine: Apple Silicon (arm64), macOS. Reference: Scanpy 1.12.3
(`scanpy.preprocessing._scale.scale_array`), NumPy 2.5.2, Rust 1.97.1, PyO3 0.23,
maturin 1.14.1. Extension built with `maturin develop --release` (`lto = true`,
`codegen-units = 1`).

Each case is the best (minimum) wall-clock time over 3 runs after a warm-up.
Numerical equivalence is asserted on every run before timing is reported.

| Case | Shape | dtype | reference | kernel | speedup |
|------|-------|-------|-----------|--------|---------|
| dense f64 | 20k × 2k | float64 | 78.4 ms | 16.1 ms | 4.9× |
| dense f64 | 100k × 1k | float64 | 205 ms | 38.3 ms | 5.4× |
| dense f32 | 20k × 2k | float32 | 84.4 ms | 10.2 ms | 8.3× |
| dense f32 | 100k × 1k | float32 | 225 ms | 22.5 ms | 10.0× |
| dense f64 + clip | 20k × 2k | float64 | 93.6 ms | 17.8 ms | 5.3× |

## How to reproduce

```bash
source .venv/bin/activate
maturin develop --release
python bench/bench_scale.py
```

## Why it is faster

- One pass over the input for `sum` and `sum(x^2)` (row-major, SIMD-friendly,
  parallel over fixed-size row blocks with a deterministic combine), vs. ~5
  separate passes in the reference.
- One fused pass for center + scale + clip.
- Zero-copy transfer of results into a NumPy array.

## Correctness contract

`|kernel(X) - reference(X)| <= atol + rtol * |reference(X)|`, with
`float64: rtol=1e-7, atol=1e-12` and `float32: rtol=1e-4, atol=1e-6`.
Verified by 55 tests (golden fixtures + hypothesis properties).
