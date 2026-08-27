//! Shared core utilities: deterministic reductions.
//!
//! Design rules (from the P2 plan):
//! - Parallelize **across** rows in fixed-size chunks; each chunk sums its
//!   columns sequentially. The chunk partitioning is fixed, so the result is
//!   bit-for-bit deterministic regardless of the thread count.
//! - Traverse memory **row-major** (cache/SIMD friendly).

use rayon::prelude::*;

/// Rows processed per parallel chunk. Fixed so results are deterministic.
const CHUNK_ROWS: usize = 2048;

fn combine(partials: Vec<(Vec<f64>, Vec<f64>)>, n_cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut sums = vec![0.0f64; n_cols];
    let mut sq = vec![0.0f64; n_cols];
    for (ps, pq) in partials {
        for j in 0..n_cols {
            sums[j] += ps[j];
            sq[j] += pq[j];
        }
    }
    (sums, sq)
}

/// Per-column sum and sum-of-squares for a row-major `f64` slice.
pub fn column_sum_and_sq(x: &[f64], n_cols: usize) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = x
        .par_chunks(CHUNK_ROWS * n_cols)
        .map(|chunk| {
            let rows = chunk.len() / n_cols;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for i in 0..rows {
                let row = &chunk[i * n_cols..(i + 1) * n_cols];
                for j in 0..n_cols {
                    let v = row[j];
                    sums[j] += v;
                    sq[j] += v * v;
                }
            }
            (sums, sq)
        })
        .collect();
    combine(partials, n_cols)
}

/// Per-column sum and sum-of-squares for a row-major `f32` slice.
///
/// Mirrors NumPy's `float32` behavior in `fast_array_utils.stats.mean_var`:
/// the **squaring is done in `float32`** (`x**2` keeps the dtype), and only the
/// accumulation is `float64`. Keeping this quirk reproduces the reference
/// output exactly.
pub fn column_sum_and_sq_f32(x: &[f32], n_cols: usize) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = x
        .par_chunks(CHUNK_ROWS * n_cols)
        .map(|chunk| {
            let rows = chunk.len() / n_cols;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for i in 0..rows {
                let row = &chunk[i * n_cols..(i + 1) * n_cols];
                for j in 0..n_cols {
                    let v = row[j];
                    sums[j] += v as f64;
                    let vf = v * v; // f32 multiply -> rounds to f32
                    sq[j] += vf as f64;
                }
            }
            (sums, sq)
        })
        .collect();
    combine(partials, n_cols)
}

/// Compute per-column mean and standard deviation using the **exact recipe**
/// used by `fast_array_utils.stats.mean_var(..., correction=1)`:
///
/// ```text
/// mean_j = sum_j / n
/// var_j  = (sumsq_j / n - mean_j^2) * n / (n - 1)     (n > 1)
/// std_j  = sqrt(var_j), with std_j = 1 where var_j == 0
/// ```
///
/// This is the *naive* `E[X^2] - E[X]^2` formula (not the two-pass formula),
/// intentionally kept identical to the reference implementation so outputs
/// match within floating-point tolerance.
pub fn column_mean_std(sums: &[f64], sumsq: &[f64], n_rows: usize) -> (Vec<f64>, Vec<f64>) {
    let n = n_rows as f64;
    let mut mean = vec![0.0f64; sums.len()];
    let mut std = vec![1.0f64; sums.len()];

    for j in 0..sums.len() {
        let m = sums[j] / n;
        let mut var = sumsq[j] / n - m * m;
        if n_rows != 1 {
            var *= n / (n - 1.0);
        }
        let s = var.sqrt();
        std[j] = if s == 0.0 { 1.0 } else { s };
        mean[j] = m;
    }
    (mean, std)
}
