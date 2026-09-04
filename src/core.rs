//! Shared core utilities: deterministic reductions.
//!
//! Design rules (from the P2 plan):
//! - Parallelize **across** rows in fixed-size chunks; each chunk sums its
//!   columns sequentially. The chunk partitioning is fixed, so the result is
//!   bit-for-bit deterministic regardless of the thread count.
//! - Traverse memory **row-major** (cache/SIMD friendly).

use rayon::prelude::*;

/// Rows processed per parallel chunk. Fixed so results are deterministic.
pub(crate) const CHUNK_ROWS: usize = 2048;

pub(crate) fn combine(partials: Vec<(Vec<f64>, Vec<f64>)>, n_cols: usize) -> (Vec<f64>, Vec<f64>) {
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

/// Compute per-column mean and variance using the **exact recipe** used by
/// `fast_array_utils.stats.mean_var(..., correction=1)`:
///
/// ```text
/// mean_j = sum_j / n
/// var_j  = (sumsq_j / n - mean_j^2) * n / (n - 1)     (n > 1)
/// ```
pub fn column_mean_var(sums: &[f64], sumsq: &[f64], n_rows: usize) -> (Vec<f64>, Vec<f64>) {
    let n = n_rows as f64;
    let mut mean = vec![0.0f64; sums.len()];
    let mut var = vec![0.0f64; sums.len()];
    for j in 0..sums.len() {
        let m = sums[j] / n;
        let mut v = sumsq[j] / n - m * m;
        if n_rows != 1 {
            v *= n / (n - 1.0);
        }
        mean[j] = m;
        var[j] = v;
    }
    (mean, var)
}

/// Compute per-column mean and standard deviation.
///
/// Identical to `column_mean_var`, then `std_j = sqrt(var_j)` with
/// `std_j = 1` where `var_j == 0` (matching Scanpy's `pp.scale`).
pub fn column_mean_std(sums: &[f64], sumsq: &[f64], n_rows: usize) -> (Vec<f64>, Vec<f64>) {
    let (mean, var) = column_mean_var(sums, sumsq, n_rows);
    let std = var
        .iter()
        .map(|&v| {
            let s = v.sqrt();
            if s == 0.0 { 1.0 } else { s }
        })
        .collect();
    (mean, std)
}

/// Non-zeros processed per parallel chunk. Fixed so results are deterministic.
pub(crate) const CHUNK_NNZ: usize = 1_000_000;

/// Per-column sum and sum-of-squares from sparse (CSR) `indices`/`data`.
///
/// `indices[k]` is the column of `data[k]`. Matches the sklearn-style sparse
/// mean/variance used by `fast_array_utils.stats._sparse_mean_var`; the
/// division and Bessel correction are applied later by `column_mean_var`.
pub fn sparse_column_sum_and_sq(
    indices: &[i64],
    data: &[f64],
    n_cols: usize,
) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = data
        .par_chunks(CHUNK_NNZ)
        .enumerate()
        .map(|(ci, chunk)| {
            let start = ci * CHUNK_NNZ;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for k in 0..chunk.len() {
                let c = indices[start + k] as usize;
                let v = chunk[k];
                sums[c] += v;
                sq[c] += v * v;
            }
            (sums, sq)
        })
        .collect();
    combine(partials, n_cols)
}

/// `float32` variant of [`sparse_column_sum_and_sq`].
///
/// `square_f64` reproduces a reference asymmetry: the CSR path squares in
/// `float32` (`x**2`), while the CSC path casts to `float64` before squaring.
pub fn sparse_column_sum_and_sq_f32(
    indices: &[i64],
    data: &[f32],
    n_cols: usize,
    square_f64: bool,
) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = data
        .par_chunks(CHUNK_NNZ)
        .enumerate()
        .map(|(ci, chunk)| {
            let start = ci * CHUNK_NNZ;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for k in 0..chunk.len() {
                let c = indices[start + k] as usize;
                let v = chunk[k];
                sums[c] += v as f64;
                let s = if square_f64 {
                    let vv = v as f64;
                    vv * vv
                } else {
                    (v * v) as f64
                };
                sq[c] += s;
            }
            (sums, sq)
        })
        .collect();
    combine(partials, n_cols)
}

/// Sparse per-column sum of `expm1(value)` and its square (Seurat HVG).
pub fn sparse_expm1_sum_and_sq(
    indices: &[i64],
    data: &[f64],
    n_cols: usize,
    scale: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = data
        .par_chunks(CHUNK_NNZ)
        .enumerate()
        .map(|(ci, chunk)| {
            let start = ci * CHUNK_NNZ;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for k in 0..chunk.len() {
                let c = indices[start + k] as usize;
                let mut v = chunk[k];
                if let Some(s) = scale {
                    v *= s;
                }
                let e = v.exp_m1();
                sums[c] += e;
                sq[c] += e * e;
            }
            (sums, sq)
        })
        .collect();
    combine(partials, n_cols)
}

/// `float32` variant of [`sparse_expm1_sum_and_sq`]: `expm1` in `f32`, squaring
/// in `f32` (`square_f64=false`, CSR) or `f64` (`square_f64=true`, CSC).
pub fn sparse_expm1_sum_and_sq_f32(
    indices: &[i64],
    data: &[f32],
    n_cols: usize,
    scale: Option<f64>,
    square_f64: bool,
) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = data
        .par_chunks(CHUNK_NNZ)
        .enumerate()
        .map(|(ci, chunk)| {
            let start = ci * CHUNK_NNZ;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for k in 0..chunk.len() {
                let c = indices[start + k] as usize;
                let v = match scale {
                    Some(s) => (chunk[k] as f64 * s) as f32,
                    None => chunk[k],
                };
                let e = v.exp_m1(); // f32
                sums[c] += e as f64;
                let s = if square_f64 {
                    let ee = e as f64;
                    ee * ee
                } else {
                    (e * e) as f64
                };
                sq[c] += s;
            }
            (sums, sq)
        })
        .collect();
    combine(partials, n_cols)
}
