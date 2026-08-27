//! `pp.highly_variable_genes` kernel — Seurat flavor, dispersion statistics.
//!
//! Mirrors the per-gene statistics of
//! `scanpy.preprocessing._highly_variable_genes._highly_variable_genes_single_batch`
//! for `flavor="seurat"` on dense arrays.
//!
//! The expensive part (O(n_obs * n_vars)) is the `expm1` of log-data followed
//! by per-gene mean/variance. The binning, normalization, and subsetting are
//! O(n_vars) and are done in the Python wrapper to match pandas exactly.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use rayon::prelude::*;

use crate::core::{self, CHUNK_ROWS};

/// Compute per-gene `means` and `dispersions` for the Seurat flavor.
///
/// `x` is the logarithmized (log1p) expression matrix, `n_obs` x `n_vars`,
/// float32 or float64. `log_base` is the base of the logarithm (`None` = natural
/// log, matching `scanpy.pp.log1p`'s default).
///
/// Returns `(means, dispersions)` as float64 arrays of length `n_vars`, where
/// `means = log1p(mean(expm1(x)))` and
/// `dispersions = log(var(expm1(x)) / mean(expm1(x)))` with the reference's
/// exact edge-case handling (`mean==0 -> 1e-12`, `dispersion==0 -> NaN`).
#[pyfunction]
#[pyo3(signature = (x, *, log_base=None))]
pub fn hvg_seurat_stats<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    log_base: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let scale = log_base.map(|b| b.ln());

    if let Ok(a) = x.extract::<PyReadonlyArray2<f64>>() {
        let view = a.as_array();
        let standard = view.as_standard_layout();
        let data = standard
            .as_slice()
            .ok_or_else(|| PyTypeError::new_err("array is not contiguous"))?;
        let (n_rows, n_cols) = view.dim();
        let (sums, sumsq) = hvg_sum_sq_f64(data, n_cols, scale);
        let (mean, var) = core::column_mean_var(&sums, &sumsq, n_rows);
        return finish_hvg(py, &mean, &var);
    }

    if let Ok(a) = x.extract::<PyReadonlyArray2<f32>>() {
        let view = a.as_array();
        let standard = view.as_standard_layout();
        let data = standard
            .as_slice()
            .ok_or_else(|| PyTypeError::new_err("array is not contiguous"))?;
        let (n_rows, n_cols) = view.dim();
        let (sums, sumsq) = hvg_sum_sq_f32(data, n_cols, scale);
        let (mean, var) = core::column_mean_var(&sums, &sumsq, n_rows);
        return finish_hvg(py, &mean, &var);
    }

    // Integer (or other numeric) input: cast to float64, matching Scanpy.
    let cast = x.call_method1("astype", ("float64",))?;
    let a = cast.extract::<PyReadonlyArray2<f64>>()?;
    let view = a.as_array();
    let standard = view.as_standard_layout();
    let data = standard
        .as_slice()
        .ok_or_else(|| PyTypeError::new_err("array is not contiguous"))?;
    let (n_rows, n_cols) = view.dim();
    let (sums, sumsq) = hvg_sum_sq_f64(data, n_cols, scale);
    let (mean, var) = core::column_mean_var(&sums, &sumsq, n_rows);
    finish_hvg(py, &mean, &var)
}

fn hvg_sum_sq_f64(x: &[f64], n_cols: usize, scale: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = x
        .par_chunks(CHUNK_ROWS * n_cols)
        .map(|chunk| {
            let rows = chunk.len() / n_cols;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for i in 0..rows {
                let row = &chunk[i * n_cols..(i + 1) * n_cols];
                for j in 0..n_cols {
                    let mut v = row[j];
                    if let Some(s) = scale {
                        v *= s;
                    }
                    let e = v.exp_m1();
                    sums[j] += e;
                    sq[j] += e * e;
                }
            }
            (sums, sq)
        })
        .collect();
    core::combine(partials, n_cols)
}

fn hvg_sum_sq_f32(x: &[f32], n_cols: usize, scale: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    let partials: Vec<(Vec<f64>, Vec<f64>)> = x
        .par_chunks(CHUNK_ROWS * n_cols)
        .map(|chunk| {
            let rows = chunk.len() / n_cols;
            let mut sums = vec![0.0f64; n_cols];
            let mut sq = vec![0.0f64; n_cols];
            for i in 0..rows {
                let row = &chunk[i * n_cols..(i + 1) * n_cols];
                for j in 0..n_cols {
                    // `x *= log(base)` happens in float32 (NumPy in-place mult),
                    // then `np.expm1` in float32, then `x**2` in float32.
                    let v = match scale {
                        Some(s) => (row[j] as f64 * s) as f32,
                        None => row[j],
                    };
                    let e = v.exp_m1(); // f32
                    sums[j] += e as f64;
                    let esq = e * e; // f32 multiply
                    sq[j] += esq as f64;
                }
            }
            (sums, sq)
        })
        .collect();
    core::combine(partials, n_cols)
}

fn finish_hvg<'py>(
    py: Python<'py>,
    mean: &[f64],
    var: &[f64],
) -> PyResult<Bound<'py, PyAny>> {
    let n = mean.len();
    let mut means = vec![0.0f64; n];
    let mut dispersions = vec![0.0f64; n];
    for j in 0..n {
        let m = if mean[j] == 0.0 { 1e-12 } else { mean[j] };
        let disp = var[j] / m;
        dispersions[j] = if disp == 0.0 { f64::NAN } else { disp.ln() };
        means[j] = m.ln_1p();
    }

    let means_arr: Bound<'_, PyArray1<f64>> = Array1::from_vec(means).into_pyarray(py);
    let disp_arr: Bound<'_, PyArray1<f64>> = Array1::from_vec(dispersions).into_pyarray(py);
    let m = means_arr.into_any();
    let d = disp_arr.into_any();
    let tup = PyTuple::new(py, [m, d])?;
    Ok(tup.into_any())
}
