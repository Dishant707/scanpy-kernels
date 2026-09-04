//! `pp.scale` kernel — per-gene standardization.
//!
//! Mirrors `scanpy.preprocessing._scale.scale_array` for dense NumPy arrays.
//! See the P2 plan and `tests/test_scale_golden.py` for the correctness contract.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use rayon::prelude::*;

use crate::core;

/// Standardize each gene (column) to zero mean and unit variance.
///
/// `x` must be a 2-D NumPy array of dtype `float32`, `float64`, or an integer
/// type (integers are cast to `float64`, matching Scanpy).
///
/// - `zero_center`: subtract the per-gene mean before scaling.
/// - `max_value`: clip (truncate) scaled values to `[-max_value, max_value]`;
///   the lower bound is only applied when `zero_center` is true.
/// - `return_mean_std`: also return the per-gene mean and std arrays.
#[pyfunction]
#[pyo3(signature = (x, *, zero_center=true, max_value=None, return_mean_std=false))]
pub fn scale<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    zero_center: bool,
    max_value: Option<f64>,
    return_mean_std: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // float64 input
    if let Ok(a) = x.extract::<PyReadonlyArray2<f64>>() {
        let view = a.as_array();
        let standard = view.as_standard_layout();
        let data = standard
            .as_slice()
            .ok_or_else(|| PyTypeError::new_err("array is not contiguous"))?;
        let (n_rows, n_cols) = view.dim();
        return scale_f64(py, data, n_rows, n_cols, zero_center, max_value, return_mean_std);
    }

    // float32 input (output stays float32, like Scanpy)
    if let Ok(a) = x.extract::<PyReadonlyArray2<f32>>() {
        let view = a.as_array();
        let standard = view.as_standard_layout();
        let data = standard
            .as_slice()
            .ok_or_else(|| PyTypeError::new_err("array is not contiguous"))?;
        let (n_rows, n_cols) = view.dim();
        return scale_f32(py, data, n_rows, n_cols, zero_center, max_value, return_mean_std);
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
    scale_f64(py, data, n_rows, n_cols, zero_center, max_value, return_mean_std)
}

fn scale_f64<'py>(
    py: Python<'py>,
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    zero_center: bool,
    max_value: Option<f64>,
    return_mean_std: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (sums, sumsq) = core::column_sum_and_sq(data, n_cols);
    let (mean, std) = core::column_mean_std(&sums, &sumsq, n_rows);

    let mut out = vec![0.0f64; n_rows * n_cols];
    apply_chunks_f64(&mut out, data, &mean, &std, n_cols, zero_center, max_value);

    let out_arr = Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape mismatch")
        .into_pyarray(py);
    return_result(py, out_arr, mean, std, return_mean_std)
}

fn scale_f32<'py>(
    py: Python<'py>,
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    zero_center: bool,
    max_value: Option<f64>,
    return_mean_std: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // Squaring happens in f32 (matching NumPy `x**2`), accumulation in f64.
    let (sums, sumsq) = core::column_sum_and_sq_f32(data, n_cols);
    let (mean, std) = core::column_mean_std(&sums, &sumsq, n_rows);

    let mut out = vec![0.0f32; n_rows * n_cols];
    apply_chunks_f32(&mut out, data, &mean, &std, n_cols, zero_center, max_value);

    let out_arr = Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape mismatch")
        .into_pyarray(py);
    return_result(py, out_arr, mean, std, return_mean_std)
}

/// Rows processed per parallel task in the elementwise pass.
const APPLY_CHUNK_ROWS: usize = 512;

/// Center, scale, and clip a `f64` matrix (row-major, parallel over row blocks).
fn apply_chunks_f64(
    out: &mut [f64],
    data: &[f64],
    mean: &[f64],
    std: &[f64],
    n_cols: usize,
    zero_center: bool,
    max_value: Option<f64>,
) {
    out.par_chunks_mut(APPLY_CHUNK_ROWS * n_cols)
        .enumerate()
        .for_each(|(c, block)| {
            let row_start = c * APPLY_CHUNK_ROWS;
            let n_block_rows = block.len() / n_cols;
            for r in 0..n_block_rows {
                let i = row_start + r;
                let row = &data[i * n_cols..(i + 1) * n_cols];
                let orow = &mut block[r * n_cols..(r + 1) * n_cols];
                for j in 0..n_cols {
                    let v = row[j];
                    let mut z = if zero_center { v - mean[j] } else { v };
                    z /= std[j];
                    if let Some(mx) = max_value {
                        if z > mx {
                            z = mx;
                        } else if z < -mx && zero_center {
                            z = -mx;
                        }
                    }
                    orow[j] = z;
                }
            }
        });
}

/// Center, scale, and clip a `f32` matrix (row-major, parallel over row blocks).
fn apply_chunks_f32(
    out: &mut [f32],
    data: &[f32],
    mean: &[f64],
    std: &[f64],
    n_cols: usize,
    zero_center: bool,
    max_value: Option<f64>,
) {
    out.par_chunks_mut(APPLY_CHUNK_ROWS * n_cols)
        .enumerate()
        .for_each(|(c, block)| {
            let row_start = c * APPLY_CHUNK_ROWS;
            let n_block_rows = block.len() / n_cols;
            for r in 0..n_block_rows {
                let i = row_start + r;
                let row = &data[i * n_cols..(i + 1) * n_cols];
                let orow = &mut block[r * n_cols..(r + 1) * n_cols];
                for j in 0..n_cols {
                    let v = row[j] as f64;
                    let mut z = if zero_center { v - mean[j] } else { v };
                    z /= std[j];
                    let zf = z as f32;
                    let val = match max_value {
                        Some(mx) => {
                            let z64 = zf as f64;
                            if z64 > mx {
                                mx as f32
                            } else if z64 < -mx && zero_center {
                                (-mx) as f32
                            } else {
                                zf
                            }
                        }
                        None => zf,
                    };
                    orow[j] = val;
                }
            }
        });
}

/// Build the return value: the scaled array, or `(scaled, mean, std)`.
fn return_result<'py, T>(
    py: Python<'py>,
    out_arr: Bound<'py, PyArray2<T>>,
    mean: Vec<f64>,
    std: Vec<f64>,
    return_mean_std: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + Copy,
{
    if return_mean_std {
        let mean_arr: Bound<'py, PyArray1<f64>> = Array1::from_vec(mean).into_pyarray(py);
        let std_arr: Bound<'py, PyArray1<f64>> = Array1::from_vec(std).into_pyarray(py);
        let o = out_arr.into_any();
        let m = mean_arr.into_any();
        let s = std_arr.into_any();
        let tup = PyTuple::new(py, [o, m, s])?;
        Ok(tup.into_any())
    } else {
        Ok(out_arr.into_any())
    }
}

/// Scale a sparse (CSR) matrix's non-zero values and return mean/std.
///
/// `indices[k]` is the column index of `data[k]`; `n_rows`/`n_cols` are the
/// matrix dimensions. Matches `scanpy.pp.scale`'s sparse path (sklearn-style
/// mean/variance, upper-bound clipping only). `promote` reproduces the
/// reference's CSC behaviour: CSC output is promoted to `float64`, CSR keeps
/// its dtype.
#[pyfunction]
#[pyo3(signature = (indices, data, n_rows, n_cols, *, max_value=None, promote=false))]
pub fn scale_sparse_data<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    n_rows: usize,
    n_cols: usize,
    max_value: Option<f64>,
    promote: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let idx = indices.extract::<PyReadonlyArray1<i64>>()?;
    let idx = idx.as_slice()?;

    if let Ok(d) = data.extract::<PyReadonlyArray1<f64>>() {
        let dslice = d.as_slice()?;
        let (sums, sumsq) = core::sparse_column_sum_and_sq(idx, dslice, n_cols);
        let (mean, std) = core::column_mean_std(&sums, &sumsq, n_rows);
        let out: Vec<f64> = dslice
            .par_iter()
            .enumerate()
            .map(|(k, &v)| {
                let c = idx[k] as usize;
                let mut z = v / std[c];
                if let Some(mx) = max_value {
                    if z > mx {
                        z = mx;
                    }
                }
                z
            })
            .collect();
        return scale_sparse_result(py, out, mean, std);
    }

    if let Ok(d) = data.extract::<PyReadonlyArray1<f32>>() {
        let dslice = d.as_slice()?;
        let (sums, sumsq) =
            core::sparse_column_sum_and_sq_f32(idx, dslice, n_cols, promote);
        let (mean, std) = core::column_mean_std(&sums, &sumsq, n_rows);

        if promote {
            let out: Vec<f64> = dslice
                .par_iter()
                .enumerate()
                .map(|(k, &v)| {
                    let c = idx[k] as usize;
                    let mut z = v as f64 / std[c];
                    if let Some(mx) = max_value {
                        if z > mx {
                            z = mx;
                        }
                    }
                    z
                })
                .collect();
            return scale_sparse_result(py, out, mean, std);
        }

        let out: Vec<f32> = dslice
            .par_iter()
            .enumerate()
            .map(|(k, &v)| {
                let c = idx[k] as usize;
                let mut z = v as f64 / std[c];
                if let Some(mx) = max_value {
                    if z > mx {
                        z = mx;
                    }
                }
                z as f32
            })
            .collect();
        return scale_sparse_result(py, out, mean, std);
    }

    Err(PyTypeError::new_err(
        "sparse data must be float32 or float64",
    ))
}

fn scale_sparse_result<'py, T>(
    py: Python<'py>,
    data: Vec<T>,
    mean: Vec<f64>,
    std: Vec<f64>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + Copy,
{
    let d: Bound<'py, PyArray1<T>> = Array1::from_vec(data).into_pyarray(py);
    let m: Bound<'py, PyArray1<f64>> = Array1::from_vec(mean).into_pyarray(py);
    let s: Bound<'py, PyArray1<f64>> = Array1::from_vec(std).into_pyarray(py);
    let tup = PyTuple::new(py, [d.into_any(), m.into_any(), s.into_any()])?;
    Ok(tup.into_any())
}
