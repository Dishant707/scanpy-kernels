//! `scanpy_kernels` — Rust-accelerated kernels for Scanpy.
//!
//! Correctness-first: every kernel must reproduce the reference Scanpy/NumPy
//! output within a strict tolerance. See `tests/` for the golden test suite.

mod core;
mod hvg;
mod scale;

use pyo3::prelude::*;

/// The native extension module (`scanpy_kernels._core`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scale::scale, m)?)?;
    m.add_function(wrap_pyfunction!(scale::scale_sparse_data, m)?)?;
    m.add_function(wrap_pyfunction!(hvg::hvg_seurat_stats, m)?)?;
    m.add_function(wrap_pyfunction!(hvg::hvg_seurat_stats_sparse, m)?)?;
    Ok(())
}
