//! `scanpy_kernels` — Rust-accelerated kernels for Scanpy.
//!
//! Correctness-first: every kernel must reproduce the reference Scanpy/NumPy
//! output within a strict tolerance. See `tests/` for the golden test suite.

mod core;
mod scale;

use pyo3::prelude::*;

/// The native extension module (`scanpy_kernels._core`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scale::scale, m)?)?;
    Ok(())
}
