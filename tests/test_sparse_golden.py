"""Golden tests for sparse (CSR/CSC) `pp.scale` and `pp.highly_variable_genes`.

The kernels must reproduce the reference exactly, including Scanpy's CSR/CSC
dtype quirks (CSC `scale` promotes data to float64; CSR keeps its dtype).
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scanpy.preprocessing._scale import scale_array as reference_scale

import scanpy_kernels

TOLERANCES = {
    np.dtype(np.float64): dict(rtol=1e-7, atol=1e-12),
    np.dtype(np.float32): dict(rtol=1e-4, atol=1e-6),
}


def assert_matches_reference(actual: np.ndarray, expected: np.ndarray, dtype) -> None:
    tol = TOLERANCES[np.dtype(dtype)]
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    np.testing.assert_array_equal(actual_nan, expected_nan, err_msg="NaN layout differs")
    np.testing.assert_allclose(
        actual[~actual_nan],
        expected[~expected_nan],
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


def sparse_fixture(dtype: np.dtype, shape: tuple[int, int], seed: int = 0, density: float = 0.3):
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal(shape) * 2.0 + 5.0
    dense[dense < 0] = 0
    dense[rng.random(shape) > density] = 0
    return dense.astype(dtype)


SHAPES = [(2, 3), (1, 8), (50, 10), (300, 40)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_scale_sparse_no_center(shape, dtype, fmt):
    dense = sparse_fixture(dtype, shape, seed=sum(shape))
    m = (sp.csr_matrix if fmt == "csr" else sp.csc_matrix)(dense)
    ref = reference_scale(m.copy(), zero_center=False)
    got = scanpy_kernels.scale(m, zero_center=False)
    assert got.format == ref.format
    assert got.dtype == ref.dtype
    assert_matches_reference(got.toarray(), ref.toarray(), dtype)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("max_value", [None, 1.5, 10.0])
def test_scale_sparse_with_clip(fmt, max_value):
    dense = sparse_fixture(np.float32, (200, 30), seed=11)
    m = (sp.csr_matrix if fmt == "csr" else sp.csc_matrix)(dense)
    ref = reference_scale(m.copy(), zero_center=False, max_value=max_value)
    got = scanpy_kernels.scale(m, zero_center=False, max_value=max_value)
    assert got.dtype == ref.dtype
    assert_matches_reference(got.toarray(), ref.toarray(), np.float32)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scale_sparse_zero_center_densifies(fmt, dtype):
    dense = sparse_fixture(dtype, (100, 25), seed=5)
    m = (sp.csr_matrix if fmt == "csr" else sp.csc_matrix)(dense)
    ref = reference_scale(m.copy(), zero_center=True)
    got = scanpy_kernels.scale(m, zero_center=True)
    assert got.dtype == ref.dtype  # float64 (reference promotes on densify)
    assert_matches_reference(got, ref, dtype)


def test_scale_sparse_return_mean_std():
    dense = sparse_fixture(np.float64, (150, 20), seed=9)
    m = sp.csr_matrix(dense)
    ref_out, ref_mean, ref_std = reference_scale(m.copy(), zero_center=False, return_mean_std=True)
    got_out, got_mean, got_std = scanpy_kernels.scale(m, zero_center=False, return_mean_std=True)
    assert_matches_reference(got_out.toarray(), ref_out.toarray(), np.float64)
    np.testing.assert_allclose(got_mean, ref_mean, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(got_std, ref_std, rtol=1e-7, atol=1e-12)


def test_scale_sparse_integer_input():
    rng = np.random.default_rng(4)
    dense = rng.poisson(3, size=(80, 15))
    dense[rng.random(dense.shape) > 0.4] = 0
    m = sp.csr_matrix(dense.astype(np.int64))
    ref = reference_scale(m.copy(), zero_center=False)
    got = scanpy_kernels.scale(m, zero_center=False)
    assert got.dtype == ref.dtype
    assert_matches_reference(got.toarray(), ref.toarray(), np.float64)


def make_log_counts(dtype: np.dtype, shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    n_obs, n_vars = shape
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=np.exp(rng.uniform(-1.0, 2.0, n_vars)), size=shape)
    if n_vars > 3:
        counts[:, 0] = 0
        counts[:, 1] = 5
        counts[:, 2] = rng.poisson(50, n_obs)
    return np.log1p(counts).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_hvg_sparse_matches_reference(dtype, fmt):
    logc = make_log_counts(dtype, (600, 90), seed=3)
    m = (sp.csr_matrix if fmt == "csr" else sp.csc_matrix)(logc)
    ref = sc.pp.highly_variable_genes(AnnData(m), flavor="seurat", inplace=False)
    got = scanpy_kernels.hvg_seurat(m, n_bins=20)
    assert_matches_reference(got["means"].to_numpy(), ref["means"].to_numpy(), dtype)
    assert_matches_reference(got["dispersions"].to_numpy(), ref["dispersions"].to_numpy(), dtype)
    np.testing.assert_array_equal(
        got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
    )


def test_hvg_sparse_n_top_genes():
    logc = make_log_counts(np.float64, (800, 120), seed=8)
    m = sp.csr_matrix(logc)
    for k in (10, 50):
        ref = sc.pp.highly_variable_genes(AnnData(m), flavor="seurat", n_top_genes=k, inplace=False)
        got = scanpy_kernels.hvg_seurat(m, n_top_genes=k)
        np.testing.assert_array_equal(
            got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
        )
