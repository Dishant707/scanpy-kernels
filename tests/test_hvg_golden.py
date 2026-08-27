"""Golden tests for the `pp.highly_variable_genes` Rust kernel (Seurat flavor).

The heavy part (expm1 + per-gene mean/variance) is computed in Rust; binning,
normalization, and subsetting reuse the reference pandas logic, so the full
DataFrame output matches `scanpy.pp.highly_variable_genes(flavor="seurat")`.
"""

from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

import scanpy_kernels

TOLERANCES = {
    np.dtype(np.float64): dict(rtol=1e-7, atol=1e-12),
    np.dtype(np.float32): dict(rtol=1e-4, atol=1e-6),
}


def assert_matches_reference(actual: np.ndarray, expected: np.ndarray) -> None:
    tol = TOLERANCES[expected.dtype]
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    np.testing.assert_array_equal(actual_nan, expected_nan, err_msg="NaN layout differs")
    np.testing.assert_allclose(
        actual[~actual_nan],
        expected[~expected_nan],
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


def reference_hvg(X: np.ndarray, **kwargs) -> "pd.DataFrame":  # noqa: F821
    return sc.pp.highly_variable_genes(AnnData(X), flavor="seurat", inplace=False, **kwargs)


def make_log_counts(
    dtype: np.dtype, shape: tuple[int, int], seed: int = 0
) -> np.ndarray:
    n_obs, n_vars = shape
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=np.exp(rng.uniform(-1.0, 2.0, n_vars)), size=shape)
    if n_vars > 3:
        counts[:, 0] = 0  # all-zero gene
        counts[:, 1] = 5  # constant gene
        counts[:, 2] = rng.poisson(50, n_obs)  # high-variance gene
    return np.log1p(counts).astype(dtype)


SHAPES = [(1, 1), (2, 3), (1, 8), (20, 4), (50, 10), (400, 30), (2000, 120)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_hvg_seurat_matches_reference(shape: tuple[int, int], dtype: np.dtype) -> None:
    x = make_log_counts(dtype, shape, seed=sum(shape))
    ref = reference_hvg(x)
    got = scanpy_kernels.hvg_seurat(x, n_bins=20)
    assert list(got.columns) == ["highly_variable", "means", "dispersions", "dispersions_norm"]
    assert_matches_reference(got["means"].to_numpy(), ref["means"].to_numpy())
    assert_matches_reference(got["dispersions"].to_numpy(), ref["dispersions"].to_numpy())
    assert_matches_reference(
        got["dispersions_norm"].to_numpy(), ref["dispersions_norm"].to_numpy()
    )
    # The boolean mask matches exactly.
    np.testing.assert_array_equal(
        got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
    )


@pytest.mark.parametrize("n_bins", [5, 10, 20, 50])
def test_hvg_seurat_n_bins_variation(n_bins: int) -> None:
    x = make_log_counts(np.float64, (500, 80), seed=7)
    ref = reference_hvg(x, n_bins=n_bins)
    got = scanpy_kernels.hvg_seurat(x, n_bins=n_bins)
    np.testing.assert_array_equal(
        got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
    )
    assert_matches_reference(got["dispersions_norm"].to_numpy(), ref["dispersions_norm"].to_numpy())


def test_hvg_seurat_n_top_genes() -> None:
    x = make_log_counts(np.float64, (1000, 200), seed=11)
    for k in (10, 100, 500):
        ref = reference_hvg(x, n_top_genes=k)
        got = scanpy_kernels.hvg_seurat(x, n_top_genes=k)
        np.testing.assert_array_equal(
            got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
        )


def test_hvg_seurat_cutoff_bounds() -> None:
    x = make_log_counts(np.float64, (1000, 200), seed=13)
    ref = reference_hvg(x, min_disp=1.0, max_disp=10.0, min_mean=0.1, max_mean=2.0)
    got = scanpy_kernels.hvg_seurat(x, min_disp=1.0, max_disp=10.0, min_mean=0.1, max_mean=2.0)
    np.testing.assert_array_equal(
        got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
    )


def test_hvg_seurat_log_base() -> None:
    counts = np.random.default_rng(5).poisson(3, size=(300, 40))
    x = np.log1p(counts) / np.log(2.0)  # log2(1 + counts)
    adata = AnnData(x)
    adata.uns["log1p"] = {"base": 2.0}
    ref = sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False)
    got = scanpy_kernels.hvg_seurat(x, log_base=2.0)
    assert_matches_reference(got["means"].to_numpy(), ref["means"].to_numpy())
    assert_matches_reference(got["dispersions"].to_numpy(), ref["dispersions"].to_numpy())
    np.testing.assert_array_equal(
        got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
    )
