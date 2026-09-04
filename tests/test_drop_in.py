"""Tests for the drop-in `install()` hook: patched Scanpy uses the kernels."""

from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

import scanpy_kernels as sk


def make_adata(dtype=np.float32, sparse=True):
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=np.exp(rng.uniform(-1, 2, 150)), size=(400, 150))
    counts[:, 0] = 0
    counts[:, 1] = 5
    x = np.log1p(counts).astype(dtype)
    return AnnData(sp.csr_matrix(x) if sparse else x)


def test_install_patches_and_uninstall_restores():
    adata = make_adata()

    # Reference results BEFORE patching.
    ref_hvg = sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False)
    adata_ref = adata.copy()
    sc.pp.scale(adata_ref, zero_center=False)

    sk.install()
    try:
        # Patched functions should now be the wrappers.
        assert sc.pp.scale.__module__.endswith("_patch")
        assert sc.pp.highly_variable_genes.__module__.endswith("_patch")

        got_hvg = sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False)
        np.testing.assert_array_equal(
            got_hvg["highly_variable"].to_numpy(), ref_hvg["highly_variable"].to_numpy()
        )
        np.testing.assert_allclose(
            got_hvg["dispersions_norm"].to_numpy(),
            ref_hvg["dispersions_norm"].to_numpy(),
            rtol=1e-4,
            atol=1e-6,
        )

        adata_got = adata.copy()
        sc.pp.scale(adata_got, zero_center=False)
        np.testing.assert_allclose(
            adata_got.X.toarray(), adata_ref.X.toarray(), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(adata_got.var["mean"], adata_ref.var["mean"])
        np.testing.assert_allclose(adata_got.var["std"], adata_ref.var["std"])
    finally:
        sk.uninstall()

    # Originals restored.
    assert sc.pp.scale is not None
    assert not sc.pp.scale.__module__.endswith("_patch")


def test_install_falls_back_for_other_flavors():
    adata = make_adata()
    ref = sc.pp.highly_variable_genes(adata, flavor="cell_ranger", inplace=False)
    sk.install()
    try:
        # cell_ranger is unsupported by the kernel -> delegated to original.
        got = sc.pp.highly_variable_genes(adata, flavor="cell_ranger", inplace=False)
        np.testing.assert_array_equal(
            got["highly_variable"].to_numpy(), ref["highly_variable"].to_numpy()
        )
    finally:
        sk.uninstall()
