"""End-to-end validation on the real PBMC 3k dataset.

Proves that the kernels reproduce reference Scanpy on a published single-cell
dataset and leave downstream results (clusters) unchanged.

Usage:
    python bench/validate_pbmc3k.py
"""

from __future__ import annotations

import numpy as np
import scanpy as sc

import scanpy_kernels as sk


def main() -> None:
    print("Loading PBMC 3k ...")
    adata = sc.datasets.pbmc3k()  # 2700 cells x 32738 genes, sparse float32

    # Standard Scanpy preprocessing (from the Scanpy PBMC3k tutorial).
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"adata.X: {adata.X.shape} {adata.X.dtype} {type(adata.X).__name__}")

    # --- highly variable genes (Seurat flavor) ---
    ref_hvg = sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False)
    got_hvg = sk.hvg_seurat(adata.X, n_bins=20)
    hv_match = np.array_equal(
        ref_hvg["highly_variable"].to_numpy(), got_hvg["highly_variable"].to_numpy()
    )
    means_diff = np.nanmax(
        np.abs(ref_hvg["means"].to_numpy() - got_hvg["means"].to_numpy())
    )
    print(
        f"HVG: selection identical = {hv_match} "
        f"(n={int(got_hvg['highly_variable'].sum())}), means max diff = {means_diff:.2e}"
    )
    assert hv_match

    # --- scale (sparse, zero_center=False, as done before PCA) ---
    adata_ref = adata.copy()
    sc.pp.scale(adata_ref, zero_center=False)
    got_scaled = sk.scale(adata.X, zero_center=False)
    scale_diff = np.max(np.abs(adata_ref.X.data - got_scaled.data))
    print(f"scale: max |data| diff = {scale_diff:.2e} (dtype {got_scaled.dtype})")
    assert scale_diff < 1e-5

    # --- downstream: PCA + Leiden clustering identical ---
    def cluster_pipeline(X):
        a = sc.AnnData(X)
        sc.tl.pca(a, n_comps=30, svd_solver="arpack")
        sc.pp.neighbors(a, n_neighbors=10, n_pcs=30)
        sc.tl.leiden(a, random_state=0)
        return a.obs["leiden"].to_numpy()

    labels_ref = cluster_pipeline(adata_ref.X)
    labels_kernel = cluster_pipeline(got_scaled)
    same = np.array_equal(labels_ref, labels_kernel)
    print(
        f"Leiden clusters: identical = {same} "
        f"(n_clusters={len(np.unique(labels_kernel))})"
    )
    assert same

    print("\nPASS: kernels reproduce reference Scanpy on PBMC 3k, downstream unchanged.")


if __name__ == "__main__":
    main()
