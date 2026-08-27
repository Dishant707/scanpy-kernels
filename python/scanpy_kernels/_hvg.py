"""Seurat-flavor highly variable genes, with the heavy work in Rust.

The per-gene `means`/`dispersions` come from the native kernel
(`scanpy_kernels._core.hvg_seurat_stats`). The binning, normalization, and
subsetting are O(n_vars) and reuse the exact pandas logic of Scanpy so the
output matches `scanpy.pp.highly_variable_genes(flavor="seurat")` bit-for-bit
on the categorical parts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import hvg_seurat_stats


def hvg_seurat(
    x: np.ndarray,
    *,
    n_bins: int = 20,
    n_top_genes: int | None = None,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    log_base: float | None = None,
) -> pd.DataFrame:
    """Highly variable genes (Seurat flavor) for a dense log1p expression matrix.

    Parameters mirror ``scanpy.pp.highly_variable_genes`` for ``flavor="seurat"``
    with a single batch. Returns a DataFrame with columns ``highly_variable``,
    ``means``, ``dispersions``, and ``dispersions_norm``.
    """
    means_arr, dispersions_arr = hvg_seurat_stats(x, log_base=log_base)
    means = pd.Series(means_arr)
    dispersions = pd.Series(dispersions_arr)

    df = pd.DataFrame({"means": means, "dispersions": dispersions})

    # --- binning (identical to scanpy._get_mean_bins) ---
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    df["mean_bin"] = df["mean_bin"].cat.set_categories(
        df["mean_bin"].cat.categories.astype("string"), rename=True
    )

    # --- per-bin dispersion stats (identical to scanpy._get_disp_stats) ---
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    disp_bin_stats = disp_grouped.agg(avg="mean", dev="std")

    # Single-gene bins: normalized dispersion is set to 1.
    one_gene_per_bin = disp_bin_stats["dev"].isnull()
    disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[
        one_gene_per_bin, "avg"
    ]
    disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0

    disp_stats = disp_bin_stats.loc[df["mean_bin"]].set_index(df.index)
    df["dispersions_norm"] = (
        df["dispersions"] - disp_stats["avg"]
    ) / disp_stats["dev"]

    # --- subsetting (identical to scanpy._subset_genes) ---
    if n_top_genes is None:
        disp_norm = np.nan_to_num(df["dispersions_norm"].to_numpy())
        highly_variable = (
            (means.to_numpy() > min_mean)
            & (means.to_numpy() < max_mean)
            & (disp_norm > min_disp)
            & (disp_norm < max_disp)
        )
    else:
        disp_norm_arr = df["dispersions_norm"].to_numpy()
        x = disp_norm_arr[~np.isnan(disp_norm_arr)]
        n = min(n_top_genes, x.size)
        x[::-1].sort()  # sorts x in descending order
        disp_cut_off = x[n - 1]
        highly_variable = np.nan_to_num(disp_norm_arr, nan=-np.inf) >= disp_cut_off

    df["highly_variable"] = highly_variable
    return df[["highly_variable", "means", "dispersions", "dispersions_norm"]]
