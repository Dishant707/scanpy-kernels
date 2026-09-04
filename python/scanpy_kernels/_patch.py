"""Optional drop-in integration: make Scanpy use the Rust kernels.

After ``scanpy_kernels.install()``, ``scanpy.pp.scale`` and
``scanpy.pp.highly_variable_genes`` are transparently accelerated for the
supported cases (dense or CSR/CSC data, Seurat flavor, single batch). Anything
unsupported (masked/backed/dask data, other flavors, batch-aware HVG, subset)
falls back to the original Scanpy implementation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._hvg import hvg_seurat
from ._scale import scale

_ORIG_SCALE = None
_ORIG_HVG = None
_patched = False


def _get_x(adata, layer, obsm):
    if layer is not None:
        return adata.layers[layer]
    if obsm is not None:
        return adata.obsm[obsm]
    return adata.X


def _supported(x) -> bool:
    return isinstance(x, np.ndarray) or sp.issparse(x)


def _patched_scale(
    adata,
    *,
    zero_center=True,
    max_value=None,
    copy=False,
    layer=None,
    obsm=None,
    mask_obs=None,
):
    if mask_obs is not None or adata.is_view or not _supported(_get_x(adata, layer, obsm)):
        return _ORIG_SCALE(
            adata,
            zero_center=zero_center,
            max_value=max_value,
            copy=copy,
            layer=layer,
            obsm=obsm,
            mask_obs=mask_obs,
        )

    adata = adata.copy() if copy else adata
    x = _get_x(adata, layer, obsm)
    scaled, mean, std = scale(
        x, zero_center=zero_center, max_value=max_value, return_mean_std=True
    )
    if layer is not None:
        adata.layers[layer] = scaled
    elif obsm is not None:
        adata.obsm[obsm] = scaled
    else:
        adata.X = scaled
    adata.var["mean"] = mean
    adata.var["std"] = std
    return adata if copy else None


def _patched_hvg(
    adata,
    *,
    layer=None,
    n_top_genes=None,
    min_disp=0.5,
    max_disp=np.inf,
    min_mean=0.0125,
    max_mean=3.0,
    span=0.3,
    n_bins=20,
    flavor="seurat",
    subset=False,
    inplace=True,
    batch_key=None,
    filter_unexpressed_genes=None,
    check_values=True,
):
    if (
        flavor != "seurat"
        or batch_key is not None
        or filter_unexpressed_genes
        or subset
        or adata.is_view
        or not _supported(_get_x(adata, layer, None))
    ):
        return _ORIG_HVG(
            adata,
            layer=layer,
            n_top_genes=n_top_genes,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
            span=span,
            n_bins=n_bins,
            flavor=flavor,
            subset=subset,
            inplace=inplace,
            batch_key=batch_key,
            filter_unexpressed_genes=filter_unexpressed_genes,
            check_values=check_values,
        )

    log_base = adata.uns.get("log1p", {}).get("base")
    df = hvg_seurat(
        _get_x(adata, layer, None),
        n_bins=n_bins,
        n_top_genes=n_top_genes,
        min_disp=min_disp,
        max_disp=max_disp,
        min_mean=min_mean,
        max_mean=max_mean,
        log_base=log_base,
    )
    if not inplace:
        return df

    adata.uns["hvg"] = {"flavor": flavor}
    adata.var["highly_variable"] = df["highly_variable"]
    adata.var["means"] = df["means"]
    adata.var["dispersions"] = df["dispersions"]
    adata.var["dispersions_norm"] = df["dispersions_norm"].astype(np.float32)
    return None


def install() -> None:
    """Monkey-patch Scanpy to use the Rust kernels for supported inputs."""
    global _ORIG_SCALE, _ORIG_HVG, _patched
    if _patched:
        return
    import scanpy

    _ORIG_SCALE = scanpy.pp.scale
    _ORIG_HVG = scanpy.pp.highly_variable_genes
    scanpy.pp.scale = _patched_scale
    scanpy.pp.highly_variable_genes = _patched_hvg
    _patched = True


def uninstall() -> None:
    """Restore the original Scanpy functions."""
    global _patched
    if not _patched:
        return
    import scanpy

    scanpy.pp.scale = _ORIG_SCALE
    scanpy.pp.highly_variable_genes = _ORIG_HVG
    _patched = False
