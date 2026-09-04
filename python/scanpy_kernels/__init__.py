"""Rust-accelerated, numerically-verified kernels for Scanpy.

The native implementation lives in :mod:`scanpy_kernels._core` (built with
maturin). This package keeps the public API tiny and mirrors Scanpy's own
calling conventions.
"""

from ._core import hvg_seurat_stats, hvg_seurat_stats_sparse
from ._hvg import hvg_seurat
from ._patch import install, uninstall
from ._scale import scale

__all__ = [
    "scale",
    "hvg_seurat",
    "hvg_seurat_stats",
    "hvg_seurat_stats_sparse",
    "install",
    "uninstall",
]
__version__ = "0.1.0"
