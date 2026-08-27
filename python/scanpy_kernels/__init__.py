"""Rust-accelerated, numerically-verified kernels for Scanpy.

The native implementation lives in :mod:`scanpy_kernels._core` (built with
maturin). This package keeps the public API tiny and mirrors Scanpy's own
calling conventions.
"""

from ._core import hvg_seurat_stats, scale
from ._hvg import hvg_seurat

__all__ = ["scale", "hvg_seurat", "hvg_seurat_stats"]
__version__ = "0.1.0"
