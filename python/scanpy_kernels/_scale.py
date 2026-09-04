"""`pp.scale` entry point, dispatching dense and sparse inputs.

The dense path is fully in Rust; the sparse path computes per-gene mean/std in
Rust and scales the non-zero values in NumPy, preserving Scanpy's exact CSR/CSC
dtype behaviour (CSC gets promoted to float64, CSR keeps its dtype).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._core import scale as _scale_dense
from ._core import scale_sparse_data as _scale_sparse_data


def scale(x, *, zero_center=True, max_value=None, return_mean_std=False):
    """Standardize each gene (column) to zero mean and unit variance.

    Accepts a dense NumPy array (`float32`/`float64`/integer) or a scipy sparse
    CSR/CSC matrix, mirroring `scanpy.pp.scale` / `scale_array`.
    """
    if sp.issparse(x):
        if zero_center:
            # The reference densifies sparse input when zero-centering, and the
            # `x -= mean` subtraction promotes the result to float64.
            dense = x.toarray()
            if dense.dtype != np.float64:
                dense = dense.astype(np.float64)
            return _scale_dense(
                dense, zero_center=True, max_value=max_value, return_mean_std=return_mean_std
            )
        return _scale_sparse(x, max_value=max_value, return_mean_std=return_mean_std)

    return _scale_dense(
        x, zero_center=zero_center, max_value=max_value, return_mean_std=return_mean_std
    )


def _scale_sparse(x, *, max_value, return_mean_std):
    fmt = x.format
    csr = x.tocsr()
    if not np.issubdtype(csr.data.dtype, np.floating):
        csr = csr.astype(np.float64)

    new_data, mean, std = _scale_sparse_data(
        csr.indices.astype(np.int64, copy=False),
        csr.data,
        x.shape[0],
        x.shape[1],
        max_value=max_value,
        promote=(fmt == "csc"),
    )

    result = sp.csr_matrix((new_data, csr.indices, csr.indptr), shape=x.shape)
    if fmt == "csc":
        result = result.tocsc()

    if return_mean_std:
        return result, mean, std
    return result
