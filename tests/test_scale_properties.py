"""Property-based tests for the `pp.scale` kernel.

These complement the fixed golden fixtures: instead of hand-picked shapes, we
randomly generate matrices and assert two things for every input:

1. The kernel matches the reference Scanpy output within tolerance.
2. The output is actually standardized (zero mean, unit variance per gene,
   except for zero-variance genes).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scanpy.preprocessing._scale import scale_array as reference_scale

import scanpy_kernels

TOLERANCES = {
    np.dtype(np.float64): dict(rtol=1e-7, atol=1e-12),
    np.dtype(np.float32): dict(rtol=1e-4, atol=1e-6),
}

# Keep the search space bounded so the suite stays fast.
shape = st.tuples(st.integers(1, 50), st.integers(1, 20))


@st.composite
def float_matrix(draw, dtype: np.dtype) -> np.ndarray:
    n_rows, n_cols = draw(shape)
    # Wide dynamic range to exercise cancellation in the variance formula.
    values = draw(
        st.lists(
            st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=n_rows * n_cols,
            max_size=n_rows * n_cols,
        )
    )
    return np.array(values, dtype=dtype).reshape(n_rows, n_cols)


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


@settings(max_examples=40, deadline=None)
@given(float_matrix(np.float64))
def test_matches_reference_float64(x: np.ndarray) -> None:
    ref = reference_scale(x.copy(), zero_center=True)
    got = scanpy_kernels.scale(x, zero_center=True)
    assert_matches_reference(got, ref)


@settings(max_examples=40, deadline=None)
@given(float_matrix(np.float32))
def test_matches_reference_float32(x: np.ndarray) -> None:
    ref = reference_scale(x.copy(), zero_center=True)
    got = scanpy_kernels.scale(x, zero_center=True)
    assert_matches_reference(got, ref)


@settings(max_examples=40, deadline=None)
@given(float_matrix(np.float64))
def test_output_is_standardized(x: np.ndarray) -> None:
    got = scanpy_kernels.scale(x, zero_center=True)
    input_std = np.std(x, axis=0, ddof=1)
    for j in range(got.shape[1]):
        col = got[:, j]
        if not np.all(np.isfinite(col)):
            # Column hit the reference NaN quirk (negative variance); skip.
            continue
        if input_std[j] == 0 or np.isnan(input_std[j]):
            # Constant gene is zeroed out.
            np.testing.assert_allclose(col, 0.0, atol=1e-8)
        else:
            np.testing.assert_allclose(col.mean(), 0.0, atol=1e-6)
            np.testing.assert_allclose(col.std(ddof=1), 1.0, atol=1e-6)
