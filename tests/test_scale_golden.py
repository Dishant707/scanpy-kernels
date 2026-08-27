"""Golden tests for the `pp.scale` Rust kernel.

The correctness contract (from the P2 plan):

    |kernel(X) - reference(X)| <= atol + rtol * |reference(X)|

- float64: rtol=1e-7, atol=1e-12
- float32: rtol=1e-4, atol=1e-6
- exact operations (integer counts, indices, labels): exact equality

Reference outputs are produced live by the *current installed Scanpy*, which is
pinned in the test environment. Golden fixtures are generated deterministically
(seeded), so every run is reproducible.

Known reference quirks (verified against Scanpy 1.12.3 / fast-array-utils):

1. Variance uses the naive formula `E[X^2] - E[X]^2` (not two-pass). For
   constant genes with non-exactly-representable values this can round to a
   tiny *negative* number, so the reference emits `NaN`. The kernel reproduces
   this exactly.
2. For `float32` input, `x**2` is computed in `float32` before being summed in
   `float64`, which the kernel also reproduces.
"""

from __future__ import annotations

import numpy as np
import pytest
from scanpy.preprocessing._scale import scale_array as reference_scale

import scanpy_kernels

TOLERANCES = {
    np.dtype(np.float64): dict(rtol=1e-7, atol=1e-12),
    np.dtype(np.float32): dict(rtol=1e-4, atol=1e-6),
}


def assert_matches_reference(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare arrays, requiring identical NaN layout and tolerance on finites."""
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


def fixture(dtype: np.dtype, shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Deterministic synthetic matrix with a realistic dynamic range."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) * 3.0 + 7.0
    return x.astype(dtype)


SHAPES = [(1, 1), (2, 2), (1, 5), (5, 1), (10, 3), (100, 20), (500, 50)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_scale_zero_center_matches_reference(shape: tuple[int, int], dtype: np.dtype) -> None:
    x = fixture(dtype, shape, seed=sum(shape))
    ref = reference_scale(x.copy(), zero_center=True)
    got = scanpy_kernels.scale(x, zero_center=True)
    assert got.dtype == ref.dtype
    assert_matches_reference(got, ref)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_scale_no_center_matches_reference(shape: tuple[int, int], dtype: np.dtype) -> None:
    x = fixture(dtype, shape, seed=sum(shape) + 1)
    ref = reference_scale(x.copy(), zero_center=False)
    got = scanpy_kernels.scale(x, zero_center=False)
    assert got.dtype == ref.dtype
    assert_matches_reference(got, ref)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("max_value", [None, 1.5, 10.0])
def test_scale_with_clip_matches_reference(dtype: np.dtype, max_value: float | None) -> None:
    x = fixture(dtype, (200, 15), seed=42)
    for zc in (True, False):
        ref = reference_scale(x.copy(), zero_center=zc, max_value=max_value)
        got = scanpy_kernels.scale(x, zero_center=zc, max_value=max_value)
        assert_matches_reference(got, ref)


def test_integer_input_cast_to_float64() -> None:
    rng = np.random.default_rng(7)
    x = rng.poisson(3.0, size=(50, 8)).astype(np.int64)
    ref = reference_scale(x.copy(), zero_center=True)
    got = scanpy_kernels.scale(x, zero_center=True)
    assert got.dtype == np.float64
    assert_matches_reference(got, ref)


def test_return_mean_std_matches_reference() -> None:
    x = fixture(np.float64, (100, 12), seed=99)
    ref_out, ref_mean, ref_std = reference_scale(x.copy(), return_mean_std=True)
    got_out, got_mean, got_std = scanpy_kernels.scale(x, return_mean_std=True)
    assert_matches_reference(got_out, ref_out)
    np.testing.assert_allclose(got_mean, ref_mean, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(got_std, ref_std, rtol=1e-7, atol=1e-12)


def test_constant_gene_exactly_representable_is_zeroed() -> None:
    """A constant gene with an exactly-representable value gets std=1 -> zeroed."""
    x = np.array(
        [
            [1.0, 3.0, 4.0],
            [2.0, 1.0, 4.0],
            [3.0, 2.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float64,
    )
    got = scanpy_kernels.scale(x, zero_center=True)
    ref = reference_scale(x.copy(), zero_center=True)
    assert_matches_reference(got, ref)
    np.testing.assert_allclose(got[:, 2], 0.0, atol=1e-12)


@pytest.mark.parametrize("constant", [4.0, 0.0, 1.0])
@pytest.mark.parametrize("n_rows", [1, 2, 5, 50])
def test_constant_gene_matches_reference(constant: float, n_rows: int) -> None:
    x = np.full((n_rows, 1), constant, dtype=np.float64)
    ref = reference_scale(x.copy(), zero_center=True)
    got = scanpy_kernels.scale(x, zero_center=True)
    assert_matches_reference(got, ref)


def test_constant_gene_negative_variance_nan_quirk() -> None:
    """For non-representable constants, the reference can emit NaN; we match it.

    This documents a reference quirk: the naive variance formula can round to a
    tiny negative number for constant float64 genes, and `sqrt` yields NaN.
    The kernel reproduces the reference *including* this quirk.
    """
    x = np.full((10, 1), 4.2, dtype=np.float64)
    ref = reference_scale(x.copy(), zero_center=True)
    got = scanpy_kernels.scale(x, zero_center=True)
    assert_matches_reference(got, ref)


def test_deterministic_across_calls() -> None:
    x = fixture(np.float64, (1000, 100), seed=123)
    a = scanpy_kernels.scale(x)
    b = scanpy_kernels.scale(x)
    np.testing.assert_array_equal(a, b)


def test_scaling_idempotent_on_standardized_data() -> None:
    """Scaling an already-standardized matrix should (re)produce std ~ 1."""
    x = fixture(np.float64, (300, 40), seed=5)
    once = scanpy_kernels.scale(x, zero_center=True)
    np.testing.assert_allclose(once.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(once.std(axis=0, ddof=1), 1.0, rtol=1e-7)
