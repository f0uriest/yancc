"""Tests for linalg stuff."""

import cola
import numpy as np
import pytest

import yancc.linalg


def test_block_operator():
    """Tests for BlockOperator."""
    rng = np.random.default_rng(123)
    A11 = rng.random((4, 5))
    A12 = rng.random((4, 5))
    A13 = rng.random((4, 5))
    A21 = rng.random((4, 5))
    A22 = rng.random((4, 5))
    A23 = rng.random((4, 5))

    A = yancc.linalg.BlockOperator([[A11, A12, A13], [A21, A22, A23]])
    B = np.block([[A11, A12, A13], [A21, A22, A23]])
    np.testing.assert_allclose(A.to_dense(), B)

    with pytest.raises(AssertionError):
        _ = yancc.linalg.BlockOperator([[A11, A12, A13], [A21, A22]])

    with pytest.raises(AssertionError):
        _ = yancc.linalg.BlockOperator([[A11.flatten(), A12, A13], [A21, A22, A23]])


def test_bordered_operator():
    """Test for BorderedOperator."""
    rng = np.random.default_rng(123)
    A = rng.random((5, 5))
    B = rng.random((5, 2))
    C = rng.random((2, 5))
    D = rng.random((2, 2))

    F = yancc.linalg.BorderedOperator(A, B, C, D)
    G = np.block([[A, B], [C, D]])

    np.testing.assert_allclose(F.to_dense(), G)
    np.testing.assert_allclose(cola.linalg.inv(F).to_dense(), np.linalg.inv(G))

    with pytest.raises(AssertionError):
        _ = yancc.linalg.BorderedOperator(A, B, C, A)


def test_pinv():
    """Test for pseudoinverse."""
    A = cola.ops.Diagonal(np.arange(5).astype(float))
    Ai = cola.linalg.pinv(A, yancc.linalg.FullRank()).to_dense()
    assert np.linalg.matrix_rank(Ai) == 5
    np.testing.assert_allclose(A @ Ai, np.diag([0, 1, 1, 1, 1]))
    np.testing.assert_allclose(Ai @ A, np.diag([0, 1, 1, 1, 1]))

    rng = np.random.default_rng(123)

    A = rng.random((5, 5))
    A[0] = 0
    A = cola.ops.Dense(A)
    Ai = cola.linalg.pinv(A).to_dense()
    assert np.linalg.matrix_rank(Ai) == 4
    np.testing.assert_allclose(A @ Ai, np.diag([0, 1, 1, 1, 1]), atol=1e-14)

    Ai = cola.linalg.pinv(A, yancc.linalg.FullRank()).to_dense()
    assert np.linalg.matrix_rank(Ai) == 5
    np.testing.assert_allclose(A @ Ai, np.diag([0, 1, 1, 1, 1]), atol=1e-14)

    I = cola.ops.I_like(A)
    Ii = cola.linalg.pinv(I)
    assert Ii is I

    A = rng.random((5, 5))
    B = rng.random((4, 4))
    A[0] = 0
    C = cola.ops.Kronecker(A, B)
    Ci = cola.linalg.pinv(C)
    np.testing.assert_allclose(Ci.to_dense(), np.linalg.pinv(C.to_dense()), atol=1e-12)


def test_dot():
    """Tests for dot."""
    rng = np.random.default_rng(123)
    diag = cola.ops.Diagonal(rng.random(5))
    dense = cola.ops.Dense(rng.random((5, 5)))
    I = cola.ops.I_like(dense)

    np.testing.assert_allclose(
        (diag @ dense).to_dense(), diag.to_dense() @ dense.to_dense()
    )
    np.testing.assert_allclose(
        (dense @ diag).to_dense(), dense.to_dense() @ diag.to_dense()
    )
    np.testing.assert_allclose(
        (diag @ diag).to_dense(), diag.to_dense() @ diag.to_dense()
    )
    assert (dense @ I) is dense
    assert (I @ dense) is dense
    assert (I @ I) is I


def test_add():
    """Tests for add."""
    rng = np.random.default_rng(123)
    diag = cola.ops.Diagonal(rng.random(5))
    dense = cola.ops.Dense(rng.random((5, 5)))
    I = cola.ops.I_like(dense)

    np.testing.assert_allclose(
        (diag + dense).to_dense(), diag.to_dense() + dense.to_dense()
    )
    np.testing.assert_allclose(
        (dense + diag).to_dense(), dense.to_dense() + diag.to_dense()
    )
    np.testing.assert_allclose(
        (diag + diag).to_dense(), diag.to_dense() + diag.to_dense()
    )
    np.testing.assert_allclose((diag + I).to_dense(), diag.to_dense() + I.to_dense())
    np.testing.assert_allclose((I + diag).to_dense(), I.to_dense() + diag.to_dense())
    np.testing.assert_allclose((dense + I).to_dense(), dense.to_dense() + I.to_dense())
    np.testing.assert_allclose((I + dense).to_dense(), I.to_dense() + dense.to_dense())
    np.testing.assert_allclose(
        (dense + dense).to_dense(), dense.to_dense() + dense.to_dense()
    )
    np.testing.assert_allclose(
        (dense + diag).to_dense(), dense.to_dense() + diag.to_dense()
    )
    np.testing.assert_allclose(
        (diag + dense).to_dense(), diag.to_dense() + dense.to_dense()
    )
    assert isinstance(dense + dense, cola.ops.Dense)
    assert isinstance(diag + diag, cola.ops.Diagonal)
    assert isinstance(I + diag, cola.ops.Diagonal)
    assert isinstance(diag + I, cola.ops.Diagonal)
    assert isinstance(I + dense, cola.ops.Dense)
    assert isinstance(dense + I, cola.ops.Dense)
    assert isinstance(diag + dense, cola.ops.Dense)
    assert isinstance(dense + diag, cola.ops.Dense)


def test_inv_sum_kron():
    """Tests for inv_sum_kron."""
    rng = np.random.default_rng(123)
    diag = cola.ops.Diagonal(rng.random(5))
    dense1 = cola.ops.Dense(rng.random((5, 5)))
    dense2 = cola.ops.Dense(rng.random((5, 5)))
    dense3 = cola.ops.Dense(rng.random((5, 5)))
    A = cola.ops.Kronecker(diag, dense1)
    B = cola.ops.Kronecker(dense2, dense3)

    C = yancc.linalg.inv_sum_kron(A, B).to_dense()
    np.testing.assert_allclose(C, np.linalg.inv((A + B).to_dense()))


def test_approx_sum_kron():
    """Tests for approx_sum_kron."""
    rng = np.random.default_rng(123)
    diag1 = cola.ops.Diagonal(rng.random(5))
    dense1 = cola.ops.Dense(rng.random((5, 5)))
    diag2 = cola.ops.Diagonal(rng.random(5))
    dense2 = cola.ops.Dense(rng.random((5, 5)))
    A = cola.ops.Kronecker(diag1, dense1)
    B = cola.ops.Kronecker(diag2, dense2)
    C = A + B
    C1 = yancc.linalg.approx_sum_kron(A, B)
    assert isinstance(C1, cola.ops.Kronecker)
    assert np.linalg.norm((C - C1).to_dense()) / np.linalg.norm(C.to_dense()) < 0.25
