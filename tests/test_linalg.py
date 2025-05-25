"""Tests for linalg stuff."""

import cola
import numpy as np
import pytest
import scipy

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


def test_full_rank():
    """Test for making a singular matrix full rank."""
    rng = np.random.default_rng(123)
    A = rng.random((5, 5))
    Af = yancc.linalg.full_rank(A).to_dense()
    np.testing.assert_allclose(A, Af)

    A[0] = 0
    assert np.linalg.matrix_rank(A) == 4
    u, s, v = np.linalg.svd(A)
    Af = yancc.linalg.full_rank(A).to_dense()
    assert np.linalg.matrix_rank(Af) == 5
    np.testing.assert_allclose(np.linalg.norm(A - Af), s[-2])


def test_inv_sum_kron():
    """Tests for inv_sum_kron."""
    rng = np.random.default_rng(123)
    diag = cola.ops.Diagonal(rng.random(5))
    dense1 = cola.ops.Dense(rng.random((5, 5)))
    dense2 = cola.ops.Dense(rng.random((5, 5)))
    dense3 = cola.ops.Dense(rng.random((5, 5)))
    A = cola.ops.Kronecker(diag, dense1)
    B = cola.ops.Kronecker(dense2, dense3)

    C = yancc.linalg.inv_sum_kron(A, B, yancc.linalg.FullRank()).to_dense()
    np.testing.assert_allclose(C, np.linalg.inv((A + B).to_dense()))
    C = yancc.linalg.inv_sum_kron(A, B, yancc.linalg.LSTSQ()).to_dense()
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
    D = cola.ops.Kronecker(dense1, diag1)
    E = cola.ops.Kronecker(dense2, diag2)
    C = A + B
    w1, f1 = yancc.linalg.approx_sum_kron_weights((A, B))
    assert w1.shape == (2, 2)
    w2, f2 = yancc.linalg.approx_sum_kron_weights((A, B), (A, B, D, E))
    assert w2.shape == (2, 4)
    C1 = yancc.linalg.approx_sum_kron((A, B))
    C2 = yancc.linalg.approx_sum_kron((A, B), (A, B, D, E))
    assert isinstance(C1, cola.ops.Kronecker)
    assert isinstance(C2, cola.ops.Kronecker)
    C1_err = np.linalg.norm((C - C1).to_dense()) / np.linalg.norm(C.to_dense())
    C2_err = np.linalg.norm((C - C2).to_dense()) / np.linalg.norm(C.to_dense())
    np.testing.assert_allclose(f1, C1_err)
    np.testing.assert_allclose(f2, C2_err)
    assert C1_err < 0.25
    assert C2_err < C1_err


def test_approx_sum_kron2():
    """Tests for approx_sum_kron."""
    rng = np.random.default_rng(123)
    diag1 = cola.ops.Diagonal(rng.random(5))
    dense1 = cola.ops.Dense(rng.random((5, 5)))
    diag2 = cola.ops.Diagonal(rng.random(5))
    dense2 = cola.ops.Dense(rng.random((5, 5)))
    diag3 = cola.ops.Diagonal(rng.random(5))
    dense3 = cola.ops.Dense(rng.random((5, 5)))
    diag4 = cola.ops.Diagonal(rng.random(5))
    dense4 = cola.ops.Dense(rng.random((5, 5)))
    A1 = cola.ops.Kronecker(diag1, dense1)
    A2 = cola.ops.Kronecker(diag2, dense2)
    A3 = cola.ops.Kronecker(diag3, dense3)
    A4 = cola.ops.Kronecker(diag4, dense4)
    A = cola.ops.Sum(A1, A2, A3, A4)
    # using 2 options for B, 2 options for C
    a1, b1, f1 = yancc.linalg.approx_sum_kron_weights2(A.Ms, A.Ms[2:])
    assert a1.shape == (2, 2)
    assert b1.shape == (2, 2)
    # using 3 options for B, 3 options for C
    a2, b2, f2 = yancc.linalg.approx_sum_kron_weights2(A.Ms, A.Ms[1:], A.Ms[:-1])
    assert a2.shape == (2, 3)
    assert b2.shape == (2, 3)
    # using 4 options for B, 4 options for C
    a3, b3, f3 = yancc.linalg.approx_sum_kron_weights2(A.Ms)
    assert a3.shape == (2, 4)
    assert b3.shape == (2, 4)
    C1, D1 = yancc.linalg.approx_sum_kron2(A.Ms, A.Ms[2:])
    C2, D2 = yancc.linalg.approx_sum_kron2(A.Ms, A.Ms[1:], A.Ms[:-1])
    C3, D3 = yancc.linalg.approx_sum_kron2(A.Ms)
    assert isinstance(C1, cola.ops.Kronecker)
    assert isinstance(C2, cola.ops.Kronecker)
    assert isinstance(C3, cola.ops.Kronecker)
    assert isinstance(D1, cola.ops.Kronecker)
    assert isinstance(D2, cola.ops.Kronecker)
    assert isinstance(D3, cola.ops.Kronecker)
    C1_err = np.linalg.norm((A - C1 - D1).to_dense()) / np.linalg.norm(A.to_dense())
    C2_err = np.linalg.norm((A - C2 - D2).to_dense()) / np.linalg.norm(A.to_dense())
    C3_err = np.linalg.norm((A - C3 - D3).to_dense()) / np.linalg.norm(A.to_dense())
    np.testing.assert_allclose(f1, C1_err)
    np.testing.assert_allclose(f2, C2_err)
    np.testing.assert_allclose(f3, C3_err)
    assert C1_err < 0.5
    assert C2_err < C1_err
    assert C3_err < C2_err


def test_real_imag():
    """Test getting real and imaginary parts of an operator."""
    rng = np.random.default_rng(123)
    A = rng.random((5, 5)) + 1j * rng.random((5, 5))
    x = rng.random(5)
    B = cola.fns.lazify(A)
    Br = yancc.linalg.RealOperator(B)
    Bi = yancc.linalg.ImagOperator(B)

    np.testing.assert_allclose(A.real, Br.to_dense())
    np.testing.assert_allclose(A.imag, Bi.to_dense())

    np.testing.assert_allclose(Br @ x, A.real @ x)
    np.testing.assert_allclose(Br @ (1j + x), A.real @ (1j + x))

    np.testing.assert_allclose(Bi @ x, (A.imag @ x))
    np.testing.assert_allclose(Bi @ (1j + x), (A.imag @ (1j + x)))


def test_tridiagonal():
    """Test for solving tridiagonal systems."""
    N = 140
    rng = np.random.default_rng(123)

    l = rng.random(N - 1)
    d = rng.random(N)
    u = rng.random(N - 1)
    b = rng.random(N)
    A = yancc.linalg.make_dense_tridiag(l, d, u)
    B = yancc.linalg.make_dense_tridiag(l, d, u, 1.2, 3.2)

    np.testing.assert_allclose(
        np.linalg.solve(A, b), yancc.linalg.tridiag_solve_dense(A, b)
    )
    np.testing.assert_allclose(
        np.linalg.solve(B, b), yancc.linalg.tridiag_solve_dense(B, b)
    )


@pytest.mark.parametrize(
    "nr", [(n, r) for n in [1, 4, 8, 16] for r in np.arange(5) if r <= n]
)
def test_banded_to_dense(nr):
    """Test conversion between banded and dense formats."""
    n, r = nr
    p = r
    q = r
    rng = np.random.default_rng(123)

    A = np.diag(np.random.random(n))
    for i in range(1, p + 1):
        A += np.diag(np.random.random(n - i), k=-i)
    for i in range(1, q + 1):
        A += np.diag(np.random.random(n - i), k=i)

    b = rng.random(n)

    C = yancc.linalg.dense_to_banded(p, q, A)
    B = yancc.linalg.banded_to_dense(p, q, C)

    np.testing.assert_allclose(A, B)

    if n > 2 * r:
        np.testing.assert_allclose(
            yancc.linalg.banded_to_dense(
                p, q, yancc.linalg.dense_to_banded(p, q, A, True), True
            ),
            A,
        )
    np.testing.assert_allclose(
        yancc.linalg.banded_to_dense(p, q, yancc.linalg.dense_to_banded(p, q, B)), B
    )
    np.testing.assert_allclose(
        scipy.linalg.solve_banded((p, q), C, b), np.linalg.solve(B, b)
    )


def test_lu_factor_banded():
    """Test that LU is the same without pivoting for diagonally dominance matrix."""
    rng = np.random.default_rng(123)

    A = (
        np.diag(rng.random(8), k=-2)
        + np.diag(rng.random(8), k=2)
        + np.diag(rng.random(9), k=-1)
        + np.diag(rng.random(9), k=1)
        + np.diag(3 + rng.random(10), k=0)
    )
    B = yancc.linalg.dense_to_banded(2, 2, A)
    lu1 = scipy.linalg.lu_factor(A)[0]
    lu2 = yancc.linalg.lu_factor_banded(2, 2, B)
    np.testing.assert_allclose(lu1, yancc.linalg.banded_to_dense(2, 2, lu2))


@pytest.mark.parametrize(
    "nr", [(n, r) for n in [1, 4, 8, 16] for r in np.arange(5) if r <= n]
)
def test_solve_banded(nr):
    """Test solving regular banded system."""
    n, r = nr
    p = r
    q = r
    rng = np.random.default_rng(123)

    A = 0.5 - rng.random((p + q + 1, n))
    b = rng.random(n)

    np.testing.assert_allclose(
        scipy.linalg.solve_banded((p, q), A, b), yancc.linalg.solve_banded(p, q, A, b)
    )


@pytest.mark.parametrize(
    "nr", [(n, r) for n in [1, 4, 8, 16] for r in np.arange(5) if r <= n]
)
def test_solve_banded_periodic(nr):
    """Test solving periodic banded system."""
    n, r = nr
    rng = np.random.default_rng(123)
    A = np.diag(np.random.random(n))
    for i in range(1, r + 1):
        A += np.diag(np.random.random(n - i), k=-i)
    for i in range(1, r + 1):
        A += np.diag(np.random.random(n - i), k=i)
    if r > 0:
        F = np.triu(np.random.random((r, r)))
        G = np.tril(np.random.random((r, r)))
        A[:r, -r:] = F
        A[-r:, :r] = G
    b = rng.random(n)

    x = yancc.linalg.solve_banded_periodic(r, A, b)
    np.testing.assert_allclose(A @ x, b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b))
