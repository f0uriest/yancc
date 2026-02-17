"""Tests for linalg stuff."""

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy

import yancc.linalg


def _random_banded(p, q, n, rng, periodic=True):
    A = np.diag(p + q + rng.random(n))  # make it diagonally dominant
    for i in range(1, p + 1):
        A += np.diag(rng.random(n - i), k=-i)
    for i in range(1, q + 1):
        A += np.diag(rng.random(n - i), k=i)

    if periodic:
        if q > 0:
            A[-q:, 0] = q
        if p > 0:
            A[0, -p:] = p
    return A


def test_bordered_operator():
    """Test for BorderedOperator."""
    n, k = 5, 2

    rng = np.random.default_rng(123)
    A = rng.random((n, n))
    e, v = np.linalg.eig(A)
    e[:k] = 0
    A = (v @ np.diag(e) @ np.linalg.inv(v)).real
    assert np.linalg.matrix_rank(A) == n - k
    B = scipy.linalg.null_space(A)
    C = scipy.linalg.null_space(A.T).T
    assert B.shape == (n, k)
    assert C.shape == (k, n)
    D = np.zeros((k, k))
    Ab = np.block([[A, B], [C, D]])

    F = yancc.linalg.BorderedOperator(
        lx.MatrixLinearOperator(jnp.array(A)),
        lx.MatrixLinearOperator(jnp.array(B)),
        lx.MatrixLinearOperator(jnp.array(C)),
    )

    np.testing.assert_allclose(F.as_matrix(), Ab, atol=1e-14)
    np.testing.assert_allclose(F.T.as_matrix(), Ab.T, atol=1e-14)

    Fi = yancc.linalg.InverseBorderedOperator(
        lx.MatrixLinearOperator(jnp.linalg.pinv(A)),
        lx.MatrixLinearOperator(jnp.array(B)),
        lx.MatrixLinearOperator(jnp.array(C)),
    )
    np.testing.assert_allclose(Fi.as_matrix(), np.linalg.inv(Ab), atol=1e-14)
    np.testing.assert_allclose(Fi.T.as_matrix(), np.linalg.inv(Ab.T), atol=1e-14)


def test_tridiagonal():
    """Test for solving tridiagonal systems."""
    N = 140
    rng = np.random.default_rng(123)

    l = rng.random(N - 1)
    d = rng.random(N)
    u = rng.random(N - 1)
    b = rng.random(N)
    A = yancc.linalg.make_dense_tridiag(l, d, u)
    B = yancc.linalg.make_dense_tridiag(l, d, u, jnp.array(1.2), jnp.array(3.2))

    np.testing.assert_allclose(
        np.linalg.solve(A, b), yancc.linalg.tridiag_solve_dense(A, b)
    )
    np.testing.assert_allclose(
        np.linalg.solve(B, b), yancc.linalg.tridiag_solve_dense(B, b)
    )


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 5, 10])
@pytest.mark.parametrize("periodic", [True, False])
def test_banded_to_dense(p, q, n, periodic):
    """Test conversion between banded and dense formats."""
    rng = np.random.default_rng(123)

    A = _random_banded(p, q, n, rng, periodic)

    C = yancc.linalg.dense_to_banded(p, q, A)
    B = yancc.linalg.banded_to_dense(p, q, C)

    np.testing.assert_allclose(A, B)

    if not periodic:
        b = rng.random(n)
        np.testing.assert_allclose(
            scipy.linalg.solve_banded((p, q), C, b), np.linalg.solve(B, b)
        )


def test_lu_factor_banded():
    """Test that LU is the same without pivoting for diagonally dominance matrix."""
    rng = np.random.default_rng(123)
    A = _random_banded(2, 2, 10, rng, False)
    B = yancc.linalg.dense_to_banded(2, 2, A)
    lu1 = scipy.linalg.lu_factor(A)[0]
    lu2 = yancc.linalg.lu_factor_banded(2, 2, B)
    np.testing.assert_allclose(lu1, yancc.linalg.banded_to_dense(2, 2, lu2))


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 5, 10])  # need n > max(p,q)
def test_solve_banded(p, q, n):
    """Test solving regular banded system."""
    rng = np.random.default_rng(123)
    A = _random_banded(p, q, n, rng, False)
    a = yancc.linalg.dense_to_banded(p, q, A)
    b = rng.random(n)
    x = yancc.linalg.solve_banded(p, q, a, b)
    np.testing.assert_allclose(yancc.linalg.banded_mv(p, q, a, x), b)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 5, 10])  # need n > max(p,q)
@pytest.mark.parametrize("periodic", [True, False])
def test_solve_banded_periodic(p, q, n, periodic):
    """Test solving periodic banded system."""
    rng = np.random.default_rng(123)
    A = _random_banded(p, q, n, rng, periodic)
    a = yancc.linalg.dense_to_banded(p, q, A)
    b = rng.random(n)

    x = yancc.linalg.solve_banded_periodic(p, q, a, b)
    np.testing.assert_allclose(A @ x, b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b))


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 5, 10])  # need n > max(p,q)
@pytest.mark.parametrize("periodic", [True, False])
def test_banded_mv(p, q, n, periodic):
    """Test for banded matrix vector multiply."""
    rng = np.random.default_rng(123)
    A = _random_banded(p, q, n, rng, periodic)
    a = yancc.linalg.dense_to_banded(p, q, A)
    x = rng.random(n)
    np.testing.assert_allclose(A @ x, yancc.linalg.banded_mv(p, q, a, x))


@pytest.mark.parametrize("p", [(0, 0), (0, 1), (2, 2)])
@pytest.mark.parametrize("q", [(0, 0), (1, 0), (1, 1)])
@pytest.mark.parametrize("n", [2, 4, 10])  # need n > max(p,q)
@pytest.mark.parametrize("periodic", [True, False])
def test_banded_mm(p, q, n, periodic):
    """Test for banded matrix matrix multiply."""
    rng = np.random.default_rng(123)
    p1, p2 = p
    q1, q2 = q
    A = _random_banded(p1, q1, n, rng, periodic)
    B = _random_banded(p2, q2, n, rng, periodic)
    a = yancc.linalg.dense_to_banded(p1, q1, A)
    b = yancc.linalg.dense_to_banded(p2, q2, B)
    c, pc, qc = yancc.linalg.banded_mm(p1, q1, p2, q2, a, b)
    C = yancc.linalg.banded_to_dense(int(pc), int(qc), c)
    np.testing.assert_allclose(A @ B, C)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 5, 10])  # need n > max(p,q)
@pytest.mark.parametrize("periodic", [True, False])
def test_banded_transpose(p, q, n, periodic):
    """Test for banded matrix transpose."""
    rng = np.random.default_rng(123)
    A = _random_banded(p, q, n, rng, periodic)
    a = yancc.linalg.dense_to_banded(p, q, A)
    at, pt, qt = yancc.linalg.banded_transpose(p, q, a)
    np.testing.assert_allclose(A.T, yancc.linalg.banded_to_dense(int(pt), int(qt), at))
