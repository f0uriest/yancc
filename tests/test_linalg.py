"""Tests for linalg stuff."""

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy

import yancc.linalg


def test_bordered_operator():
    """Test for BorderedOperator."""
    rng = np.random.default_rng(123)
    A = jnp.array(rng.random((5, 5)))
    B = jnp.array(rng.random((5, 2)))
    C = jnp.array(rng.random((2, 5)))
    D = jnp.array(rng.random((2, 2)))

    F = yancc.linalg.BorderedOperator(
        lx.MatrixLinearOperator(A),
        lx.MatrixLinearOperator(B),
        lx.MatrixLinearOperator(C),
        lx.MatrixLinearOperator(D),
    )
    G = jnp.block([[A, B], [C, D]])

    np.testing.assert_allclose(F.as_matrix(), G)
    np.testing.assert_allclose(F.T.as_matrix(), G.T)

    Fi = yancc.linalg.InverseBorderedOperator(
        lx.MatrixLinearOperator(jnp.linalg.inv(A)),
        lx.MatrixLinearOperator(B),
        lx.MatrixLinearOperator(C),
        lx.MatrixLinearOperator(D),
    )
    np.testing.assert_allclose(Fi.as_matrix(), np.linalg.inv(G))
    np.testing.assert_allclose(Fi.T.as_matrix(), np.linalg.inv(G.T))


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
