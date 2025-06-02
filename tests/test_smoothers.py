"""Tests for constructing smoothing operators."""

import jax
import numpy as np

from yancc.finite_diff import fd_coeffs
from yancc.smoothers import get_block_diag, permute_f
from yancc.trajectories import dfdpitch, dfdtheta, dfdxi, dfdzeta, mdke


def extract_blocks(a, m):
    """Get diagonal blocks from a of size m."""
    return np.array(
        [a[i * m : (i + 1) * m, i * m : (i + 1) * m] for i in range(a.shape[0] // m)]
    )


def test_get_block_diag(pitchgrid, field):
    """Test that using jax tricks to get the block diagonal works correctly."""
    nt, nz = field.ntheta, field.nzeta
    nl = pitchgrid.nxi

    order = "tza"
    a1 = get_block_diag(field, pitchgrid, 1e-3, 1e-3, order)
    a2 = extract_blocks(
        jax.jacfwd(lambda x: mdke(x, field, pitchgrid, 1e-3, 1e-3, axorder=order))(
            np.zeros(nt * nz * nl)
        ),
        nl,
    )
    np.testing.assert_allclose(a1, a2, err_msg=order)

    order = "zat"
    a1 = get_block_diag(field, pitchgrid, 1e-3, 1e-3, order)
    a2 = extract_blocks(
        jax.jacfwd(lambda x: mdke(x, field, pitchgrid, 1e-3, 1e-3, axorder=order))(
            np.zeros(nt * nz * nl)
        ),
        nt,
    )
    np.testing.assert_allclose(a1, a2, err_msg=order)

    order = "atz"
    a1 = get_block_diag(field, pitchgrid, 1e-3, 1e-3, order)
    a2 = extract_blocks(
        jax.jacfwd(lambda x: mdke(x, field, pitchgrid, 1e-3, 1e-3, axorder=order))(
            np.zeros(nt * nz * nl)
        ),
        nz,
    )
    np.testing.assert_allclose(a1, a2, err_msg=order)


def test_diagonals(field, pitchgrid):
    """Test that tricks for getting the diagonal work correctly."""
    nt, nz = field.ntheta, field.nzeta
    nl = pitchgrid.nxi
    E_psi = 1e-4
    nu = 1e-4

    order = "tza"
    for p in fd_coeffs[1].keys():
        Ai = jax.jacfwd(lambda x: dfdxi(x, field, pitchgrid, nu, axorder=order, p=p))(
            np.zeros(nt * nz * nl)
        )
        np.testing.assert_allclose(
            dfdxi(
                np.ones(nt * nz * nl),
                field,
                pitchgrid,
                nu,
                axorder=order,
                diag=True,
                p=p,
            ),
            np.diag(Ai),
        )

    order = "tza"
    for p in [2, 4, 6]:
        Ai = jax.jacfwd(
            lambda x: dfdpitch(x, field, pitchgrid, nu, axorder=order, p=p)
        )(np.zeros(nt * nz * nl))
        np.testing.assert_allclose(
            dfdpitch(
                np.ones(nt * nz * nl),
                field,
                pitchgrid,
                1e-4,
                axorder=order,
                diag=True,
                p=p,
            ),
            np.diag(Ai),
        )

    order = "atz"
    for p in fd_coeffs[1].keys():
        Az = jax.jacfwd(
            lambda x: dfdzeta(x, field, pitchgrid, E_psi, axorder=order, p=p)
        )(np.zeros(nt * nz * nl))
        np.testing.assert_allclose(
            dfdzeta(
                np.ones(nt * nz * nl),
                field,
                pitchgrid,
                E_psi,
                axorder=order,
                diag=True,
                p=p,
            ),
            np.diag(Az),
        )

    order = "zat"
    for p in fd_coeffs[1].keys():
        At = jax.jacfwd(
            lambda x: dfdtheta(x, field, pitchgrid, E_psi, axorder=order, p=p)
        )(np.zeros(nt * nz * nl))
        np.testing.assert_allclose(
            dfdtheta(
                np.ones(nt * nz * nl),
                field,
                pitchgrid,
                E_psi,
                axorder=order,
                diag=True,
                p=p,
            ),
            np.diag(At),
        )


def test_M_matrix(field, pitchgrid):
    """Test that a first/second order upwind discretization gives and M matrix."""
    E_psi = 1e-4
    nu = 1e-4
    N = field.ntheta * field.nzeta * pitchgrid.nxi

    Dt = jax.jacfwd(dfdtheta)(
        np.zeros(N), field, pitchgrid, E_psi, "atz", p="1a", gauge=True
    )
    Dz = jax.jacfwd(dfdzeta)(
        np.zeros(N), field, pitchgrid, E_psi, "atz", p="1a", gauge=True
    )
    Dx = jax.jacfwd(dfdxi)(np.zeros(N), field, pitchgrid, nu, "atz", p="1a", gauge=True)
    Dp = jax.jacfwd(dfdpitch)(np.zeros(N), field, pitchgrid, nu, "atz", p=2, gauge=True)
    A = Dt + Dz + Dx + Dp

    assert (np.diag(Dt) >= 0).all()
    assert (np.diag(Dz) >= 0).all()
    assert (np.diag(Dx) >= 0).all()
    assert (np.diag(Dp) >= 0).all()
    assert (np.diag(A) >= 0).all()

    assert ((Dt - np.diag(np.diag(Dt))) <= 0).all()
    assert ((Dz - np.diag(np.diag(Dz))) <= 0).all()
    assert ((Dx - np.diag(np.diag(Dx))) <= 0).all()
    assert ((Dp - np.diag(np.diag(Dp))) <= 0).all()
    assert ((A - np.diag(np.diag(A))) <= 0).all()

    assert (np.linalg.inv(A) >= 0).all()


def test_permutations(field, pitchgrid):
    """Test that re-ordering the grid points gives equivalent operators."""
    p1 = "2a"
    p2 = 2
    E_psi = 1e-4
    nu = 1e-4
    N = field.ntheta * field.nzeta * pitchgrid.nxi

    A0f = jax.jacfwd(mdke)(
        np.zeros(N),
        field,
        pitchgrid,
        E_psi,
        nu,
        "atz",
        p1=p1,
        p2=p2,
        gauge=True,
    )
    A1f = jax.jacfwd(mdke)(
        np.zeros(N),
        field,
        pitchgrid,
        E_psi,
        nu,
        "zat",
        p1=p1,
        p2=p2,
        gauge=True,
    )
    A2f = jax.jacfwd(mdke)(
        np.zeros(N),
        field,
        pitchgrid,
        E_psi,
        nu,
        "tza",
        p1=p1,
        p2=p2,
        gauge=True,
    )

    P0f = jax.jacfwd(permute_f)(np.zeros(N), field, pitchgrid, "atz")
    P1f = jax.jacfwd(permute_f)(np.zeros(N), field, pitchgrid, "zat")
    P2f = jax.jacfwd(permute_f)(np.zeros(N), field, pitchgrid, "tza")

    # dummy check that Ps are permutation matrices
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P0f @ P0f.T)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P1f @ P1f.T)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P2f @ P2f.T)

    np.testing.assert_allclose(np.eye(P0f.shape[0]), P0f.T @ P0f)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P1f.T @ P1f)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P2f.T @ P2f)

    # applying permutation matrices should be the same as the operator in
    # re-ordered basis
    np.testing.assert_allclose(A0f, P0f @ A0f @ P0f.T)
    np.testing.assert_allclose(A0f, P1f @ A1f @ P1f.T)
    np.testing.assert_allclose(A0f, P2f @ A2f @ P2f.T)
