"""Tests for constructing smoothing operators."""

import itertools

import jax
import numpy as np

from yancc.smoothers import permute_f_3d
from yancc.trajectories import (
    MDKE,
    MDKEPitch,
    MDKEPitchAngleScattering,
    MDKETheta,
    MDKEZeta,
)


def extract_blocks(a, m):
    """Get diagonal blocks from a of size m."""
    return np.array(
        [a[i * m : (i + 1) * m, i * m : (i + 1) * m] for i in range(a.shape[0] // m)]
    )


def test_get_block_diag(pitchgrid, field):
    """Test that using jax tricks to get the block diagonal works correctly."""
    nt, nz = field.ntheta, field.nzeta
    nl = pitchgrid.nxi
    nu = 1e-2
    E_psi = 1e-1
    p1 = "4d"
    p2 = 4

    sizes = {"a": nl, "t": nt, "z": nz}

    for axorder in itertools.permutations("atz"):
        for gauge in [True, False]:
            axorder = "".join(axorder)
            f = MDKETheta(field, pitchgrid, E_psi, p1, p2, axorder, gauge=gauge)
            A = f.as_matrix()
            np.testing.assert_allclose(
                np.diag(A), f.diagonal(), err_msg=axorder + str(gauge)
            )
            B = extract_blocks(A, sizes[axorder[-1]])
            np.testing.assert_allclose(
                B, f.block_diagonal(), err_msg=axorder + str(gauge)
            )

    for axorder in itertools.permutations("atz"):
        for gauge in [True, False]:
            axorder = "".join(axorder)
            f = MDKEZeta(field, pitchgrid, E_psi, p1, p2, axorder, gauge=gauge)
            A = f.as_matrix()
            np.testing.assert_allclose(
                np.diag(A), f.diagonal(), err_msg=axorder + str(gauge)
            )
            B = extract_blocks(A, sizes[axorder[-1]])
            np.testing.assert_allclose(
                B, f.block_diagonal(), err_msg=axorder + str(gauge)
            )

    for axorder in itertools.permutations("atz"):
        for gauge in [True, False]:
            axorder = "".join(axorder)
            f = MDKEPitch(field, pitchgrid, E_psi, p1, p2, axorder, gauge=gauge)
            A = f.as_matrix()
            np.testing.assert_allclose(
                np.diag(A), f.diagonal(), err_msg=axorder + str(gauge)
            )
            B = extract_blocks(A, sizes[axorder[-1]])
            np.testing.assert_allclose(
                B, f.block_diagonal(), err_msg=axorder + str(gauge)
            )

    for axorder in itertools.permutations("atz"):
        for gauge in [True, False]:
            axorder = "".join(axorder)
            f = MDKEPitchAngleScattering(
                field, pitchgrid, nu, p1, p2, axorder, gauge=gauge
            )
            A = f.as_matrix()
            np.testing.assert_allclose(
                np.diag(A), f.diagonal(), err_msg=axorder + str(gauge)
            )
            B = extract_blocks(A, sizes[axorder[-1]])
            np.testing.assert_allclose(
                B, f.block_diagonal(), err_msg=axorder + str(gauge)
            )

    for axorder in itertools.permutations("atz"):
        for gauge in [True, False]:
            axorder = "".join(axorder)
            f = MDKE(field, pitchgrid, E_psi, nu, p1, p2, axorder, gauge=gauge)
            A = f.as_matrix()
            np.testing.assert_allclose(
                np.diag(A), f.diagonal(), err_msg=axorder + str(gauge)
            )
            B = extract_blocks(A, sizes[axorder[-1]])
            np.testing.assert_allclose(
                B, f.block_diagonal(), err_msg=axorder + str(gauge)
            )


def test_permutations(field, pitchgrid):
    """Test that re-ordering the grid points gives equivalent operators."""
    p1 = "2a"
    p2 = 2
    E_psi = 1e-4
    nu = 1e-4
    N = field.ntheta * field.nzeta * pitchgrid.nxi

    A0f = MDKE(
        field, pitchgrid, E_psi, nu, p1=p1, p2=p2, axorder="atz", gauge=True
    ).as_matrix()
    A1f = MDKE(
        field, pitchgrid, E_psi, nu, p1=p1, p2=p2, axorder="zat", gauge=True
    ).as_matrix()
    A2f = MDKE(
        field, pitchgrid, E_psi, nu, p1=p1, p2=p2, axorder="tza", gauge=True
    ).as_matrix()

    P0f = jax.jacfwd(permute_f_3d)(np.zeros(N), field, pitchgrid, "atz")
    P1f = jax.jacfwd(permute_f_3d)(np.zeros(N), field, pitchgrid, "zat")
    P2f = jax.jacfwd(permute_f_3d)(np.zeros(N), field, pitchgrid, "tza")

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
