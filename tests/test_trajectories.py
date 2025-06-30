"""Tests for MDKE operators."""

import numpy as np
import pytest

import yancc.trajectories as trajectories
import yancc.trajectories_scipy as trajectories_scipy


def extract_blocks(a, m):
    """Get diagonal blocks from a of size m."""
    return np.array(
        [a[i * m : (i + 1) * m, i * m : (i + 1) * m] for i in range(a.shape[0] // m)]
    )


@pytest.mark.parametrize("p1", ["1a", "4b"])
@pytest.mark.parametrize("p2", [2, 4])
@pytest.mark.parametrize("E_psi", [1e-3, 1e3])
@pytest.mark.parametrize("nu", [1e-3, 1e3])
@pytest.mark.parametrize("gauge", [True, False])
def test_scipy_operators(p1, p2, E_psi, nu, gauge, field, pitchgrid):
    """Test that scipy sparse matrices are the same as jax jacobians."""
    A1 = trajectories_scipy.mdke(field, pitchgrid, E_psi, nu, p1, p2, gauge=gauge)
    A2 = trajectories.MDKE(field, pitchgrid, E_psi, nu, p1, p2, "atz", gauge=gauge)
    np.testing.assert_allclose(A1.toarray(), A2.as_matrix())


@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals(axorder, field, pitchgrid, speedgrid, species2):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    E_psi = np.array(1e3)

    # DKE
    f = trajectories.DKE(
        field, pitchgrid, speedgrid, species2, E_psi, "2d", 4, axorder=axorder
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    # speed
    f = trajectories.DKESpeed(
        field, pitchgrid, speedgrid, species2, E_psi, axorder=axorder
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    # energy scattering
    f = trajectories.DKEEnergyScattering(
        field, pitchgrid, speedgrid, species2, axorder=axorder
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    for p2 in [2, 4, 6]:
        f = trajectories.DKEPitchAngleScattering(
            field, pitchgrid, speedgrid, species2, p2=p2, axorder=axorder
        )
        A = f.as_matrix()
        np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
        B = extract_blocks(A, sizes[axorder[-1]])
        np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    for p1 in ["1a", "2b", "3c", "4d"]:
        # theta
        f = trajectories.DKETheta(
            field, pitchgrid, speedgrid, species2, E_psi, p1=p1, axorder=axorder
        )
        A = f.as_matrix()
        np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
        B = extract_blocks(A, sizes[axorder[-1]])
        np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

        # zeta
        f = trajectories.DKEZeta(
            field, pitchgrid, speedgrid, species2, E_psi, p1=p1, axorder=axorder
        )
        A = f.as_matrix()
        np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
        B = extract_blocks(A, sizes[axorder[-1]])
        np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

        # pitch
        f = trajectories.DKEPitch(
            field, pitchgrid, speedgrid, species2, E_psi, p1=p1, axorder=axorder
        )
        A = f.as_matrix()
        np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
        B = extract_blocks(A, sizes[axorder[-1]])
        np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
