"""Tests for constructing smoothing operators."""

import jax
import numpy as np
import pytest

from yancc.collisions import (
    EnergyScattering,
    FieldPartCD,
    FieldPartCG,
    FieldPartCH,
    FieldParticleScattering,
    FokkerPlanckLandau,
    MDKEPitchAngleScattering,
    PitchAngleScattering,
    RosenbluthPotentials,
)
from yancc.smoothers import permute_f_3d
from yancc.trajectories import (
    DKE,
    MDKE,
    DKEPitch,
    DKESpeed,
    DKETheta,
    DKEZeta,
    MDKEPitch,
    MDKETheta,
    MDKEZeta,
)


def extract_blocks(a, m):
    """Get diagonal blocks from a of size m."""
    return np.array(
        [a[i * m : (i + 1) * m, i * m : (i + 1) * m] for i in range(a.shape[0] // m)]
    )


@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
@pytest.mark.parametrize("gauge", [True, False])
def test_get_block_diag_mdke(pitchgrid, field, axorder, gauge):
    """Test that using jax tricks to get the block diagonal works correctly."""
    nt, nz = field.ntheta, field.nzeta
    nl = pitchgrid.nxi
    nuhat = 1e-2
    erhohat = 1e-1
    p1 = "4d"
    p2 = 4

    sizes = {"a": nl, "t": nt, "z": nz}

    f = MDKETheta(field, pitchgrid, erhohat, p1, p2, axorder, gauge=gauge)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder + str(gauge))
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder + str(gauge))

    f = MDKEZeta(field, pitchgrid, erhohat, p1, p2, axorder, gauge=gauge)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder + str(gauge))
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder + str(gauge))

    f = MDKEPitch(field, pitchgrid, erhohat, p1, p2, axorder, gauge=gauge)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder + str(gauge))
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder + str(gauge))

    f = MDKEPitchAngleScattering(field, pitchgrid, nuhat, p1, p2, axorder, gauge=gauge)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder + str(gauge))
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder + str(gauge))

    f = MDKE(field, pitchgrid, erhohat, nuhat, p1, p2, axorder, gauge=gauge)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder + str(gauge))
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder + str(gauge))


@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_get_block_diag_dke(pitchgrid, speedgrid, species2, field, axorder):
    """Test that using jax tricks to get the block diagonal works correctly."""
    nt, nz = field.ntheta, field.nzeta
    nl = pitchgrid.nxi
    nx = speedgrid.nx
    potentials = RosenbluthPotentials(speedgrid, species2)
    Erho = 1e-1
    p1 = "4d"
    p2 = 4

    sizes = {"s": len(species2), "x": nx, "a": nl, "t": nt, "z": nz}

    f = DKETheta(field, pitchgrid, speedgrid, species2, Erho, p1, p2, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = DKEZeta(field, pitchgrid, speedgrid, species2, Erho, p1, p2, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = DKEPitch(field, pitchgrid, speedgrid, species2, Erho, p1, p2, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = DKESpeed(field, pitchgrid, speedgrid, species2, Erho, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = PitchAngleScattering(field, pitchgrid, speedgrid, species2, p2, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = EnergyScattering(field, pitchgrid, speedgrid, species2, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = FieldPartCD(field, pitchgrid, speedgrid, species2, potentials, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = FieldPartCG(field, pitchgrid, speedgrid, species2, potentials, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = FieldPartCH(field, pitchgrid, speedgrid, species2, potentials, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = FieldParticleScattering(
        field, pitchgrid, speedgrid, species2, potentials, axorder
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = FokkerPlanckLandau(
        field, pitchgrid, speedgrid, species2, potentials, p2, axorder
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)

    f = DKE(field, pitchgrid, speedgrid, species2, Erho, potentials, p1, p2, axorder)
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


def test_permutations_mdke(field, pitchgrid):
    """Test that re-ordering the grid points gives equivalent operators."""
    p1 = "2a"
    p2 = 2
    erhohat = 1e-4
    nuhat = 1e-4
    N = field.ntheta * field.nzeta * pitchgrid.nxi

    A0f = MDKE(
        field, pitchgrid, erhohat, nuhat, p1=p1, p2=p2, axorder="atz", gauge=True
    ).as_matrix()
    A1f = MDKE(
        field, pitchgrid, erhohat, nuhat, p1=p1, p2=p2, axorder="zat", gauge=True
    ).as_matrix()
    A2f = MDKE(
        field, pitchgrid, erhohat, nuhat, p1=p1, p2=p2, axorder="tza", gauge=True
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
