"""Tests for preconditioners."""

import numpy as np
import pytest

from yancc.collisions import RosenbluthPotentials
from yancc.preconditioner import (
    DKEMPreconditioner,
    DKEPreconditioner,
    MDKEPreconditioner,
)
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


@pytest.mark.parametrize(
    "build",
    [
        lambda f, pg, sg, sp, pot: MDKEPreconditioner(f, pg, 0.01, 0.1),
        lambda f, pg, sg, sp, pot: DKEPreconditioner(f, pg, sg, sp, 100.0, None, pot),
        lambda f, pg, sg, sp, pot: DKEMPreconditioner(
            field=f, pitchgrid=pg, speedgrid=sg, species=sp, Erho=100.0
        ),
    ],
    ids=["MDKE", "DKE", "DKEM"],
)
def test_preconditioner_interface(field, species1, build):
    """Materialize each preconditioner and its (approximate) transpose.

    as_matrix materializes the operator, and the materialized shape must
    match the declared in/out structure. The transpose of a multigrid cycle
    is only an approximate adjoint (not the exact matrix transpose), so we just
    require transpose().as_matrix() to be close to A.T.
    """
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(2)
    potentials = RosenbluthPotentials(speedgrid, species1)
    op = build(field, pitchgrid, speedgrid, species1, potentials)

    A = np.asarray(op.as_matrix())
    assert A.shape == (op.out_structure().shape[0], op.in_structure().shape[0])

    AT = np.asarray(op.transpose().as_matrix())
    np.testing.assert_allclose(AT, A.T, atol=1e-2 * np.abs(A).max())


def test_dke_preconditioner_smooth_type2(field, species1):
    """smooth_type=2 builds the Jacobi-2 smoothers instead of the default."""
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(2)
    potentials = RosenbluthPotentials(speedgrid, species1)
    op = DKEPreconditioner(
        field,
        pitchgrid,
        speedgrid,
        species1,
        100.0,
        None,
        potentials,
        smooth_type=2,
    )
    A = np.asarray(op.as_matrix())
    assert A.shape == (op.out_structure().shape[0], op.in_structure().shape[0])
