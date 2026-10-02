"""Tests for preconditioners."""

import equinox as eqx
import jax
import numpy as np
import pytest

from yancc._collisions import RosenbluthPotentials
from yancc._preconditioner import (
    DKEMPreconditioner,
    DKEPreconditioner,
    MDKEPreconditioner,
)
from yancc._smoothers import DKEJacobiSmoother, MDKEJacobiSmoother
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


def test_preconditioner_smoother_fd_order(field, species1):
    """Smoothers with their own FD order don't reuse the level operators."""
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(2)
    potentials = RosenbluthPotentials(speedgrid, species1)
    M = DKEPreconditioner(
        field,
        pitchgrid,
        speedgrid,
        species1,
        100.0,
        None,
        potentials,
        p1="2d",
        smooth_p1="4d",
        smooth_type="t",
    )
    sm = M.smoothers[-1][0]
    assert isinstance(sm, DKEJacobiSmoother)
    ref = DKEJacobiSmoother(
        sm.field,
        sm.pitchgrid,
        speedgrid,
        species1,
        100.0,
        potentials=potentials,
        p1="4d",
        p2=sm.p2,
        axorder=sm.axorder,
        gauge=True,
    )
    jax.tree_util.tree_map(
        np.testing.assert_allclose,
        eqx.filter(sm.mats, eqx.is_inexact_array),
        eqx.filter(ref.mats, eqx.is_inexact_array),
    )

    M = MDKEPreconditioner(
        field, pitchgrid, 0.01, 0.1, p1="2d", smooth_p1="4d", smooth_type="t"
    )
    sm = M.smoothers[-1][0]
    assert isinstance(sm, MDKEJacobiSmoother)
    ref = MDKEJacobiSmoother(
        sm.field, sm.pitchgrid, 0.01, 0.1, "4d", sm.p2, sm.axorder, True
    )
    jax.tree_util.tree_map(
        np.testing.assert_allclose,
        eqx.filter(sm.mats, eqx.is_inexact_array),
        eqx.filter(ref.mats, eqx.is_inexact_array),
    )
