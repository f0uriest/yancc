"""Tests for preconditioners."""

from typing import cast

import equinox as eqx
import jax
import numpy as np
import pytest

from yancc._collisions import RosenbluthPotentials
from yancc._misc import DKESources
from yancc._preconditioner import (
    DKEMPreconditioner,
    DKEPreconditioner,
    MDKEPreconditioner,
)
from yancc._smoothers import DKEJacobiSmoother, MDKEJacobiSmoother
from yancc._trajectories import DKE
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


def test_dke_coarse_solve(field, pitchgrid, speedgrid, species2, potentials2):
    """The coarse solve inverts the coarse operator on its range, without null part.

    By default the DKE preconditioner levels keep the density and energy null space,
    and the coarse solve factors a shifted operator. For a right hand side in the
    range of the coarse operator it must return a solution with no component along
    the null modes, to the accuracy of the LU solve.
    """
    M = DKEPreconditioner(
        field, pitchgrid, speedgrid, species2, 100.0, None, potentials2
    )
    Ac = cast(DKE, M.operators[0])
    assert not Ac.gauge
    x = jax.random.normal(jax.random.key(0), (Ac.in_size(),))
    b = Ac.mv(x)
    y = M.coarse_opinv.mv(b)
    np.testing.assert_allclose(Ac.mv(y), b, atol=1e-5 * np.abs(b).max())
    B = DKESources(Ac.field, Ac.pitchgrid, speedgrid, species2).as_matrix()
    B = np.linalg.qr(B)[0]
    assert np.linalg.norm(B.T @ y) < 1e-3 * np.linalg.norm(y)

    # the point gauge is still available, and other values are rejected
    M = DKEPreconditioner(
        field, pitchgrid, speedgrid, species2, 100.0, None, potentials2, gauge=True
    )
    assert cast(DKE, M.operators[0]).gauge
    with pytest.raises(ValueError, match="gauge"):
        DKEPreconditioner(
            field, pitchgrid, speedgrid, species2, 100.0, None, potentials2, gauge="x"
        )


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
        gauge=False,
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
