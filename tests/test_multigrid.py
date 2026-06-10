"""Tests for multigrid parts."""

import jax.numpy as jnp
import numpy as np
import pytest

from yancc.misc import dke_rhs
from yancc.multigrid import (
    Prolongation,
    Restriction,
    _half_next_odd,
    get_dke_operators,
    get_fields_grids,
    get_grid_resolutions,
    get_prolongations,
    get_restrictions,
    krylov1_coarse_correction,
    krylov1s_coarse_correction,
    krylov2_coarse_correction,
    krylov2s_coarse_correction,
    standard_coarse_correction,
)
from yancc.velocity_grids import UniformPitchAngleGrid

COARSE_CORRECTIONS = [
    standard_coarse_correction,
    krylov1_coarse_correction,
    krylov1s_coarse_correction,
    krylov2_coarse_correction,
    krylov2s_coarse_correction,
]


@pytest.mark.parametrize(
    "k, expected",
    [
        (1, 1),  # k // 2 == 0 -> floor branch returns 1
        (8, 5),  # k // 2 == 4 (even) -> 4 + 1
        (6, 3),  # k // 2 == 3 (odd)  -> 3
    ],
)
def test_half_next_odd(k, expected):
    """``_half_next_odd`` halves ``k`` and rounds to the next odd integer,
    flooring at 1 (coarsening must stay odd and >= 1).
    """
    assert _half_next_odd(k) == expected


@pytest.mark.parametrize("nx", [1, 3])
def test_prolongation_restriction(field, nx):
    """Test that prolongation and restriction are transposes (volume weighted)."""
    field_c = field.resample(11, 13)
    field_f = field.resample(15, 17)
    pitchgrid_c = UniformPitchAngleGrid(11)
    pitchgrid_f = UniformPitchAngleGrid(15)
    N_c = nx * pitchgrid_c.nalpha * field_c.ntheta * field_c.nzeta
    N_f = nx * pitchgrid_f.nalpha * field_f.ntheta * field_f.nzeta

    t_c, t_f = field_c.theta, field_f.theta
    z_c, z_f = field_c.zeta, field_f.zeta
    a_c, a_f = pitchgrid_c.alpha, pitchgrid_f.alpha
    x = (1 + np.arange(nx)) ** 2

    def foo(x, a, t, z):
        return (
            x
            * np.sin(t)
            * np.cos(z * field_c.NFP)
            * (np.exp(-(a**2)) + 2 * np.exp(-((a - np.pi) ** 2)))
        )

    f_c = foo(
        x[:, None, None, None],
        a_c[None, :, None, None],
        t_c[None, None, :, None],
        z_c[None, None, None, :],
    ).flatten()
    f_f = foo(
        x[:, None, None, None],
        a_f[None, :, None, None],
        t_f[None, None, :, None],
        z_f[None, None, None, :],
    ).flatten()

    P = Prolongation(field_c, field_f, pitchgrid_c, pitchgrid_f, prefix_size=nx)
    R = Restriction(field_c, field_f, pitchgrid_c, pitchgrid_f, prefix_size=nx)

    # prolongation maps coarse -> fine; restriction maps fine -> coarse.
    assert P.in_structure().shape == (N_c,)
    assert P.out_structure().shape == (N_f,)
    assert R.in_structure().shape == (N_f,)
    assert R.out_structure().shape == (N_c,)

    Pmat = P.as_matrix()
    Rmat = R.as_matrix()

    np.testing.assert_allclose(Pmat @ f_c, P.mv(f_c), atol=1e-12)
    np.testing.assert_allclose(Rmat @ f_f, R.mv(f_f), atol=1e-12)
    # Restriction is the volume-weighted transpose of prolongation.
    np.testing.assert_allclose(Pmat.T, Rmat * N_f / N_c, atol=1e-12)

    P_cubic = Prolongation(
        field_c, field_f, pitchgrid_c, pitchgrid_f, prefix_size=nx, method="cubic"
    )
    np.testing.assert_allclose(f_f, P_cubic.mv(f_c), atol=2e-2, rtol=2e-2)


def test_get_grid_resolutions_max_grids():
    """Specifying max_grids derives the coarsening factor and caps the levels."""
    res = get_grid_resolutions(2, 10, 51, 51, 51, max_grids=4)
    assert len(res) == 4
    # list is ordered coarse -> fine, so the finest grid is last
    assert res[-1] == (2, 10, 51, 51, 51)
    # algebraic axes refine monotonically and the coarsest stays >= the minimums
    for i in range(len(res) - 1):
        for ax in (2, 3, 4):
            assert res[i + 1][ax] >= res[i][ax]
    assert res[0][2] >= 5 and res[0][3] >= 5 and res[0][4] >= 5


def test_get_grid_resolutions_coarsening_factor():
    """Specifying coarsening_factor derives the number of levels."""
    res = get_grid_resolutions(2, 10, 51, 51, 51, coarsening_factor=2.0)
    assert len(res) >= 2
    assert res[-1] == (2, 10, 51, 51, 51)


def test_get_grid_resolutions_conflicting_args():
    """Cannot specify both coarsening_factor and max_grids."""
    with pytest.raises(ValueError):
        get_grid_resolutions(2, 10, 51, 51, 51, coarsening_factor=2, max_grids=4)


def _two_level_dke(field, pitchgrid, speedgrid, species):
    """Build a coarse+fine DKE 2-level setup with matching P/R operators."""
    Erho = jnp.array(0.0)
    operator_weights = jnp.ones(8).at[-2:].set(0)
    ns, nx = len(species), speedgrid.nx
    # (ns, nx, na, nt, nz), coarse then fine. Need >= 5 points per algebraic axis
    # (a, t, z) for the finite-difference stencils.
    resolutions = [
        (ns, nx, 5, 5, 5),
        (ns, nx, pitchgrid.nalpha, field.ntheta, field.nzeta),
    ]
    fields, grids = get_fields_grids(field, pitchgrid, resolutions)
    ops = get_dke_operators(
        fields,
        grids,
        speedgrid,
        species,
        Erho,
        [],
        None,
        "2d",
        2,
        True,
        operator_weights=operator_weights,
    )
    prolongations = get_prolongations(fields, grids, prefix_size=ns * nx)
    restrictions = get_restrictions(fields, grids, prefix_size=ns * nx)
    return fields, grids, ops, prolongations, restrictions, Erho


@pytest.mark.parametrize("correction", COARSE_CORRECTIONS)
def test_coarse_correction_reduces_error(
    field, pitchgrid, speedgrid, species1, correction
):
    """A 2-level coarse-grid correction should reduce the error vs the true solution.

    The residual norm is not guaranteed to decrease (the operator is not SPD), but
    the error relative to the exact solution should.
    """
    fields, grids, ops, P, R, Erho = _two_level_dke(
        field, pitchgrid, speedgrid, species1
    )
    A_c, A_f = ops[0], ops[1]
    b = dke_rhs(
        fields[1], grids[1], speedgrid, species1, Erho, include_constraints=False
    )
    x_true = jnp.linalg.solve(A_f.as_matrix(), b)
    x = jnp.zeros_like(b)
    rk = b - A_f.mv(x)
    # exact coarse solve of the restricted residual, prolong back to the fine grid
    ykm1 = jnp.linalg.solve(A_c.as_matrix(), R[0].mv(rk))
    yk = P[0].mv(ykm1)
    x_new = correction(x, 1, 0, A_f, yk, rk, 1.0, verbose=True)
    err_before = float(jnp.linalg.norm(x - x_true))
    err_after = float(jnp.linalg.norm(x_new - x_true))
    assert err_after < err_before
