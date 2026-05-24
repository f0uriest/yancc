"""Tests for multigrid parts."""

import numpy as np
import pytest

from yancc.multigrid import (
    Prolongation,
    Restriction,
    _half_next_odd,
    get_grid_resolutions,
)
from yancc.velocity_grids import UniformPitchAngleGrid


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
    N_c = nx * pitchgrid_c.na * field_c.ntheta * field_c.nzeta
    N_f = nx * pitchgrid_f.na * field_f.ntheta * field_f.nzeta

    t_c, t_f = field_c.theta, field_f.theta
    z_c, z_f = field_c.zeta, field_f.zeta
    a_c, a_f = pitchgrid_c.a, pitchgrid_f.a
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

    Pmat = P.as_matrix()
    Rmat = R.as_matrix()

    np.testing.assert_allclose(Pmat @ f_c, P.mv(f_c), atol=1e-12)
    np.testing.assert_allclose(Rmat @ f_f, R.mv(f_f), atol=1e-12)
    # Restriction is the volume-weighted transpose of prolongation.
    np.testing.assert_allclose(Pmat.T, Rmat * N_f / N_c, atol=1e-12)

    P_cubic = Prolongation(
        field_c, field_f, pitchgrid_c, pitchgrid_f, prefix_size=nx, method="cubic"
    )
    np.testing.assert_allclose(f_f, P_cubic.mv(f_c), atol=1e-2, rtol=1e-2)


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
