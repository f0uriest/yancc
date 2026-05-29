"""Tests for misc helpers (RHS assembly and magnetic drifts)."""

import numpy as np

from yancc.misc import dke_rhs
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def test_dke_rhs_single_rhs_in_span_of_unit_drives(field, species2):
    """The combined single_rhs drive lies in the span of the unit drives."""
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(3)
    ns = len(species2)
    N = ns * speedgrid.nx * pitchgrid.na * field.ntheta * field.nzeta

    unit = dke_rhs(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho=10.0,
        include_constraints=True,
        single_rhs=False,
    )
    assert unit.shape == (3 * ns, N + 2 * ns)

    combined = dke_rhs(
        field, pitchgrid, speedgrid, species2, Erho=10.0, single_rhs=True
    )
    assert combined.shape == (N + 2 * ns,)

    # the combined drive is a linear combination of the unit drives, so it must lie
    # in their row space: the least-squares fit should reproduce it exactly.
    unit = np.asarray(unit)
    combined = np.asarray(combined)
    coeffs, *_ = np.linalg.lstsq(unit.T, combined, rcond=None)
    np.testing.assert_allclose(unit.T @ coeffs, combined, atol=1e-12)
