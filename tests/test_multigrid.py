"""Tests for multigrid parts."""

import jax
import numpy as np
import pytest

from yancc.multigrid import interpolate
from yancc.velocity_grids import UniformPitchAngleGrid


@pytest.mark.parametrize("nx", [1, 3])
def test_interpolate(field, nx):
    """Test that interpolation is transposed correctly."""
    field1 = field
    field2 = field.resample(11, 13)
    pitchgrid1 = UniformPitchAngleGrid(5)
    pitchgrid2 = UniformPitchAngleGrid(11)
    N1 = nx * pitchgrid1.nxi * field1.ntheta * field1.nzeta
    N2 = nx * pitchgrid2.nxi * field2.ntheta * field2.nzeta

    f1 = np.random.random(N1)
    f2 = np.random.random(N2)
    J1 = jax.jacfwd(interpolate)(f1, field1, field2, pitchgrid1, pitchgrid2)
    J2 = jax.jacfwd(interpolate)(f2, field2, field1, pitchgrid2, pitchgrid1)

    np.testing.assert_allclose(
        J1 @ f1, interpolate(f1, field1, field2, pitchgrid1, pitchgrid2)
    )
    np.testing.assert_allclose(
        J2 @ f2, interpolate(f2, field2, field1, pitchgrid2, pitchgrid1)
    )
    np.testing.assert_allclose(J1.T, J2 * N2 / N1)
