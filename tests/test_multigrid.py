"""Tests for multigrid parts."""

import jax
import numpy as np
import pytest

from yancc.multigrid import interpolate
from yancc.velocity_grids import UniformPitchAngleGrid


@pytest.mark.parametrize("nx", [1, 3])
def test_interpolate(field, nx):
    """Test that interpolation is transposed correctly."""
    field1 = field.resample(11, 13)
    field2 = field.resample(15, 17)
    pitchgrid1 = UniformPitchAngleGrid(11)
    pitchgrid2 = UniformPitchAngleGrid(15)
    N1 = nx * pitchgrid1.nxi * field1.ntheta * field1.nzeta
    N2 = nx * pitchgrid2.nxi * field2.ntheta * field2.nzeta

    t1, t2 = field1.theta, field2.theta
    z1, z2 = field1.zeta, field2.zeta
    a1, a2 = pitchgrid1.gamma, pitchgrid2.gamma
    x = (1 + np.arange(nx)) ** 2

    def foo(x, a, t, z):
        return (
            x
            * np.sin(t)
            * np.cos(z * field1.NFP)
            * (np.exp(-(a**2)) + 2 * np.exp(-((a - np.pi) ** 2)))
        )

    f1 = foo(
        x[:, None, None, None],
        a1[None, :, None, None],
        t1[None, None, :, None],
        z1[None, None, None, :],
    ).flatten()
    f2 = foo(
        x[:, None, None, None],
        a2[None, :, None, None],
        t2[None, None, :, None],
        z2[None, None, None, :],
    ).flatten()
    J1 = jax.jacfwd(interpolate)(f1, field1, field2, pitchgrid1, pitchgrid2)
    J2 = jax.jacfwd(interpolate)(f2, field2, field1, pitchgrid2, pitchgrid1)

    np.testing.assert_allclose(
        J1 @ f1, interpolate(f1, field1, field2, pitchgrid1, pitchgrid2), atol=1e-12
    )
    np.testing.assert_allclose(
        J2 @ f2, interpolate(f2, field2, field1, pitchgrid2, pitchgrid1), atol=1e-12
    )
    np.testing.assert_allclose(J1.T, J2 * N2 / N1, atol=1e-12)

    np.testing.assert_allclose(
        f2,
        interpolate(f1, field1, field2, pitchgrid1, pitchgrid2, method="cubic"),
        atol=1e-2,
        rtol=1e-2,
    )
