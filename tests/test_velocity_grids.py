"""Tests for velocity grids/integrals/derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import orthax
import quadax

from yancc.misc import _d3v
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def test_speed_quadrature():
    """Test that speedgrid.wx properly integrates functions."""
    grid = MaxwellSpeedGrid(10)
    p = np.random.random(10)
    weight = grid.xrec.weight

    def foo(x):
        return weight(x) * orthax.orthval(x, p, grid.xrec)

    np.testing.assert_allclose(
        quadax.quadgk(foo, jnp.array((0, np.inf)))[0], (foo(grid.x) * grid.wx).sum()
    )


def test_speed_fit():
    """Test that speedgrid.xvander properly recovers basis function coefficients."""
    grid = MaxwellSpeedGrid(10)
    p = np.random.random(10)
    weight = grid.xrec.weight

    def foo(x):
        return weight(x) * orthax.orthval(x, p, grid.xrec)

    fx = foo(grid.x)
    fitp = jnp.linalg.lstsq(grid.xvander, fx)[0]
    np.testing.assert_allclose(p, fitp)


def test_speed_derivatives():
    """Test that derivative matrix properly differentiates basis functions."""
    grid = MaxwellSpeedGrid(10)
    p = np.random.random(10)
    weight = grid.xrec.weight

    def foo(x):
        return weight(x) * orthax.orthval(x, p, grid.xrec)

    f = foo(grid.x)
    df1 = jnp.vectorize(jax.grad(foo))(grid.x)
    df2 = grid.Dx_pseudospectral @ f
    np.testing.assert_allclose(df1, df2)

    f = foo(grid.x)
    ddf1 = jnp.vectorize(jax.grad(jax.grad(foo)))(grid.x)
    ddf2 = grid.D2x_pseudospectral @ f
    np.testing.assert_allclose(ddf1, ddf2)


def test_pitch_quadrature():
    """Test that pitchgrid.wxi properly integrates functions."""
    pitchgrid = UniformPitchAngleGrid(31)
    rng = np.random.default_rng(123)
    p = rng.random(10)
    f = lambda x: jnp.polyval(p, x)
    i1 = (f(pitchgrid.xi) * pitchgrid.wxi).sum()
    i2 = quadax.quadgk(f, jnp.array((-1, 1)))[0]
    np.testing.assert_allclose(i1, i2)


def test_velocity_integral(speedgrid, pitchgrid, species1):
    """Test integrals over all velocity space."""
    d3v = _d3v(speedgrid, pitchgrid, species1)
    n = (species1[0](speedgrid.x * species1[0].v_thermal)[None, :, None] * d3v).sum()
    np.testing.assert_allclose(n, species1[0].density)
