"""Tests for velocity grids/integrals/derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import orthax
import quadax

from yancc.velocity_grids import MaxwellSpeedGrid


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
