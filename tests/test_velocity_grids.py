"""Tests for velocity grids/integrals/derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import orthax
import pytest
import quadax
import scipy.integrate

from yancc.misc import _d3v
from yancc.velocity_grids import (
    MaxwellSpeedGrid,
    UniformPitchAngleGrid,
    composite_newton_cotes_weights,
)


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


@pytest.mark.parametrize("order", [2, 4, 6])
def test_newton_cotes(order):
    """Test composite newton cotes integration."""
    Ns = 2 ** (np.arange(2, 10)).astype(int) + 1

    fun = lambda x: jnp.sin(10 * x) ** 2 / (1 + (5 * x) ** 2)
    domain_xi = (-1, 1)
    domain_a = (0, np.pi)

    c = 0.7

    def map1(x):
        x = 2 * (x / np.pi - 0.5)
        x = c * x**3 + (1 - c) * x
        x = (x + 1) / 2 * np.pi
        return x

    integral_exact = scipy.integrate.quad(fun, domain_xi[0], domain_xi[1])[0]

    errs_uni = []
    errs_non = []

    for N in Ns:
        a_uni = jnp.linspace(domain_a[0], domain_a[1], N, endpoint=False) + (
            domain_a[1] - domain_a[0]
        ) / (2 * N)
        xi_uni = -np.cos(a_uni)
        a_non = map1(a_uni)
        xi_non = -np.cos(a_non)

        f_uni = fun(xi_uni)
        f_non = fun(xi_non)
        w_uni = composite_newton_cotes_weights(
            xi_uni, order=order, global_limits=(domain_xi[0], domain_xi[1])
        )
        w_non = composite_newton_cotes_weights(
            xi_non, order=order, global_limits=(domain_xi[0], domain_xi[1])
        )
        integral_uni = jnp.sum(f_uni * w_uni)
        integral_non = jnp.sum(f_non * w_non)
        errs_uni.append(np.abs(integral_exact - integral_uni))
        errs_non.append(np.abs(integral_exact - integral_non))

    order_uni, _ = np.polyfit(np.log(Ns)[-4:], -np.log(errs_uni)[-4:], deg=1)
    order_non, _ = np.polyfit(np.log(Ns)[-4:], -np.log(errs_non)[-4:], deg=1)
    assert order_uni >= order
    assert order_non >= order
