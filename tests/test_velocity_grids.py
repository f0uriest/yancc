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
    LegendrePitchAngleGrid,
    MaxwellSpeedGrid,
    NonUniformPitchAngleGrid,
    QuadraticPitchAngleGrid,
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


@pytest.mark.parametrize("c", [0.0, 0.5, 0.8, 1.0])
def test_quadratic_pitch_quadrature(c):
    """Test that QuadraticPitchAngleGrid.wxi properly integrates functions."""
    grid = QuadraticPitchAngleGrid(101, c)
    f = lambda x: jnp.exp(-((3 * x) ** 2)) * jnp.cos(5 * x)
    i1 = (f(grid.xi) * grid.wxi).sum()
    i2 = quadax.quadgk(f, jnp.array((-1, 1)))[0]
    np.testing.assert_allclose(i1, i2, rtol=1e-5)


def test_quadratic_pitch_reduces_to_uniform():
    """c=0 should give the same nodes as the uniform grid."""
    quad = QuadraticPitchAngleGrid(31, 0.0)
    uni = UniformPitchAngleGrid(31)
    np.testing.assert_allclose(quad.alpha, uni.alpha)
    np.testing.assert_allclose(quad.xi, uni.xi)


@pytest.mark.parametrize("na", [31, 101])
def test_quadratic_pitch_properties(na):
    """Nodes should be monotonic, odd-symmetric in xi, and span (-1, 1)."""
    grid = QuadraticPitchAngleGrid(na, 0.7)
    assert grid.nalpha == na
    assert grid.xi.min() > -1 and grid.xi.max() < 1
    assert np.all(np.diff(grid.xi) > 0)
    # odd map + symmetric a-grid => xi is antisymmetric about 0
    np.testing.assert_allclose(grid.xi, -grid.xi[::-1], atol=1e-14)
    # weights integrate f(x)=1 over (-1, 1)
    np.testing.assert_allclose(grid.wxi.sum(), 2.0, rtol=1e-12)


@pytest.mark.parametrize("c", [0.5, 0.8])
def test_quadratic_pitch_node_packing(c):
    """c>0 packs nodes more densely near v||=0 (xi=0) than the uniform grid."""
    na = 101
    mid = na // 2
    quad = QuadraticPitchAngleGrid(na, c)
    uni = UniformPitchAngleGrid(na)
    quad_dxi = quad.xi[mid + 1] - quad.xi[mid]
    uni_dxi = uni.xi[mid + 1] - uni.xi[mid]
    assert quad_dxi < uni_dxi


def test_quadratic_pitch_validation():
    """Even na and out-of-range c should raise."""
    with pytest.raises(Exception, match="nalpha must be odd"):
        QuadraticPitchAngleGrid(30, 0.5)
    with pytest.raises(Exception, match="c must be between"):
        QuadraticPitchAngleGrid(31, 1.5)
    with pytest.raises(Exception, match="c must be between"):
        QuadraticPitchAngleGrid(31, -0.1)


def test_quadratic_pitch_resample():
    """Resampling should change na but preserve the packing parameter c."""
    grid = QuadraticPitchAngleGrid(31, 0.7)
    new = grid.resample(51)
    assert new.nalpha == 51
    np.testing.assert_allclose(new.c, grid.c)
    np.testing.assert_allclose(new.xi, QuadraticPitchAngleGrid(51, 0.7).xi)


def test_nonuniform_pitch_custom_map():
    """A custom odd map should define the node spacing, and survive resampling."""
    map_func = lambda x: 0.7 * x**3 + 0.3 * x
    grid = NonUniformPitchAngleGrid(31, map_func)
    # this particular map matches the c=0.7 quadratic grid
    quad = QuadraticPitchAngleGrid(31, 0.7)
    np.testing.assert_allclose(grid.xi, quad.xi)
    np.testing.assert_allclose(grid.wxi, quad.wxi)

    new = grid.resample(51)
    assert new.nalpha == 51
    np.testing.assert_allclose(new.xi, NonUniformPitchAngleGrid(51, map_func).xi)


def test_nonuniform_pitch_identity_map():
    """The identity map reproduces the uniform a-grid (tests the [0,pi] remapping)."""
    na = 31
    grid = NonUniformPitchAngleGrid(na, lambda x: x)
    a = jnp.linspace(0, jnp.pi, na, endpoint=False) + jnp.pi / (2 * na)
    np.testing.assert_allclose(grid.alpha, a)
    np.testing.assert_allclose(grid.xi, -jnp.cos(a))


def test_maxwell_speed_grid_resample():
    """resample() should give a fresh grid of the requested size, and the
    high-nx branch (which builds its own recurrence rather than using the
    tabulated default) should still produce a valid quadrature.
    """
    g_lo = MaxwellSpeedGrid(8)
    assert g_lo.nx == 8
    g_hi = g_lo.resample(24)  # crosses the nx==20 threshold -> generate_recurrence
    assert g_hi.nx == 24
    assert g_hi.x.shape == (24,)
    assert g_hi.gauge_idx.shape == (2,)

    # Verify the high-nx grid still integrates a known function correctly.
    weight = g_hi.xrec.weight
    p = np.random.default_rng(0).random(24)

    def foo(x):
        return weight(x) * orthax.orthval(x, p, g_hi.xrec)

    np.testing.assert_allclose(
        quadax.quadgk(foo, jnp.array((0, np.inf)))[0],
        (foo(g_hi.x) * g_hi.wx).sum(),
        rtol=1e-6,
    )


def test_legendre_pitch_grid_resample():
    """LegendrePitchAngleGrid.resample returns a fresh grid of the
    requested size.
    """
    g = LegendrePitchAngleGrid(6)
    assert g.nalpha == 6
    g2 = g.resample(10)
    assert isinstance(g2, LegendrePitchAngleGrid)
    assert g2.nalpha == 10
    assert g2.xi.shape == (10,)


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


def test_composite_newton_cotes_default_limits():
    """Omitting global_limits uses the first/last sample points as the domain."""
    x = jnp.linspace(-1.0, 1.0, 13)
    w_default = composite_newton_cotes_weights(x, order=2)
    w_explicit = composite_newton_cotes_weights(x, order=2, global_limits=(x[0], x[-1]))
    np.testing.assert_allclose(w_default, w_explicit, atol=1e-12)
    # weights of a quadrature rule on [-1, 1] sum to the interval length.
    np.testing.assert_allclose(jnp.sum(w_default), 2.0, atol=1e-12)
