"""Tests for computing Rosenbluth potentials."""

import jax.numpy as jnp
import mpmath
import numpy as np
import orthax
import pytest
import sympy

from yancc.collisions import RosenbluthPotentials
from yancc.utils import lGammainc, lGammaincc
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

from .conftest import _compute_G_sympy, _compute_H_sympy, _eval_f

mpmath.mp.dps = 100


def rosenbluth_ddG_jax(f, a, b, speedgrid, pitchgrid, potentials, Txi_inv):
    assert f.shape == (speedgrid.nx, pitchgrid.nxi, 1, 1)
    # convert nodal alpha -> legendre l
    f = jnp.einsum("la,xatz->xltz", Txi_inv, f)
    # convert nodal x -> speed k
    f = jnp.einsum("kx,xltz->kltz", speedgrid.xvander_inv, f)
    Gabxlk = potentials.ddGxlk[a, b, :, : potentials.legendregrid.nxi]
    df = jnp.einsum("xlk,kltz->xltz", Gabxlk, f)
    return df


def rosenbluth_dH_jax(f, a, b, speedgrid, pitchgrid, potentials, Txi_inv):
    assert f.shape == (speedgrid.nx, pitchgrid.nxi, 1, 1)
    # convert nodal alpha -> legendre l
    f = jnp.einsum("la,xatz->xltz", Txi_inv, f)
    # convert nodal x -> speed k
    f = jnp.einsum("kx,xltz->kltz", speedgrid.xvander_inv, f)
    Gabxlk = potentials.dHxlk[a, b, :, : potentials.legendregrid.nxi]
    df = jnp.einsum("xlk,kltz->xltz", Gabxlk, f)
    return df


def rosenbluth_H_jax(f, a, b, speedgrid, pitchgrid, potentials, Txi_inv):
    assert f.shape == (speedgrid.nx, pitchgrid.nxi, 1, 1)
    # convert nodal alpha -> legendre l
    f = jnp.einsum("la,xatz->xltz", Txi_inv, f)
    # convert nodal x -> speed k
    f = jnp.einsum("kx,xltz->kltz", speedgrid.xvander_inv, f)
    Gabxlk = potentials.Hxlk[a, b, :, : potentials.legendregrid.nxi]
    df = jnp.einsum("xlk,kltz->xltz", Gabxlk, f)
    return df


@pytest.fixture
def xgrid():
    """Speed grid for testing."""
    return MaxwellSpeedGrid(10)


@pytest.fixture
def xigrid():
    """Pitch angle grid for testing"""
    return UniformPitchAngleGrid(17)


@pytest.fixture
def potential_quad(xgrid, species2):
    """Single species potentials with quadrature."""
    return RosenbluthPotentials(
        xgrid,
        species2,
        nL=6,
        quad=True,
    )


@pytest.fixture
def potential_gamma(xgrid, species2):
    """Single species potential without quadrature."""
    return RosenbluthPotentials(
        xgrid,
        species2,
        nL=6,
        quad=False,
    )


@pytest.mark.parametrize("l", [2, 4])
@pytest.mark.parametrize("k", [2, 4])
def test_rosenbluth_derivatives(potential_quad, l, k):
    """Test derivatives of Rosenbluth potentials."""
    R = potential_quad
    x0 = np.linspace(0.1, 5, 3)
    eps = 1e-4
    dHfd = (R._Hlk(x0 + eps, l, k) - R._Hlk(x0 - eps, l, k)) / (2 * eps)
    dHan = R._dHlk(x0, l, k)
    np.testing.assert_allclose(dHfd, dHan, rtol=1e-2)
    dGfd = (R._Glk(x0 + eps, l, k) - R._Glk(x0 - eps, l, k)) / (2 * eps)
    dGan = R._dGlk(x0, l, k)
    np.testing.assert_allclose(dGfd, dGan, rtol=1e-2)
    d2Gfd = (R._dGlk(x0 + eps, l, k) - R._dGlk(x0 - eps, l, k)) / (2 * eps)
    d2Gan = R._ddGlk(x0, l, k)
    np.testing.assert_allclose(d2Gfd, d2Gan, rtol=1e-2)


def test_rosenbluth_quad_vs_gamma(potential_quad, potential_gamma):
    """Test for potentials using incomplete gamma functions."""
    R1 = potential_quad
    R2 = potential_gamma
    # a,a
    np.testing.assert_allclose(
        R1.Hxlk[0, 0],
        R2.Hxlk[0, 0],
        rtol=1e-8,
        atol=1e-8 * abs(R1.Hxlk[0, 0]).mean(),
    )
    np.testing.assert_allclose(
        R1.dHxlk[0, 0],
        R2.dHxlk[0, 0],
        rtol=1e-8,
        atol=1e-8 * abs(R1.dHxlk[0, 0]).mean(),
    )
    np.testing.assert_allclose(
        R1.ddGxlk[0, 0],
        R2.ddGxlk[0, 0],
        rtol=1e-4,
        atol=1e-8 * abs(R1.ddGxlk[0, 0]).mean(),
    )
    # a,b
    np.testing.assert_allclose(
        R1.Hxlk[0, 1],
        R2.Hxlk[0, 1],
        rtol=1e-6,
        atol=1e-6 * abs(R1.Hxlk[0, 1]).mean(),
    )
    np.testing.assert_allclose(
        R1.dHxlk[0, 1],
        R2.dHxlk[0, 1],
        rtol=1e-6,
        atol=1e-6 * abs(R1.dHxlk[0, 1]).mean(),
    )
    np.testing.assert_allclose(
        R1.ddGxlk[0, 1],
        R2.ddGxlk[0, 1],
        rtol=1e-6,
        atol=1e-6 * abs(R1.ddGxlk[0, 1]).mean(),
    )
    # b,a
    np.testing.assert_allclose(
        R1.Hxlk[1, 0],
        R2.Hxlk[1, 0],
        rtol=1e-8,
        atol=1e-8 * abs(R1.Hxlk[1, 0]).mean(),
    )
    np.testing.assert_allclose(
        R1.dHxlk[1, 0],
        R2.dHxlk[1, 0],
        rtol=1e-8,
        atol=1e-8 * abs(R1.dHxlk[1, 0]).mean(),
    )
    np.testing.assert_allclose(
        R1.ddGxlk[1, 0],
        R2.ddGxlk[1, 0],
        rtol=1e-8,
        atol=1e-8 * abs(R1.ddGxlk[1, 0]).mean(),
    )
    # b,b
    np.testing.assert_allclose(
        R1.Hxlk[1, 1],
        R2.Hxlk[1, 1],
        rtol=1e-8,
        atol=1e-8 * abs(R1.Hxlk[1, 1]).mean(),
    )
    np.testing.assert_allclose(
        R1.dHxlk[1, 1],
        R2.dHxlk[1, 1],
        rtol=1e-8,
        atol=1e-8 * abs(R1.dHxlk[1, 1]).mean(),
    )
    np.testing.assert_allclose(
        R1.ddGxlk[1, 1],
        R2.ddGxlk[1, 1],
        rtol=1e-4,
        atol=1e-8 * abs(R1.ddGxlk[1, 1]).mean(),
    )


@np.vectorize
def mplGammainc(s, x):
    f = mpmath.gammainc(s, 0, x)
    s = mpmath.sign(f)
    lf = mpmath.log(mpmath.fabs(f))
    return int(s), float(lf)


@np.vectorize
def mplGammaincc(s, x):
    f = mpmath.gammainc(s, x, mpmath.inf)
    s = mpmath.sign(f)
    lf = mpmath.log(mpmath.fabs(f))
    return int(s), float(lf)


def test_lower_Gamma():
    """Test for lower incomplete gamma."""
    l = np.arange(7)[:, None, None]
    k = np.arange(11)[None, :, None]
    x0 = np.logspace(-4, 3, 50)[None, None, :]
    s = l / 2 + k / 2 + 5 / 2  # for I_4

    s1, f1 = lGammainc(s, x0**2)
    s2, f2 = mplGammainc(s, x0**2)
    np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10)
    assert np.all(s1 == s2)


def test_upper_Gamma():
    """Test for upper incomplete gamma."""
    l = np.arange(7)[:, None, None]
    k = np.arange(11)[None, :, None]
    x0 = np.logspace(-4, 3, 50)[None, None, :]
    s = -l / 2 + k / 2 + 1  # for I_1

    s1, f1 = lGammaincc(s, x0**2)
    s2, f2 = mplGammaincc(s, x0**2)
    np.testing.assert_allclose(f1, f2, rtol=5e-8, atol=5e-8)
    assert np.all(s1 == s2)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_single_species_potentials_vs_sympy(l, potential_gamma):
    potentials = potential_gamma
    speedgrid = potentials.speedgrid
    pitchgrid = UniformPitchAngleGrid(41)

    v = sympy.symbols("v", real=True)
    va = sympy.symbols("v_a", real=True)
    x = v / va
    vta = potentials.species[0].v_thermal
    fa = (1 - x + x**2) * sympy.exp(-(x**2))

    Ha = _compute_H_sympy(fa, v, l, va)
    Ga = _compute_G_sympy(fa, v, l, va)

    ddGasympy = _eval_f(Ga.diff(v).diff(v), v, speedgrid.x * vta, {va: vta})
    Hasympy = _eval_f(Ha, v, speedgrid.x * vta, {va: vta})
    dHasympy = _eval_f(Ha.diff(v), v, speedgrid.x * vta, {va: vta})
    ffa = _eval_f(fa, v, speedgrid.x * vta, {va: vta})

    f = np.ones((1, speedgrid.nx, pitchgrid.nxi, 1, 1))
    f[0] *= (
        ffa[:, None, None, None]
        * orthax.orthval(
            pitchgrid.xi,
            jnp.zeros(potentials.legendregrid.nxi).at[l].set(1.0),
            potentials.legendregrid.xirec,
        )[None, :, None, None]
    )

    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nxi - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)

    Hajax = rosenbluth_H_jax(f[0], 0, 0, speedgrid, pitchgrid, potentials, Txi_inv)
    dHajax = rosenbluth_dH_jax(f[0], 0, 0, speedgrid, pitchgrid, potentials, Txi_inv)
    ddGajax = rosenbluth_ddG_jax(f[0], 0, 0, speedgrid, pitchgrid, potentials, Txi_inv)

    # potentials are diagonal in legendre index, so outputs for idx != l should be 0
    # scale for H ~ v^2, dH/dv ~ v, G ~ v^4, dG/dv^2 ~ v^2
    np.testing.assert_allclose(Hajax[:, :l, :, :], 0, atol=1e-12 * vta**2)
    np.testing.assert_allclose(Hajax[:, l + 1 :, :, :], 0, atol=1e-12 * vta**2)
    np.testing.assert_allclose(dHajax[:, :l, :, :], 0, atol=1e-12 * vta)
    np.testing.assert_allclose(dHajax[:, l + 1 :, :, :], 0, atol=1e-12 * vta)
    np.testing.assert_allclose(ddGajax[:, :l, :, :], 0, atol=1e-12 * vta**2)
    np.testing.assert_allclose(ddGajax[:, l + 1 :, :, :], 0, atol=1e-12 * vta**2)

    np.testing.assert_allclose(Hajax[:, l, 0, 0], Hasympy, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(dHajax[:, l, 0, 0], dHasympy, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(ddGajax[:, l, 0, 0], ddGasympy, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_sympy_potentials(l):
    v = sympy.symbols("v", real=True)
    va = sympy.symbols("v_a", real=True)
    x = v / va
    f = (1 - x + x**2) * sympy.exp(-(x**2))
    H = _compute_H_sympy(f, v, l, va)
    G = _compute_G_sympy(f, v, l, va)

    def _poisson_spherical_1d(p, r, f, l):
        return (r**2 * p.diff(r)).diff(r) - l * (l + 1) * p - r**2 * f

    assert sympy.simplify(_poisson_spherical_1d(H, v, -4 * sympy.pi * f, l)) == 0
    assert sympy.simplify(_poisson_spherical_1d(G, v, 2 * H, l)) == 0
