"""Tests for computing Rosenbluth potentials."""

import jax.numpy as jnp
import mpmath
import numpy as np
import orthax
import pytest
import sympy

from yancc.collisions import RosenbluthPotentials
from yancc.species import GlobalMaxwellian, Hydrogen
from yancc.utils import Gammainc, Gammaincc
from yancc.velocity_grids import SpeedGrid, UniformPitchAngleGrid

from .conftest import _compute_G_sympy, _compute_H_sympy

mpmath.mp.dps = 100


@pytest.fixture
def xgrid():
    """Speed grid for testing."""
    return SpeedGrid(10)


@pytest.fixture
def xigrid():
    """Pitch angle grid for testing"""
    return UniformPitchAngleGrid(17)


@pytest.fixture
def species():
    """Single ion species for testing."""
    return [
        GlobalMaxwellian(
            Hydrogen, lambda x: 1e3 * (1 - x**2), lambda x: 1e19 * (1 - x**4)
        ).localize(0.5)
    ]


@pytest.fixture
def potential_quad(xgrid, xigrid, species):
    """Single species potentials with quadrature."""
    return RosenbluthPotentials(
        xgrid,
        xigrid,
        species,
        nL=6,
        quad=True,
    )


@pytest.fixture
def potential_gamma(xgrid, xigrid, species):
    """Single species potential without quadrature."""
    return RosenbluthPotentials(
        xgrid,
        xigrid,
        species,
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
    np.testing.assert_allclose(R1.Hxlk, R2.Hxlk, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(R1.dHxlk, R2.dHxlk, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(R1.ddGxlk, R2.ddGxlk, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("x0", np.linspace(0.1, 4, 3))
@pytest.mark.parametrize("l", [0, 2, 4])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_lower_Gamma(x0, l, k):
    """Test for lower incomplete gamma."""
    s = l / 2 + k / 2 + 5 / 2  # for I_4
    mpGammainc = lambda s, x: float(mpmath.gammainc(s, 0, x))

    f1 = Gammainc(s, x0**2)
    f2 = mpGammainc(s, x0**2)
    np.testing.assert_allclose(f1, f2, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("x0", np.linspace(0.1, 4, 3))
@pytest.mark.parametrize("l", [0, 2, 4, 6])
@pytest.mark.parametrize("k", [0, 2, 4, 8])
def test_upper_Gamma(x0, l, k):
    """Test for upper incomplete gamma."""
    s = -l / 2 + k / 2 + 1  # for I_1
    mpGammaincc = lambda s, x: float(mpmath.gammainc(s, x, mpmath.inf))

    f1 = Gammaincc(s, x0**2)
    f2 = mpGammaincc(s, x0**2)
    np.testing.assert_allclose(f1, f2, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_single_species_potentials_vs_sympy(l, potential_gamma):
    potentials = potential_gamma
    speedgrid = potentials.speedgrid
    pitchgrid = potentials.pitchgrid

    x = sympy.symbols("x", real=True)
    va = sympy.symbols("v_a", real=True)

    fa = (1 - x + x**2) * sympy.exp(-(x**2))

    Ha = _compute_H_sympy(fa, x, l, va)
    Ga = _compute_G_sympy(fa, x, l, va)

    ddGasympy = np.array(
        [(Ga / va**4).diff(x).diff(x).evalf(subs={x: xi_}) for xi_ in speedgrid.x],
        dtype=np.float64,
    )
    Hasympy = np.array(
        [(Ha / va**2).evalf(subs={x: xi_}) for xi_ in speedgrid.x], dtype=np.float64
    )
    dHasympy = np.array(
        [(Ha / va**2).diff(x).evalf(subs={x: xi_}) for xi_ in speedgrid.x],
        dtype=np.float64,
    )
    ffa = np.array([fa.evalf(subs={x: xi_}) for xi_ in speedgrid.x], dtype=np.float64)
    f = np.ones((1, speedgrid.nx, pitchgrid.nxi, 1, 1))
    f[0] *= (
        ffa[:, None, None, None]
        * orthax.orthval(
            pitchgrid.xi,
            jnp.zeros(potentials.legendregrid.nxi).at[l].set(1.0),
            potentials.legendregrid.xirec,
        )[None, :, None, None]
    )

    def rosenbluth_ddG_jax(f, a, b):
        assert f.shape == (speedgrid.nx, pitchgrid.nxi, 1, 1)
        # convert nodal alpha -> legendre l
        f = jnp.einsum("la,xatz->xltz", potentials.Txi_inv, f)
        # convert nodal x -> speed k
        f = jnp.einsum("kx,xltz->kltz", speedgrid.xvander_inv, f)
        Gabxlk = potentials.ddGxlk[a, b, :, : potentials.legendregrid.nxi]
        df = jnp.einsum("xlk,kltz->xltz", Gabxlk, f)
        return df

    def rosenbluth_dH_jax(f, a, b):
        assert f.shape == (speedgrid.nx, pitchgrid.nxi, 1, 1)
        # convert nodal alpha -> legendre l
        f = jnp.einsum("la,xatz->xltz", potentials.Txi_inv, f)
        # convert nodal x -> speed k
        f = jnp.einsum("kx,xltz->kltz", speedgrid.xvander_inv, f)
        Gabxlk = potentials.dHxlk[a, b, :, : potentials.legendregrid.nxi]
        df = jnp.einsum("xlk,kltz->xltz", Gabxlk, f)
        return df

    def rosenbluth_H_jax(f, a, b):
        assert f.shape == (speedgrid.nx, pitchgrid.nxi, 1, 1)
        # convert nodal alpha -> legendre l
        f = jnp.einsum("la,xatz->xltz", potentials.Txi_inv, f)
        # convert nodal x -> speed k
        f = jnp.einsum("kx,xltz->kltz", speedgrid.xvander_inv, f)
        Gabxlk = potentials.Hxlk[a, b, :, : potentials.legendregrid.nxi]
        df = jnp.einsum("xlk,kltz->xltz", Gabxlk, f)
        return df

    Hajax = rosenbluth_H_jax(f[0], 0, 0)
    dHajax = rosenbluth_dH_jax(f[0], 0, 0)
    ddGajax = rosenbluth_ddG_jax(f[0], 0, 0)

    # potentials are diagonal in legendre index, so outputs for idx != l should be 0
    np.testing.assert_allclose(Hajax[:, :l, :, :], 0, atol=1e-12)
    np.testing.assert_allclose(Hajax[:, l + 1 :, :, :], 0, atol=1e-12)
    np.testing.assert_allclose(dHajax[:, :l, :, :], 0, atol=1e-12)
    np.testing.assert_allclose(dHajax[:, l + 1 :, :, :], 0, atol=1e-12)
    np.testing.assert_allclose(ddGajax[:, :l, :, :], 0, atol=1e-12)
    np.testing.assert_allclose(ddGajax[:, l + 1 :, :, :], 0, atol=1e-12)

    np.testing.assert_allclose(Hajax[:, l, 0, 0], Hasympy, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(dHajax[:, l, 0, 0], dHasympy, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(ddGajax[:, l, 0, 0], ddGasympy, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_sympy_potentials(l):
    x = sympy.symbols("x", real=True)
    va = sympy.symbols("v_a", real=True)
    f = (1 - x + x**2) * sympy.exp(-(x**2))
    H = _compute_H_sympy(f, x, l, va)
    G = _compute_G_sympy(f, x, l, va)

    assert (
        sympy.simplify(
            (x**2 * H.diff(x)).diff(x)
            - l * (l + 1) * H
            + 4 * sympy.pi * va**2 * x**2 * f
        )
        == 0
    )
    assert (
        sympy.simplify(
            (x**2 * G.diff(x)).diff(x) - l * (l + 1) * G - 2 * va**2 * x**2 * H
        )
        == 0
    )
