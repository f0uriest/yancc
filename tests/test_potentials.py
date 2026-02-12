"""Tests for computing Rosenbluth potentials."""

import jax.numpy as jnp
import mpmath
import numpy as np
import orthax
import pytest
import sympy

from yancc.utils import lGammainc, lGammaincc
from yancc.velocity_grids import UniformPitchAngleGrid

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
    subs = {va: float(vta)}
    fa = (1 - x + 3 * x**2) * sympy.exp(-(x**2))

    Ha = _compute_H_sympy(fa, v, l, va)
    Ga = _compute_G_sympy(fa, v, l, va)

    ddGasympy = _eval_f(Ga.diff(v).diff(v), v, speedgrid.x * vta, subs)
    Hasympy = _eval_f(Ha, v, speedgrid.x * vta, subs)
    dHasympy = _eval_f(Ha.diff(v), v, speedgrid.x * vta, subs)
    ffa = _eval_f(fa, v, speedgrid.x * vta, subs)

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

    np.testing.assert_allclose(Hajax[:, l, 0, 0], Hasympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(dHajax[:, l, 0, 0], dHasympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(ddGajax[:, l, 0, 0], ddGasympy, rtol=1e-10, atol=0)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_2_species_potentials_vs_sympy(l, potential_gamma):
    potentials = potential_gamma
    speedgrid = potentials.speedgrid
    species = potentials.species
    pitchgrid = UniformPitchAngleGrid(41)

    v = sympy.symbols("v", real=True, positive=True)
    vta, vtb = sympy.symbols("v_a v_b", real=True, positive=True)
    va, vb = species[0].v_thermal, species[1].v_thermal
    subs = {
        vta: float(va),
        vtb: float(vb),
    }

    xa = v / vta
    xb = v / vtb

    fa = (1 - xa + 3 * xa**2) * sympy.exp(-(xa**2))
    fb = (4 + xb - 2 * xb**2) * sympy.exp(-(xb**2))
    Ha = _compute_H_sympy(fa, v, l, vta)
    Ga = _compute_G_sympy(fa, v, l, vta)
    Hb = _compute_H_sympy(fb, v, l, vtb)
    Gb = _compute_G_sympy(fb, v, l, vtb)

    Haa_sympy = _eval_f(Ha, v, speedgrid.x * va, subs)
    Hab_sympy = _eval_f(Hb, v, speedgrid.x * va, subs)
    Hba_sympy = _eval_f(Ha, v, speedgrid.x * vb, subs)
    Hbb_sympy = _eval_f(Hb, v, speedgrid.x * vb, subs)

    dHaa_sympy = _eval_f(Ha.diff(v), v, speedgrid.x * va, subs)
    dHab_sympy = _eval_f(Hb.diff(v), v, speedgrid.x * va, subs)
    dHba_sympy = _eval_f(Ha.diff(v), v, speedgrid.x * vb, subs)
    dHbb_sympy = _eval_f(Hb.diff(v), v, speedgrid.x * vb, subs)

    ddGaa_sympy = _eval_f(Ga.diff(v).diff(v), v, speedgrid.x * va, subs)
    ddGab_sympy = _eval_f(Gb.diff(v).diff(v), v, speedgrid.x * va, subs)
    ddGba_sympy = _eval_f(Ga.diff(v).diff(v), v, speedgrid.x * vb, subs)
    ddGbb_sympy = _eval_f(Gb.diff(v).diff(v), v, speedgrid.x * vb, subs)

    ffa = _eval_f(fa, v, speedgrid.x * va, subs)
    ffb = _eval_f(fb, v, speedgrid.x * vb, subs)

    f = np.ones((2, speedgrid.nx, pitchgrid.nxi, 1, 1))
    f[0] *= (
        ffa[:, None, None, None]
        * orthax.orthval(
            pitchgrid.xi,
            jnp.zeros(potentials.legendregrid.nxi).at[l].set(1.0),
            potentials.legendregrid.xirec,
        )[None, :, None, None]
    )
    f[1] *= (
        ffb[:, None, None, None]
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

    Haa_jax = rosenbluth_H_jax(f[0], 0, 0, speedgrid, pitchgrid, potentials, Txi_inv)
    dHaa_jax = rosenbluth_dH_jax(f[0], 0, 0, speedgrid, pitchgrid, potentials, Txi_inv)
    ddGaa_jax = rosenbluth_ddG_jax(
        f[0], 0, 0, speedgrid, pitchgrid, potentials, Txi_inv
    )
    Hab_jax = rosenbluth_H_jax(f[1], 0, 1, speedgrid, pitchgrid, potentials, Txi_inv)
    dHab_jax = rosenbluth_dH_jax(f[1], 0, 1, speedgrid, pitchgrid, potentials, Txi_inv)
    ddGab_jax = rosenbluth_ddG_jax(
        f[1], 0, 1, speedgrid, pitchgrid, potentials, Txi_inv
    )
    Hba_jax = rosenbluth_H_jax(f[0], 1, 0, speedgrid, pitchgrid, potentials, Txi_inv)
    dHba_jax = rosenbluth_dH_jax(f[0], 1, 0, speedgrid, pitchgrid, potentials, Txi_inv)
    ddGba_jax = rosenbluth_ddG_jax(
        f[0], 1, 0, speedgrid, pitchgrid, potentials, Txi_inv
    )
    Hbb_jax = rosenbluth_H_jax(f[1], 1, 1, speedgrid, pitchgrid, potentials, Txi_inv)
    dHbb_jax = rosenbluth_dH_jax(f[1], 1, 1, speedgrid, pitchgrid, potentials, Txi_inv)
    ddGbb_jax = rosenbluth_ddG_jax(
        f[1], 1, 1, speedgrid, pitchgrid, potentials, Txi_inv
    )

    # potentials are diagonal in legendre index, so outputs for idx != l should be 0
    # scale for H ~ v^2, dH/dv ~ v, G ~ v^4, dG/dv^2 ~ v^2
    # a,a
    np.testing.assert_allclose(Haa_jax[:, :l, :, :], 0, atol=1e-12 * va**2)
    np.testing.assert_allclose(Haa_jax[:, l + 1 :, :, :], 0, atol=1e-12 * va**2)
    np.testing.assert_allclose(dHaa_jax[:, :l, :, :], 0, atol=1e-12 * va)
    np.testing.assert_allclose(dHaa_jax[:, l + 1 :, :, :], 0, atol=1e-12 * va)
    np.testing.assert_allclose(ddGaa_jax[:, :l, :, :], 0, atol=1e-12 * va**2)
    np.testing.assert_allclose(ddGaa_jax[:, l + 1 :, :, :], 0, atol=1e-12 * va**2)
    # a,b
    np.testing.assert_allclose(Hab_jax[:, :l, :, :], 0, atol=1e-12 * vb**2)
    np.testing.assert_allclose(Hab_jax[:, l + 1 :, :, :], 0, atol=1e-12 * vb**2)
    np.testing.assert_allclose(dHab_jax[:, :l, :, :], 0, atol=1e-12 * vb)
    np.testing.assert_allclose(dHab_jax[:, l + 1 :, :, :], 0, atol=1e-12 * vb)
    np.testing.assert_allclose(ddGab_jax[:, :l, :, :], 0, atol=1e-12 * vb**2)
    np.testing.assert_allclose(ddGab_jax[:, l + 1 :, :, :], 0, atol=1e-12 * vb**2)
    # b,a
    np.testing.assert_allclose(Hba_jax[:, :l, :, :], 0, atol=1e-12 * va**2)
    np.testing.assert_allclose(Hba_jax[:, l + 1 :, :, :], 0, atol=1e-12 * va**2)
    np.testing.assert_allclose(dHba_jax[:, :l, :, :], 0, atol=1e-12 * va)
    np.testing.assert_allclose(dHba_jax[:, l + 1 :, :, :], 0, atol=1e-12 * va)
    np.testing.assert_allclose(ddGba_jax[:, :l, :, :], 0, atol=1e-12 * va**2)
    np.testing.assert_allclose(ddGba_jax[:, l + 1 :, :, :], 0, atol=1e-12 * va**2)
    # b,b
    np.testing.assert_allclose(Hbb_jax[:, :l, :, :], 0, atol=1e-12 * vb**2)
    np.testing.assert_allclose(Hbb_jax[:, l + 1 :, :, :], 0, atol=1e-12 * vb**2)
    np.testing.assert_allclose(dHbb_jax[:, :l, :, :], 0, atol=1e-12 * vb)
    np.testing.assert_allclose(dHbb_jax[:, l + 1 :, :, :], 0, atol=1e-12 * vb)
    np.testing.assert_allclose(ddGbb_jax[:, :l, :, :], 0, atol=1e-12 * vb**2)
    np.testing.assert_allclose(ddGbb_jax[:, l + 1 :, :, :], 0, atol=1e-12 * vb**2)

    # a,a
    np.testing.assert_allclose(Haa_jax[:, l, 0, 0], Haa_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(dHaa_jax[:, l, 0, 0], dHaa_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(ddGaa_jax[:, l, 0, 0], ddGaa_sympy, rtol=1e-10, atol=0)
    # a,b
    np.testing.assert_allclose(Hab_jax[:, l, 0, 0], Hab_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(dHab_jax[:, l, 0, 0], dHab_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(ddGab_jax[:, l, 0, 0], ddGab_sympy, rtol=1e-10, atol=0)
    # b,a
    np.testing.assert_allclose(Hba_jax[:, l, 0, 0], Hba_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(dHba_jax[:, l, 0, 0], dHba_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(ddGba_jax[:, l, 0, 0], ddGba_sympy, rtol=1e-10, atol=0)
    # b,b
    np.testing.assert_allclose(Hbb_jax[:, l, 0, 0], Hbb_sympy, rtol=1e-8, atol=0)
    np.testing.assert_allclose(dHbb_jax[:, l, 0, 0], dHbb_sympy, rtol=1e-8, atol=0)
    np.testing.assert_allclose(ddGbb_jax[:, l, 0, 0], ddGbb_sympy, rtol=1e-8, atol=0)


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
