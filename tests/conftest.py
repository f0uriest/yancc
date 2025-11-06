"""Fixtures etc for testing."""

import desc
import pytest
import sympy

from yancc.collisions import RosenbluthPotentials
from yancc.field import Field
from yancc.species import Electron, GlobalMaxwellian, Hydrogen
from yancc.velocity_grids import LegendrePitchAngleGrid, MaxwellSpeedGrid


@pytest.fixture(scope="session")
def field():
    """Field for testing."""
    eq = desc.examples.get("W7-X")
    field = Field.from_desc(eq, 0.5, 5, 7)
    return field


@pytest.fixture(scope="session")
def pitchgrid():
    """Pitch angle grid for testing"""
    return LegendrePitchAngleGrid(9)


@pytest.fixture(scope="session")
def speedgrid():
    """Pitch angle grid for testing"""
    return MaxwellSpeedGrid(3)


@pytest.fixture(scope="session")
def species1():
    return [
        GlobalMaxwellian(
            Hydrogen,
            lambda x: 6.60e4 * (1 - x**2),
            lambda x: 5e19 * (1 - x**4),
        ).localize(0.5),
    ]


@pytest.fixture(scope="session")
def species2():
    return [
        GlobalMaxwellian(
            Hydrogen,
            lambda x: 6.60e4 * (1 - x**2),
            lambda x: 5e19 * (1 - x**4),
        ).localize(0.5),
        GlobalMaxwellian(
            Electron,
            lambda x: 2.00e4 * (1 - x**2),
            lambda x: 5e19 * (1 - x**4),
        ).localize(0.5),
    ]


@pytest.fixture(scope="session")
def potentials1(speedgrid, species1):
    return RosenbluthPotentials(speedgrid, species1)


@pytest.fixture(scope="session")
def potentials2(speedgrid, species2):
    return RosenbluthPotentials(speedgrid, species2)


def _compute_H_sympy(f, x, l, v):
    z = sympy.symbols("z", real=True)
    integrand1 = z ** (-l + 1) * f.subs(x, z)
    integrand2 = z ** (l + 2) * f.subs(x, z)
    integral1 = sympy.integrate(integrand1, (z, x, sympy.oo))
    integral2 = sympy.integrate(integrand2, (z, 0, x))
    Hterm1 = 1 / x ** (l + 1) * integral2
    Hterm2 = x**l * integral1
    H = v**2 * 4 * sympy.pi / (2 * l + 1) * (Hterm1 + Hterm2)
    return H


def _compute_G_sympy(f, x, l, v):
    z = sympy.symbols("z", real=True)
    integrand1 = z ** (-l + 1) * f.subs(x, z)
    integrand2 = z ** (l + 2) * f.subs(x, z)
    integrand3 = z ** (-l + 3) * f.subs(x, z)
    integrand4 = z ** (l + 4) * f.subs(x, z)
    integral1 = sympy.integrate(integrand1, (z, x, sympy.oo))
    integral2 = sympy.integrate(integrand2, (z, 0, x))
    integral3 = sympy.integrate(integrand3, (z, x, sympy.oo))
    integral4 = sympy.integrate(integrand4, (z, 0, x))
    Gterm1 = x**l * integral3
    Gterm2 = -sympy.Rational(2 * l - 1, 2 * l + 3) * x ** (l + 2) * integral1
    Gterm3 = -sympy.Rational(2 * l - 1, 2 * l + 3) / x ** (l + 1) * integral4
    Gterm4 = 1 / x ** (l - 1) * integral2
    G = -(4 * sympy.pi * v**4) / (4 * l**2 - 1) * (Gterm1 + Gterm2 + Gterm3 + Gterm4)
    return G


def _chan(x):
    return (sympy.erf(x) - 2 * x / sympy.sqrt(sympy.pi) * sympy.exp(-(x**2))) / (
        2 * x**2
    )


def _compute_CE_sympy(Fb, fa, x, v, va, vb, ma, mb, nb, Gamma_ab):
    nuD = Gamma_ab * nb / v**3 * (sympy.erf(v / vb) - _chan(v / vb))
    nupar = 2 * Gamma_ab * nb / v**3 * _chan(v / vb)
    CEterm1 = nupar * v**2 / 2 * (fa.subs(x, v / va).diff(v).diff(v))
    CEterm2 = -nupar * v**2 / vb**2 * (1 - ma / mb) * v * (fa.subs(x, v / va).diff(v))
    CEterm3 = nuD * v * (fa.subs(x, v / va).diff(v))
    CEterm4 = (
        4
        * sympy.pi
        * Gamma_ab
        * (ma / mb)
        * (Fb.subs(x, v / vb))
        * (fa.subs(x, v / va))
    )
    CE = CEterm1 + CEterm2 + CEterm3 + CEterm4
    return CE


def _compute_CD_sympy(Fa, fb, x, v, va, vb, ma, mb, Gamma_ab):
    CD = (4 * sympy.pi * ma / mb * fb).subs(x, v / vb)
    return Gamma_ab * Fa.subs(x, v / va) * CD


def _compute_CH_sympy(Fa, fb, x, l, v, va, vb, ma, mb, Gamma_ab):
    Hb = _compute_H_sympy(fb, x, l, vb)
    CH = -2 * v / va**2 * (1 - ma / mb) * Hb.diff(x).subs(
        x, v / vb
    ) / vb - 2 / va**2 * Hb.subs(x, v / vb)
    return Gamma_ab * Fa.subs(x, v / va) * CH


def _compute_CG_sympy(Fa, fb, x, l, v, va, vb, ma, mb, Gamma_ab):
    Gb = _compute_G_sympy(fb, x, l, vb)
    CG = 2 * v**2 / va**4 * Gb.diff(x).diff(x).subs(x, v / vb) / vb**2
    return Gamma_ab * Fa.subs(x, v / va) * CG


def _compute_CF_sympy(Fa, fb, x, l, v, va, vb, ma, mb, Gamma_ab):
    CH = _compute_CH_sympy(Fa, fb, x, l, v, va, vb, ma, mb, Gamma_ab)
    CG = _compute_CG_sympy(Fa, fb, x, l, v, va, vb, ma, mb, Gamma_ab)
    CD = _compute_CD_sympy(Fa, fb, x, v, va, vb, ma, mb, Gamma_ab)
    CF = CG + CH + CD
    return CF
