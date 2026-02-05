"""Fixtures etc for testing."""

import desc
import numpy as np
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


def _compute_H_sympy(f, v, l, vt):
    z = sympy.symbols("z", real=True)
    integrand1 = z ** (-l + 1) * f.subs(v, z * vt)
    integrand2 = z ** (l + 2) * f.subs(v, z * vt)
    integral1 = sympy.integrate(integrand1, (z, v / vt, sympy.oo))
    integral2 = sympy.integrate(integrand2, (z, 0, v / vt))
    Hterm1 = 1 / (v / vt) ** (l + 1) * integral2
    Hterm2 = (v / vt) ** l * integral1
    H = vt**2 * 4 * sympy.pi / (2 * l + 1) * (Hterm1 + Hterm2)
    return H


def _compute_G_sympy(f, v, l, vt):
    z = sympy.symbols("z", real=True)
    integrand1 = z ** (-l + 1) * f.subs(v, z * vt)
    integrand2 = z ** (l + 2) * f.subs(v, z * vt)
    integrand3 = z ** (-l + 3) * f.subs(v, z * vt)
    integrand4 = z ** (l + 4) * f.subs(v, z * vt)
    integral1 = sympy.integrate(integrand1, (z, v / vt, sympy.oo))
    integral2 = sympy.integrate(integrand2, (z, 0, v / vt))
    integral3 = sympy.integrate(integrand3, (z, v / vt, sympy.oo))
    integral4 = sympy.integrate(integrand4, (z, 0, v / vt))
    Gterm1 = (v / vt) ** l * integral3
    Gterm2 = -sympy.Rational(2 * l - 1, 2 * l + 3) * (v / vt) ** (l + 2) * integral1
    Gterm3 = -sympy.Rational(2 * l - 1, 2 * l + 3) / (v / vt) ** (l + 1) * integral4
    Gterm4 = 1 / (v / vt) ** (l - 1) * integral2
    G = -(4 * sympy.pi * vt**4) / (4 * l**2 - 1) * (Gterm1 + Gterm2 + Gterm3 + Gterm4)
    return G


def _compute_CDab_sympy(Fa, fb, ma, mb, Gamma_ab):
    CD = 4 * sympy.pi * ma / mb * fb
    return Gamma_ab * Fa * CD


def _compute_CHab_sympy(Fa, fb, l, v, vta, vtb, ma, mb, Gamma_ab):
    Hb = _compute_H_sympy(fb, v, l, vtb)
    CH = -2 * v / vta**2 * (1 - ma / mb) * Hb.diff(v) - 2 / vta**2 * Hb
    return Gamma_ab * Fa * CH


def _compute_CGab_sympy(Fa, fb, l, v, vta, vtb, Gamma_ab):
    Gb = _compute_G_sympy(fb, v, l, vtb)
    CG = 2 * v**2 / vta**4 * Gb.diff(v).diff(v)
    return Gamma_ab * Fa * CG


def _compute_CFab_sympy(Fa, fb, l, v, vta, vtb, ma, mb, Gamma_ab):
    CH = _compute_CHab_sympy(Fa, fb, l, v, vta, vtb, ma, mb, Gamma_ab)
    CG = _compute_CGab_sympy(Fa, fb, l, v, vta, vtb, Gamma_ab)
    CD = _compute_CDab_sympy(Fa, fb, ma, mb, Gamma_ab)
    CF = CG + CH + CD
    return CF


def _compute_CEab_sympy(Fb, fa, v, vta, vtb, ma, mb, nb, Gamma_ab):
    nuD = Gamma_ab * nb / v**3 * (sympy.erf(v / vtb) - _chandrasekhar_sympy(v / vtb))
    nupar = 2 * Gamma_ab * nb / v**3 * _chandrasekhar_sympy(v / vtb)
    CEterm1 = nupar * v**2 / 2 * (fa.diff(v).diff(v))
    CEterm2 = -nupar * v**2 / vtb**2 * (1 - ma / mb) * v * (fa.diff(v))
    CEterm3 = nuD * v * (fa.diff(v))
    CEterm4 = 4 * sympy.pi * Gamma_ab * (ma / mb) * Fb * fa
    CE = CEterm1 + CEterm2 + CEterm3 + CEterm4
    return CE


def _compute_CLab_sympy(fa, l, v, vb, nb, Gamma_ab):
    nu_D = Gamma_ab * nb / v**3 * (sympy.erf(v / vb) - _chandrasekhar_sympy(v / vb))
    df = -l * (l + 1) / 2 * fa
    return nu_D * df


def _compute_Cab_sympy(Fa, Fb, fa, fb, l, v, vta, vtb, ma, mb, na, nb, Gamma_ab):
    CFab = _compute_CFab_sympy(Fa, fb, l, v, vta, vtb, ma, mb, Gamma_ab)
    CEab = _compute_CEab_sympy(Fb, fa, v, vta, vtb, ma, mb, nb, Gamma_ab)
    CLab = _compute_CLab_sympy(fa, l, v, vtb, nb, Gamma_ab)
    return CFab + CEab + CLab


def _coulomb_logarithm_sympy(ma, mb, Ta, Tb, na, nb, qa, qb, epsilon_0):
    bmin, bmax = _impact_parameter_sympy(ma, mb, Ta, Tb, na, nb, qa, qb, epsilon_0)
    return sympy.log(bmax / bmin)


def _impact_parameter_sympy(ma, mb, Ta, Tb, na, nb, qa, qb, epsilon_0):
    vta = sympy.sqrt(2 * Ta / ma)
    vtb = sympy.sqrt(2 * Tb / mb)
    bmin = _impact_parameter_perp_sympy(ma, mb, vta, vtb, qa, qb, epsilon_0)
    bmax = _debye_length_sympy(ma, mb, Ta, Tb, na, nb, qa, qb, epsilon_0)
    return bmin, bmax


def _impact_parameter_perp_sympy(ma, mb, vta, vtb, qa, qb, epsilon_0):
    m_reduced = ma * mb / (ma + mb)
    return sympy.Abs(qa * qb) / (
        4 * sympy.pi * epsilon_0 * m_reduced * (vta**2 + vtb**2)
    )


def _debye_length_sympy(ma, mb, Ta, Tb, na, nb, qa, qb, epsilon_0):
    den = na / Ta * qa**2 + nb / Tb * qb**2
    return sympy.sqrt(epsilon_0 / den)


def _chandrasekhar_sympy(x):
    return (sympy.erf(x) - 2 * x / sympy.sqrt(sympy.pi) * sympy.exp(-(x**2))) / (
        2 * x**2
    )


def _gamma_ab_sympy(ma, mb, Ta, Tb, na, nb, qa, qb, epsilon_0):
    lnlambda = _coulomb_logarithm_sympy(ma, mb, Ta, Tb, na, nb, qa, qb, epsilon_0)
    return qa**2 * qb**2 * lnlambda / (4 * sympy.pi * epsilon_0**2 * ma**2)


def _eval_f(f, v, vi, subs):
    f = np.array([f.evalf(subs={v: float(v_), **subs}, n=30, maxn=200) for v_ in vi])
    f = f.astype(np.complex128)
    if (abs(f.imag) > 1e-16).any():
        print(f.imag)
    return f.real
