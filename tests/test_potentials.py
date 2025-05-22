"""Tests for computing Rosenbluth potentials."""

import monkes
import mpmath
import numpy as np
import pytest

from yancc.collisions import RosenbluthPotentials
from yancc.utils import Gammainc, Gammaincc
from yancc.velocity_grids import LegendrePitchAngleGrid, SpeedGrid

mpmath.mp.dps = 100


@pytest.fixture
def xgrid():
    """Speed grid for testing."""
    return SpeedGrid(5)


@pytest.fixture
def xigrid():
    """Pitch angle grid for testing"""
    return LegendrePitchAngleGrid(30)


@pytest.fixture
def species():
    """Single ion species for testing."""
    return [
        monkes.GlobalMaxwellian(
            monkes.Hydrogen, lambda x: 1e3 * (1 - x**2), lambda x: 1e19 * (1 - x**4)
        ).localize(0.5)
    ]


@pytest.fixture
def potential_quad(xgrid, xigrid, species):
    """Single species potentials with quadrature."""
    return RosenbluthPotentials(
        xgrid,
        xigrid,
        species,
        quad=True,
    )


@pytest.fixture
def potential_gamma(xgrid, xigrid, species):
    """Single species potential without quadrature."""
    return RosenbluthPotentials(
        xgrid,
        xigrid,
        species,
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
    np.testing.assert_allclose(R1.Hxlk, R2.Hxlk, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(R1.dHxlk, R2.dHxlk, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(R1.ddGxlk, R2.ddGxlk, rtol=1e-10, atol=1e-10)


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
