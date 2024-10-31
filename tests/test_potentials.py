"""Tests for computing Rosenbluth potentials."""

import monkes
import mpmath
import numpy as np
import pytest

from yancc.collisions import RosenbluthPotentials
from yancc.utils import Gammainc, Gammaincc
from yancc.velocity_grids import PitchAngleGrid, SpeedGrid

mpmath.mp.dps = 100


@pytest.fixture
def xgrid():
    """Speed grid for testing."""
    return SpeedGrid(5)


@pytest.fixture
def xigrid():
    """Pitch angle grid for testing"""
    return PitchAngleGrid(30)


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


@pytest.mark.parametrize("x0", np.linspace(0.1, 5, 3))
@pytest.mark.parametrize("l", [0, 10, 20])
@pytest.mark.parametrize("k", [0, 2, 5])
def test_rosenbluth_derivatives_quad(potential_quad, x0, l, k):
    """Test for potentials using quadrature."""
    R = potential_quad
    eps = 1e-3
    dHfd = (R._Hlk(x0 + eps, l, k) - R._Hlk(x0 - eps, l, k)) / (2 * eps)
    dHan = R._dHlk(x0, l, k)
    np.testing.assert_allclose(dHfd, dHan, rtol=1e-2)
    dGfd = (R._Glk(x0 + eps, l, k) - R._Glk(x0 - eps, l, k)) / (2 * eps)
    dGan = R._dGlk(x0, l, k)
    np.testing.assert_allclose(dGfd, dGan, rtol=1e-2)
    d2Gfd = (R._dGlk(x0 + eps, l, k) - R._dGlk(x0 - eps, l, k)) / (2 * eps)
    d2Gan = R._ddGlk(x0, l, k)
    np.testing.assert_allclose(d2Gfd, d2Gan, rtol=1e-2)


@pytest.mark.parametrize("x0", np.linspace(0.1, 5, 3))
@pytest.mark.parametrize("l", [0, 10, 20])
@pytest.mark.parametrize("k", [0, 2, 5])
def test_rosenbluth_derivatives_gamma(potential_gamma, x0, l, k):
    """Test for potentials using incomplete gamma functions."""
    R = potential_gamma
    eps = 1e-3
    dHfd = (R._Hlk(x0 + eps, l, k) - R._Hlk(x0 - eps, l, k)) / (2 * eps)
    dHan = R._dHlk(x0, l, k)
    np.testing.assert_allclose(dHfd, dHan, rtol=1e-2)
    dGfd = (R._Glk(x0 + eps, l, k) - R._Glk(x0 - eps, l, k)) / (2 * eps)
    dGan = R._dGlk(x0, l, k)
    np.testing.assert_allclose(dGfd, dGan, rtol=1e-2)
    d2Gfd = (R._dGlk(x0 + eps, l, k) - R._dGlk(x0 - eps, l, k)) / (2 * eps)
    d2Gan = R._ddGlk(x0, l, k)
    np.testing.assert_allclose(d2Gfd, d2Gan, rtol=1e-2)


@pytest.mark.parametrize("x0", np.linspace(0.1, 5, 3))
@pytest.mark.parametrize("l", [0, 10, 20])
@pytest.mark.parametrize("k", [0, 2, 5])
def test_rosenbluth_quad_vs_gamma(potential_quad, potential_gamma, x0, l, k):
    """Test for potentials using incomplete gamma functions."""
    Rq = potential_quad
    Rg = potential_gamma

    Hq = Rq._Hlk(x0, l, k)
    Hg = Rg._Hlk(x0, l, k)
    np.testing.assert_allclose(Hq, Hg)

    dHq = Rq._dHlk(x0, l, k)
    dHg = Rg._dHlk(x0, l, k)
    np.testing.assert_allclose(dHq, dHg)

    Gq = Rq._Glk(x0, l, k)
    Gg = Rg._Glk(x0, l, k)
    np.testing.assert_allclose(Gq, Gg)

    dGq = Rq._dGlk(x0, l, k)
    dGg = Rg._dGlk(x0, l, k)
    np.testing.assert_allclose(dGq, dGg)

    ddGq = Rq._ddGlk(x0, l, k)
    ddGg = Rg._ddGlk(x0, l, k)
    np.testing.assert_allclose(ddGq, ddGg)


@pytest.mark.parametrize("x0", np.linspace(0.1, 5, 3))
@pytest.mark.parametrize("l", [0, 10, 20])
@pytest.mark.parametrize("k", [0, 2, 5])
def test_lower_Gamma(x0, l, k):
    """Test for lower incomplete gamma."""
    s = l / 2 + k / 2 + 5 / 2  # for I_4
    mpGammainc = lambda s, x: float(mpmath.gammainc(s, 0, x))

    f1 = Gammainc(s, x0**2)
    f2 = mpGammainc(s, x0**2)
    np.testing.assert_allclose(f1, f2)


@pytest.mark.parametrize("x0", np.linspace(0.1, 5, 3))
@pytest.mark.parametrize("l", [0, 10, 20])
@pytest.mark.parametrize("k", [0, 2, 5])
def test_upper_Gamma(x0, l, k):
    """Test for upper incomplete gamma."""
    s = -l / 2 + k / 2 + 1  # for I_1
    mpGammaincc = lambda s, x: float(mpmath.gammainc(s, x, mpmath.inf))

    f1 = Gammaincc(s, x0**2)
    f2 = mpGammaincc(s, x0**2)
    np.testing.assert_allclose(f1, f2)
