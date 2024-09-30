"""Tests for computing Rosenbluth potentials."""

import monkes
import numpy as np
import pytest

from yancc.collisions import RosenbluthPotentials
from yancc.velocity_grids import PitchAngleGrid, SpeedGrid


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
def potential1(xgrid, xigrid, species):
    """Single species potentials with quadrature."""
    return RosenbluthPotentials(
        xgrid,
        xigrid,
        species,
        quad=True,
    )


@pytest.fixture
def potential2(xgrid, xigrid, species):
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
def test_rosenbluth_derivatives_quad(potential1, x0, l, k):
    """Test for potentials using quadrature."""
    R = potential1
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
def test_rosenbluth_erivatives_gamma(potential2, x0, l, k):
    """Test for potentials using incomplete gamma functions."""
    R = potential2
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
