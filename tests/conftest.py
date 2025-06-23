"""Fixtures etc for testing."""

import desc
import pytest

from yancc.field import Field
from yancc.species import Electron, GlobalMaxwellian, Hydrogen
from yancc.velocity_grids import LegendrePitchAngleGrid, SpeedGrid


@pytest.fixture(scope="session")
def field():
    """Field for testing."""
    eq = desc.examples.get("W7-X")
    field = Field.from_desc(eq, 0.5, 7, 9)
    return field


@pytest.fixture(scope="session")
def pitchgrid():
    """Pitch angle grid for testing"""
    return LegendrePitchAngleGrid(11)


@pytest.fixture(scope="session")
def speedgrid():
    """Pitch angle grid for testing"""
    return SpeedGrid(5)


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
