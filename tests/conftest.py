"""Fixtures etc for testing."""

import desc
import pytest

from yancc.field import Field
from yancc.velocity_grids import LegendrePitchAngleGrid


@pytest.fixture(scope="session")
def field():
    """Field for testing."""
    eq = desc.examples.get("W7-X")
    field = Field.from_desc(eq, 0.5, 15, 15)
    return field


@pytest.fixture(scope="session")
def pitchgrid():
    """Pitch angle grid for testing"""
    return LegendrePitchAngleGrid(15)
