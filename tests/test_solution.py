"""Tests for solution containers."""

import jax.numpy as jnp
import pytest

from yancc.solution import DKESolution, clean_units
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def test_clean_units_empty_and_none():
    """clean_units returns the empty string for empty / "None" input."""
    assert clean_units("") == ""
    assert clean_units("None") == ""


def test_clean_units_renders_latex_to_unicode():
    r"""Render a LaTeX units string to unicode (superscripts, \cdot -> ·)."""
    assert (
        clean_units("kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}")
        == "kg·m⁻¹·s⁻³ = W·m⁻³"
    )


def test_dkesolution_wrong_f1_size(dummy_field, species1):
    """DKE solution rejects an f1 that is neither N nor N + 2*ns."""
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(3)
    ns = len(species1)
    N = ns * speedgrid.nx * pitchgrid.na * dummy_field.ntheta * dummy_field.nzeta

    with pytest.raises(ValueError, match="wrong size for f1"):
        DKESolution(
            F0=jnp.zeros((ns, speedgrid.nx)),
            f1=jnp.zeros(N + 1),  # neither N nor N + 2*ns
            rhs=jnp.zeros(N),
            field=dummy_field,
            pitchgrid=pitchgrid,
            speedgrid=speedgrid,
            species=species1,
            Erho=jnp.array(0.0),
            EparB=jnp.array(0.0),
            background=[],
        )
