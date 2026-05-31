"""Tests for solution containers."""

import jax.numpy as jnp
import numpy as np
import pytest

from yancc.solution import DKESolution, MDKESolution, clean_units
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
    N = ns * speedgrid.nx * pitchgrid.nalpha * dummy_field.ntheta * dummy_field.nzeta

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


def _make_dke_solution(dummy_field, species, f1):
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(3)
    ns = len(species)
    N = ns * speedgrid.nx * pitchgrid.nalpha * dummy_field.ntheta * dummy_field.nzeta
    return (
        DKESolution(
            F0=jnp.zeros((ns, speedgrid.nx)),
            f1=f1,
            rhs=jnp.zeros(N),
            field=dummy_field,
            pitchgrid=pitchgrid,
            speedgrid=speedgrid,
            species=species,
            Erho=jnp.array(0.0),
            EparB=jnp.array(0.0),
            background=[],
        ),
        N,
        ns,
    )


def test_dkesolution_f1_size_N_has_nan_sources(dummy_field, species1):
    """An f1 of exactly size N carries no solvability sources (NaN placeholders)."""
    ns = len(species1)
    N = ns * 3 * 5 * dummy_field.ntheta * dummy_field.nzeta
    sol, _, _ = _make_dke_solution(dummy_field, species1, jnp.zeros(N))
    assert np.all(np.isnan(np.asarray(sol.get("particle_source"))))
    assert np.all(np.isnan(np.asarray(sol.get("heat_source"))))


def test_dkesolution_f1_with_sources_roundtrips(dummy_field, species1):
    """f1 of size N + 2*ns splits off the particle/heat solvability sources."""
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(3)
    ns = len(species1)
    N = ns * speedgrid.nx * pitchgrid.nalpha * dummy_field.ntheta * dummy_field.nzeta
    particle = jnp.arange(ns, dtype=float) + 1.0
    heat = jnp.arange(ns, dtype=float) + 10.0
    f1 = jnp.concatenate([jnp.zeros(N), particle, heat])

    sol, _, _ = _make_dke_solution(dummy_field, species1, f1)
    np.testing.assert_allclose(np.asarray(sol.get("particle_source")), particle)
    np.testing.assert_allclose(np.asarray(sol.get("heat_source")), heat)


def test_dkesolution_qtys_list(dummy_field, species1):
    """qtys_list returns the registered DKE output quantities."""
    ns = len(species1)
    N = ns * 3 * 5 * dummy_field.ntheta * dummy_field.nzeta
    sol, _, _ = _make_dke_solution(dummy_field, species1, jnp.zeros(N))
    qtys = sol.qtys_list()
    assert isinstance(qtys, list)
    assert "particle_source" in qtys
    assert "heat_source" in qtys
    assert "<particle_flux>" in qtys
    assert "<heat_flux>" in qtys
    assert "<V||B>" in qtys


def test_mdkesolution_qtys_list(dummy_field):
    """MDKESolution.qtys_list returns the registered MDKE output quantities."""
    pitchgrid = UniformPitchAngleGrid(5)
    n = 3 * pitchgrid.nalpha * dummy_field.ntheta * dummy_field.nzeta
    sol = MDKESolution(
        f=jnp.zeros(n),
        rhs=jnp.zeros(n),
        field=dummy_field,
        pitchgrid=pitchgrid,
        nuhat=jnp.array(0.1),
        erhohat=jnp.array(0.01),
    )
    qtys = sol.qtys_list()
    assert isinstance(qtys, list)
    assert len(qtys) > 0
    assert "Dij" in qtys
    assert "Dij_DKES" in qtys
