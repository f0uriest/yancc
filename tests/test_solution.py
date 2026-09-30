"""Tests for solution containers."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import elementary_charge, epsilon_0

from yancc._misc import DKEConstraint, DKESources
from yancc.field import Field
from yancc.solution import DKESolution, MDKESolution, _clean_units
from yancc.species import (
    _JOULE_PER_EV,
    Electron,
    Hydrogen,
    LocalMaxwellian,
    Species,
)
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def test_clean_units_empty_and_none():
    """clean_units returns the empty string for empty / "None" input."""
    assert _clean_units("") == ""
    assert _clean_units("None") == ""


def test_clean_units_renders_latex_to_unicode():
    r"""Render a LaTeX units string to unicode (superscripts, \cdot -> ·)."""
    assert (
        _clean_units("kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}")
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


def test_dkesolution_f1_with_sources_roundtrips(dummy_field, species2):
    """f1 of size N + 2*ns splits off per-species particle/heat solvability sources.

    The source unknowns are ordered like the DKESources columns: one (particle,
    heat) pair per species. The particle column carries density but no energy and
    the heat column energy but no density, as measured by DKEConstraint.
    """
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(3)
    ns = len(species2)
    N = ns * speedgrid.nx * pitchgrid.nalpha * dummy_field.ntheta * dummy_field.nzeta
    particle = jnp.arange(ns, dtype=float) + 1.0
    heat = jnp.arange(ns, dtype=float) + 10.0
    sources = jnp.stack([particle, heat], axis=1).flatten()
    f1 = jnp.concatenate([jnp.arange(N, dtype=float), sources])

    sol, _, _ = _make_dke_solution(dummy_field, species2, f1)
    np.testing.assert_allclose(np.asarray(sol.get("particle_source")), particle)
    np.testing.assert_allclose(np.asarray(sol.get("heat_source")), heat)
    np.testing.assert_allclose(np.asarray(sol.f1_krylov), np.asarray(f1))

    # nx=5 makes the Maxwell quadrature exact for the density/energy moments
    speedgrid = MaxwellSpeedGrid(5)
    B = DKESources(dummy_field, pitchgrid, speedgrid, species2).as_matrix()
    C = DKEConstraint(dummy_field, pitchgrid, speedgrid, species2).as_matrix()
    CB = np.asarray(C @ B)  # rows: (density, energy) per species
    for i in range(ns):
        particle_col, heat_col = 2 * i, 2 * i + 1
        density_row, energy_row = 2 * i, 2 * i + 1
        scale = np.abs(CB[density_row]).max() + np.abs(CB[energy_row]).max()
        np.testing.assert_allclose(CB[energy_row, particle_col] / scale, 0, atol=1e-12)
        np.testing.assert_allclose(CB[density_row, heat_col] / scale, 0, atol=1e-12)
        assert abs(CB[density_row, particle_col]) > 1e-3 * scale
        assert abs(CB[energy_row, heat_col]) > 1e-3 * scale


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


def test_dkesolution_density_pressure_Phi1(species2):
    """Density/pressure moments and Phi_1 for f1 = c_s(theta, zeta) * F0.

    For this f1 the perturbations are dn_s = c_s n_s and dp_s = c_s n_s T_s, and
    Phi_1 follows from quasi-neutrality, so the total charge including the
    Boltzmann response is constant on the surface to first order in Phi_1.
    """
    nt, nz = 4, 3
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)[:, None]
    zeta = np.linspace(0, 2 * np.pi, nz, endpoint=False)[None, :]
    ones = np.ones((nt, nz))
    field = Field(
        rho=0.5,
        B_sup_t=ones,
        B_sup_z=ones,
        B_sub_t=ones,
        B_sub_z=ones,
        Bmag=ones,
        sqrtg=1 + 0.3 * np.cos(theta) + 0.1 * np.sin(zeta),
        Psi=1.0,
        iota=1.0,
        R_major=10.0,
        a_minor=1.0,
    )
    pitchgrid = UniformPitchAngleGrid(5)
    # nx=5 makes the Maxwell quadrature exact for the density/pressure moments
    speedgrid = MaxwellSpeedGrid(5)
    F0 = jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species2])
    F0 = F0[:, :, None, None, None]
    c = jnp.stack(
        [1e-3 * jnp.cos(theta + zeta) * ones, -2e-3 * jnp.sin(2 * theta) * ones]
    )
    f1 = c[:, None, None] * F0 * jnp.ones((1, 1, pitchgrid.nalpha, 1, 1))
    sol = DKESolution(
        F0=F0,
        f1=f1,
        rhs=jnp.zeros(f1.size),
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species2,
        Erho=jnp.array(0.0),
        EparB=jnp.array(0.0),
        background=[],
    )

    n0 = np.array([sp.density for sp in species2])[:, None, None]
    T = np.array([sp.temperature for sp in species2])[:, None, None] * _JOULE_PER_EV
    q = np.array([sp.species.charge for sp in species2])[:, None, None]
    np.testing.assert_allclose(sol.get("n1"), c * n0, rtol=1e-10)
    np.testing.assert_allclose(sol.get("p1"), c * n0 * T, rtol=1e-10)

    fsa = lambda x: (x * field.sqrtg).sum() / field.sqrtg.sum()  # noqa: E731
    charge = (q * c * n0).sum(axis=0)
    Phi1 = (charge - fsa(charge)) / (q**2 * n0 / T).sum()
    np.testing.assert_allclose(
        sol.get("Phi_1"), Phi1, rtol=1e-10, atol=1e-10 * np.abs(Phi1).max()
    )
    assert sol.get("Phi_1").shape == (nt, nz)

    boltz = np.exp(-q * Phi1 / T)
    np.testing.assert_allclose(sol.get("n"), boltz * n0 + c * n0, rtol=1e-10)
    np.testing.assert_allclose(sol.get("p"), boltz * n0 * T + c * n0 * T, rtol=1e-10)
    # an isotropic f carries no parallel momentum
    Pi = np.asarray(sol.get("<momentum_flux>"))
    assert Pi.shape == (2,)
    scale = np.abs(np.asarray(sol.get("<heat_flux>"))).max() + 1e-300
    np.testing.assert_allclose(Pi, 0, atol=1e-12 * scale)
    # Phi_1 solves the linearized quasi-neutrality condition, so the charge
    # variation left by the full Boltzmann response is second order in Phi_1
    total_charge = np.asarray((q * sol.get("n")).sum(axis=0))
    second_order = (np.abs(q) * n0 * (q * Phi1 / T) ** 2).sum(axis=0).max()
    np.testing.assert_allclose(total_charge - fsa(total_charge), 0, atol=second_order)


def _solution_with_Phi1(field, species, f1=None, Phi1_max=None, Erho=0.0):
    """DKESolution with F0 = 0, optionally with Phi_1 from f1.

    f1 is rescaled so that max|Phi_1| = Phi1_max.
    """
    pitchgrid = UniformPitchAngleGrid(5)
    speedgrid = MaxwellSpeedGrid(3)
    ns = len(species)
    N = ns * speedgrid.nx * pitchgrid.nalpha * field.ntheta * field.nzeta
    kwargs: dict[str, Any] = dict(
        F0=jnp.zeros((ns, speedgrid.nx)),
        f1=jnp.zeros(N),
        rhs=jnp.zeros(N),
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        Erho=jnp.array(Erho),
        EparB=jnp.array(0.0),
        background=[],
    )
    if f1 is not None:
        Phi1 = DKESolution(field=field, **{**kwargs, "f1": f1}).get("Phi_1")
        kwargs["f1"] = f1 * Phi1_max / np.abs(Phi1).max()
    return DKESolution(field=field, **kwargs)


def _classical_fluxes(field, species, f1=None, Phi1_max=None, Erho=0.0):
    """Classical (particle, conductive heat) fluxes, and the solution."""
    sol = _solution_with_Phi1(field, species, f1, Phi1_max, Erho)
    T = np.array([sp.temperature for sp in species]) * _JOULE_PER_EV
    Gamma = np.asarray(sol.get("<classical_particle_flux>", coulomb_log=17.0))
    Q = np.asarray(sol.get("<classical_heat_flux>", coulomb_log=17.0))
    return Gamma, Q - 2.5 * T * Gamma, sol


def test_dkesolution_classical_fluxes_and_flows(dummy_field, species2):
    """Classical fluxes and perpendicular flows satisfy conservation laws,
    symmetries and known limits.
    """
    sol = _solution_with_Phi1(dummy_field, species2)
    with pytest.raises(ValueError, match="g_sup_rr"):
        sol.get("<classical_particle_flux>")
    with pytest.raises(ValueError, match="g_sup_rr"):
        sol.get("Vperp")

    rng = np.random.default_rng(0)
    nt = nz = 3
    field = Field.from_boozer(
        rho=0.5,
        Bmag=2 + 0.3 * rng.random((nt, nz)),
        I=0.1,
        G=5.0,
        iota=0.5,
        Psi=1.0,
        g_sup_rr=3 + rng.random((nt, nz)),
    )
    assert field.g_sup_rr is not None
    geometry = field.flux_surface_average(field.g_sup_rr / field.Bmag**2)
    e, lnlambda = elementary_charge, 17.0
    Carbon6 = Species(12, 6)

    # Phi_1 up to 1 kV and unequal temperatures, so the Boltzmann weights are
    # nonlinear and the Phi_1 * dlnT terms are nonzero
    species = [
        LocalMaxwellian(Hydrogen, 5e3, 4e19, -3e3, -2e19),
        LocalMaxwellian(Electron, 8e3, 7e19, -5e3, -3e19),
        LocalMaxwellian(Carbon6, 4e3, 5e18, -1e3, 2e18),
    ]
    N = len(species) * 3 * 5 * nt * nz
    Erho = 2e3
    Gamma, q, sol = _classical_fluxes(field, species, rng.standard_normal(N), 1e3, Erho)
    Phi1 = np.asarray(sol.get("Phi_1"))
    T = np.array([sp.temperature for sp in species]) * _JOULE_PER_EV
    Z = np.array([sp.species.charge for sp in species])

    # momentum conservation makes the classical particle flux ambipolar
    np.testing.assert_allclose(
        (Z * Gamma).sum(), 0, atol=1e-12 * np.abs(Z * Gamma).max()
    )

    # with pseudo-densities, the fluxes are the surface average of the Phi_1 = 0
    # fluxes with the local density and its radial derivative at fixed Phi_1. The
    # radial derivative of Phi_1 is the same for all species, so like E_r it
    # does not drive classical transport. Likewise the perpendicular flows are
    # the Phi_1 = 0 flows with the local density and gradient, as the radial
    # derivative of Phi_1 cancels between the Boltzmann factor and the ExB drift.
    Vpar = np.asarray(sol.get("V||"))
    Vperp = np.asarray(sol.get("Vperp"))
    Vperp_theta = sol.get("V^theta") - Vpar * field.B_sup_t / field.Bmag
    Vperp_zeta = sol.get("V^zeta") - Vpar * field.B_sup_z / field.Bmag
    Vperp_local = np.zeros((len(species), nt, nz))
    Vperp_theta_local = np.zeros((len(species), nt, nz))
    Vperp_zeta_local = np.zeros((len(species), nt, nz))
    Gamma_local = np.zeros((len(species), nt, nz))
    q_local = np.zeros((len(species), nt, nz))
    for i in range(nt):
        for j in range(nz):
            local_field = Field.from_boozer(
                rho=0.5,
                Bmag=field.Bmag[i : i + 1, j : j + 1],
                I=0.1,
                G=5.0,
                iota=0.5,
                Psi=1.0,
                g_sup_rr=field.g_sup_rr[i : i + 1, j : j + 1],
            )
            local_species = []
            for sp, Ts, Zs in zip(species, T, Z):
                boltzmann = np.exp(-Zs * Phi1[i, j] / Ts)
                dlnn = (
                    sp.dndrho / sp.density
                    + Zs * Phi1[i, j] / Ts * sp.dTdrho / sp.temperature
                )
                local_species.append(
                    LocalMaxwellian(
                        sp.species,
                        sp.temperature,
                        sp.density * boltzmann,
                        sp.dTdrho,
                        sp.density * boltzmann * dlnn,
                    )
                )
            G_ij, q_ij, sol_ij = _classical_fluxes(
                local_field, local_species, Erho=Erho
            )
            Gamma_local[:, i, j] = G_ij
            q_local[:, i, j] = q_ij
            Vperp_local[:, i, j] = sol_ij.get("Vperp")[:, 0, 0]
            Vperp_theta_local[:, i, j] = sol_ij.get("V^theta")[:, 0, 0]
            Vperp_zeta_local[:, i, j] = sol_ij.get("V^zeta")[:, 0, 0]
    np.testing.assert_allclose(
        Gamma, field.flux_surface_average(Gamma_local), rtol=1e-10
    )
    np.testing.assert_allclose(q, field.flux_surface_average(q_local), rtol=1e-10)
    np.testing.assert_allclose(Vperp, Vperp_local, rtol=1e-10)
    np.testing.assert_allclose(Vperp_theta, Vperp_theta_local, rtol=1e-10)
    np.testing.assert_allclose(Vperp_zeta, Vperp_zeta_local, rtol=1e-10)
    # without Phi_1, the diamagnetic and ExB flow along b x grad(rho)
    sol0 = _solution_with_Phi1(field, species, Erho=Erho)
    dp = np.array(
        [sp.dndrho * sp.temperature + sp.density * sp.dTdrho for sp in species]
    )
    n0 = np.array([sp.density for sp in species])
    omega = (dp * _JOULE_PER_EV / (Z * n0) - Erho)[:, None, None]
    np.testing.assert_allclose(
        sol0.get("Vperp"),
        np.sqrt(field.g_sup_rr) / field.Bmag * omega,
        rtol=1e-10,
    )

    # Onsager symmetry at equal temperatures, for fluxes (Gamma_a, q_a/T)
    # conjugate to forces (p_a'/n_a, T_a')
    Teq = 5e3
    L = np.zeros((6, 6))
    for k in range(6):
        forces = np.zeros(6)
        forces[k] = 1.0
        sps = []
        for sp, A1, A2 in zip(species, forces[:3], forces[3:]):
            n = sp.density
            sps.append(LocalMaxwellian(sp.species, Teq, n, A2, n * (A1 - A2) / Teq))
        G_k, q_k, _ = _classical_fluxes(field, sps)
        L[:, k] = np.concatenate([G_k, q_k / (Teq * _JOULE_PER_EV)])
    np.testing.assert_allclose(L, L.T, atol=1e-12 * np.abs(L).max())

    # Braginskii's strongly magnetized limit for a single ion species: no
    # particle flux, and perpendicular conductivity 2 n T / (m Omega^2 tau_i)
    n, T, dT = 5e19, 5e3 * _JOULE_PER_EV, -3e3 * _JOULE_PER_EV
    Gamma, q, _ = _classical_fluxes(
        field, [LocalMaxwellian(Hydrogen, 5e3, n, -3e3, -2e19)]
    )
    m = Hydrogen.mass
    tau_i = 12 * np.pi**1.5 * epsilon_0**2 * np.sqrt(m) * T**1.5 / (n * e**4 * lnlambda)
    np.testing.assert_allclose(Gamma, 0, atol=1e-30)
    np.testing.assert_allclose(
        q, -2 * n * T * m / (e**2 * tau_i) * geometry * dT, rtol=1e-10
    )

    # Braginskii's strongly magnetized electron friction and heat flux with
    # infinitely heavy singly charged ions: resistive and thermal force friction
    # R = -(m n / tau) u - 3/2 n / (Omega tau) b x grad(T), and conductivity
    # kappa = (sqrt(2) + 13/4) n T / (m Omega^2 tau_e) with the u coupling
    # q_u = 3/2 n T / (Omega tau) b x u
    n, T, Ti = 5e19, 8e3 * _JOULE_PER_EV, 4e3 * _JOULE_PER_EV
    dn, dT = -2e19, -5e3 * _JOULE_PER_EV
    species = [
        LocalMaxwellian(Electron, 8e3, n, -5e3, -2e19),
        LocalMaxwellian(Species(1e8, 1), 4e3, n, 0.0, -2e19),
    ]
    Gamma, q, _ = _classical_fluxes(field, species)
    m = Electron.mass
    tau_e = (
        6
        * np.sqrt(2)
        * np.pi**1.5
        * epsilon_0**2
        * np.sqrt(m)
        * T**1.5
        / (n * e**4 * lnlambda)
    )
    dp = T * dn + n * dT + Ti * dn
    C = m / (e**2 * tau_e) * geometry
    kappa = np.sqrt(2) + 13 / 4
    np.testing.assert_allclose(Gamma[0], -C * (dp - 1.5 * n * dT), rtol=1e-8)
    np.testing.assert_allclose(
        q[0], -C * (kappa * n * T * dT - 1.5 * T * dp), rtol=1e-8
    )


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
