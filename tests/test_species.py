"""Unit tests for yancc/species.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import elementary_charge, epsilon_0, proton_mass

from yancc.species import (
    JOULE_PER_EV,
    Deuterium,
    Electron,
    Estar,
    GlobalMaxwellian,
    Hydrogen,
    LocalMaxwellian,
    Species,
    chandrasekhar,
    collisionality,
    coulomb_logarithm,
    debye_length,
    gamma_ab,
    nuD_ab,
    nupar_ab,
    poloidal_mach,
    rhostar,
)


@pytest.fixture
def hydrogen_maxwellian():
    return LocalMaxwellian(
        Hydrogen, temperature=1000.0, density=1e19, dTdrho=-500.0, dndrho=-2e18
    )


@pytest.fixture
def electron_maxwellian():
    return LocalMaxwellian(
        Electron, temperature=1000.0, density=1e19, dTdrho=-500.0, dndrho=-2e18
    )


def test_species_mass_units():
    s = Species(1, 1)
    np.testing.assert_allclose(float(s.mass), proton_mass, rtol=1e-10)


def test_species_charge_units():
    s = Species(1, 1)
    np.testing.assert_allclose(float(s.charge), elementary_charge, rtol=1e-10)


def test_species_charge_negative():
    np.testing.assert_allclose(float(Electron.charge), -elementary_charge, rtol=1e-10)


def test_species_mass_ratio():
    # Deuterium should be twice the mass of Hydrogen
    np.testing.assert_allclose(
        float(Deuterium.mass), 2 * float(Hydrogen.mass), rtol=1e-10
    )


def test_local_maxwellian_v_thermal(hydrogen_maxwellian):
    lm = hydrogen_maxwellian
    expected = float(jnp.sqrt(2 * lm.temperature * JOULE_PER_EV / lm.species.mass))
    np.testing.assert_allclose(float(lm.v_thermal), expected, rtol=1e-10)


def test_local_maxwellian_from_scale_lengths_round_trip():
    aLT, aLn = 2.0, 3.0
    T, n = 1000.0, 1e19
    lm = LocalMaxwellian.from_scale_lengths(Hydrogen, T, n, aLT=aLT, aLn=aLn)
    np.testing.assert_allclose(float(lm.aLT), aLT, rtol=1e-10)
    np.testing.assert_allclose(float(lm.aLn), aLn, rtol=1e-10)
    np.testing.assert_allclose(float(lm.temperature), T, rtol=1e-10)
    np.testing.assert_allclose(float(lm.density), n, rtol=1e-10)


def test_local_maxwellian_normalization(hydrogen_maxwellian):
    """Integral of Maxwellian over all velocity space equals density."""
    lm = hydrogen_maxwellian
    v_th = float(lm.v_thermal)
    # integrate 4 pi v^2 f(v) dv numerically
    v_vals = np.linspace(0, 6 * v_th, 2000)
    f_vals = np.array([float(lm(v)) for v in v_vals])
    integral = 4 * np.pi * np.trapezoid(v_vals**2 * f_vals, v_vals)
    np.testing.assert_allclose(integral, float(lm.density), rtol=1e-4)


def test_local_maxwellian_peak(hydrogen_maxwellian):
    """Maxwellian peaks at v=0."""
    lm = hydrogen_maxwellian
    f0 = float(lm(0.0))
    f1 = float(lm(lm.v_thermal))
    assert f0 > f1


def test_global_maxwellian_localize_constant():
    """Constant profiles -> zero gradients in localized Maxwellian."""
    T0, n0 = 1000.0, 1e19
    gm = GlobalMaxwellian(
        Hydrogen,
        temperature=lambda r: T0 * jnp.ones_like(r),
        density=lambda r: n0 * jnp.ones_like(r),
    )
    lm = gm.localize(0.5)
    np.testing.assert_allclose(float(lm.dTdrho), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(lm.dndrho), 0.0, atol=1e8)
    np.testing.assert_allclose(float(lm.temperature), T0, rtol=1e-10)
    np.testing.assert_allclose(float(lm.density), n0, rtol=1e-10)


def test_global_maxwellian_localize_gradient():
    """Linear T profile -> gradient matches slope in localized Maxwellian."""
    dT_drho = -500.0
    T0 = 1000.0
    gm = GlobalMaxwellian(
        Hydrogen,
        temperature=lambda r: T0 + dT_drho * r,
        density=lambda r: 1e19 * jnp.ones_like(r),
    )
    lm = gm.localize(0.5)
    np.testing.assert_allclose(float(lm.dTdrho), dT_drho, rtol=1e-6)


def test_global_maxwellian_v_thermal():
    """GlobalMaxwellian.v_thermal matches sqrt(2 T / m) and the localized value."""
    T0, n0 = 1000.0, 1e19
    gm = GlobalMaxwellian(
        Hydrogen,
        temperature=lambda r: T0 * jnp.ones_like(r),
        density=lambda r: n0 * jnp.ones_like(r),
    )
    expected = float(jnp.sqrt(2 * T0 * JOULE_PER_EV / Hydrogen.mass))
    np.testing.assert_allclose(float(gm.v_thermal(0.5)), expected, rtol=1e-12)
    # the global v_thermal at r should equal the localized Maxwellian's v_thermal
    np.testing.assert_allclose(
        float(gm.v_thermal(0.5)), float(gm.localize(0.5).v_thermal), rtol=1e-12
    )


def test_global_maxwellian_call_peaks_at_zero():
    """GlobalMaxwellian(rho, v) peaks at v=0 and matches the analytic value there."""
    T0, n0 = 1000.0, 1e19
    gm = GlobalMaxwellian(
        Hydrogen,
        temperature=lambda r: T0 * jnp.ones_like(r),
        density=lambda r: n0 * jnp.ones_like(r),
    )
    rho = 0.5
    vth = float(gm.v_thermal(rho))
    f0 = float(gm(rho, 0.0))
    f1 = float(gm(rho, vth))
    assert f0 > f1
    expected0 = n0 / (np.sqrt(np.pi) * vth) ** 3
    np.testing.assert_allclose(f0, expected0, rtol=1e-10)


def test_rhostar_scales_linearly_with_x(hydrogen_maxwellian, field):
    lm = hydrogen_maxwellian
    r1 = float(rhostar(lm, field, 1.0))
    r2 = float(rhostar(lm, field, 2.0))
    assert r1 > 0
    np.testing.assert_allclose(r2 / r1, 2.0, rtol=1e-10)


def test_poloidal_mach_scales_inversely_with_x(hydrogen_maxwellian, field):
    # E×B rotation is speed-independent while the transit rate ~ x, so M_p ~ 1/x.
    m1 = float(poloidal_mach(hydrogen_maxwellian, field, 3e3, 1.0))
    m2 = float(poloidal_mach(hydrogen_maxwellian, field, 3e3, 2.0))
    np.testing.assert_allclose(m2 / m1, 0.5, rtol=1e-10)


def test_poloidal_mach_matches_definition(hydrogen_maxwellian, field):
    # M_p = E* / (a_minor <B^theta/|B|>), the ratio of E×B to streaming poloidal
    # rotation frequencies.
    bdgt = float(field.flux_surface_average(field.B_sup_t / field.Bmag))
    expected = float(Estar(hydrogen_maxwellian, field, 3e3, 1.0)) / (
        float(field.a_minor) * abs(bdgt)
    )
    got = float(poloidal_mach(hydrogen_maxwellian, field, 3e3, 1.0))
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_chandrasekhar_positive():
    xs = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    vals = chandrasekhar(xs)
    assert jnp.all(vals > 0)


def test_chandrasekhar_large_x():
    """For large x, chandrasekhar(x) ~= 1 / (2 x^2)."""
    x = jnp.array(10.0)
    np.testing.assert_allclose(
        float(chandrasekhar(x)), float(1 / (2 * x**2)), rtol=1e-4
    )


def test_chandrasekhar_monotone_decrease():
    """Chandrasekhar should decrease for large x (beyond peak)."""
    xs = jnp.array([1.0, 2.0, 5.0, 10.0])
    vals = chandrasekhar(xs)
    assert jnp.all(jnp.diff(vals) < 0)


def test_debye_length_single_species(hydrogen_maxwellian):
    """Debye length matches analytic formula for single species."""
    lm = hydrogen_maxwellian
    expected = float(
        jnp.sqrt(
            epsilon_0
            * lm.temperature
            * JOULE_PER_EV
            / lm.density
            / lm.species.charge**2
        )
    )
    result = float(debye_length(lm))
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_debye_length_two_species_smaller(hydrogen_maxwellian, electron_maxwellian):
    """Adding a second species reduces the Debye length."""
    lm_i = hydrogen_maxwellian
    lm_e = electron_maxwellian
    ld_single = float(debye_length(lm_i))
    ld_both = float(debye_length(lm_i, lm_e))
    assert ld_both < ld_single


def test_nuD_nupar_identity(hydrogen_maxwellian, electron_maxwellian):
    """nuD_ab + nupar_ab / 2 = gamma_ab * nb / v^3 * erf(v / vt_b)."""
    lm_a = hydrogen_maxwellian
    lm_b = electron_maxwellian
    v = lm_a.v_thermal * 1.5
    gab = gamma_ab(lm_a, lm_b)
    x = v / lm_b.v_thermal
    expected = float(gab * lm_b.density / v**3 * jax.scipy.special.erf(x))
    result = float(nuD_ab(lm_a, lm_b, v) + nupar_ab(lm_a, lm_b, v) / 2)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_collisionality_includes_self(hydrogen_maxwellian):
    """Collisionality with no extra species should still include self-collision."""
    lm = hydrogen_maxwellian
    v = lm.v_thermal
    nu = float(collisionality(lm, v))
    nu_self = float(nuD_ab(lm, lm, v))
    np.testing.assert_allclose(nu, nu_self, rtol=1e-10)


def test_collisionality_increases_with_background(
    hydrogen_maxwellian, electron_maxwellian
):
    """Adding background electrons increases ion collisionality."""
    lm_i = hydrogen_maxwellian
    lm_e = electron_maxwellian
    v = lm_i.v_thermal
    nu_self = float(collisionality(lm_i, v))
    nu_both = float(collisionality(lm_i, v, lm_e))
    assert nu_both > nu_self


# ---------------------------------------------------------------------------
# NRL formulary comparisons for coulomb_logarithm
# ---------------------------------------------------------------------------
# NRL Plasma Formulary 2019, p. 34.
# The NRL expressions are derived from ln(b_max/b_min) with specific choices
# for the typical collision velocity; agreement within ~1–2 is expected.


def _nrl_lnlambda_ee_cold(n_m3, T_eV):
    """NRL e-e, classical regime T_e < 10 eV:

    ln Λ = 23 - ln(n^{1/2} T^{-3/2}).
    """
    n_cm3 = n_m3 * 1e-6
    return 23.0 - np.log(np.sqrt(n_cm3) * T_eV ** (-1.5))


def _nrl_lnlambda_ee_hot(n_m3, T_eV):
    """NRL e-e, quantum regime T_e > 10 eV:

    ln Λ = 23.5 - ln(n^{1/2} T^{-5/4}) - correction.
    """
    n_cm3 = n_m3 * 1e-6
    return (
        23.5
        - np.log(np.sqrt(n_cm3) * T_eV ** (-1.25))
        - np.sqrt(1e-5 + (np.log(T_eV) - 2.0) ** 2 / 16.0)
    )


def _nrl_lnlambda_ei_hot_electron(n_e_m3, T_e_eV):
    """NRL e-i, T_e > 10 Z^2 eV: ln Λ = 24 - ln(n_e^{1/2} T_e^{-1})."""
    n_cm3 = n_e_m3 * 1e-6
    return 24.0 - np.log(np.sqrt(n_cm3) / T_e_eV)


def _nrl_lnlambda_ii(n_m3, T_a_eV, T_b_eV, Za, Zb, mu_a, mu_b):
    """NRL i-i: 23 - ln[Za Zb (mu_a+mu_b)/(mu_a T_b + mu_b T_a) * sqrt(sum n Z^2/T)]."""
    n_cm3 = n_m3 * 1e-6
    mass_factor = Za * Zb * (mu_a + mu_b) / (mu_a * T_b_eV + mu_b * T_a_eV)
    debye_factor = np.sqrt(n_cm3 * Za**2 / T_a_eV + n_cm3 * Zb**2 / T_b_eV)
    return 23.0 - np.log(mass_factor * debye_factor)


def test_coulomb_logarithm_ee_cold_nrl():
    """e-e ln Λ at T_e=5 eV matches NRL classical-regime formula."""
    T_eV, n = 5.0, 1e19
    lm = LocalMaxwellian(Electron, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0)
    result = float(coulomb_logarithm(lm, lm))
    expected = _nrl_lnlambda_ee_cold(n, T_eV)
    np.testing.assert_allclose(result, expected, rtol=3e-2)


def test_coulomb_logarithm_ee_hot_nrl():
    """e-e ln Λ at T_e=100 eV matches NRL quantum-regime formula."""
    T_eV, n = 100.0, 1e19
    lm = LocalMaxwellian(Electron, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0)
    result = float(coulomb_logarithm(lm, lm))
    expected = _nrl_lnlambda_ee_hot(n, T_eV)
    np.testing.assert_allclose(result, expected, rtol=3e-2)


def test_coulomb_logarithm_ei_nrl():
    """e-i ln Λ at T=1 keV matches NRL hot-electron formula (T_e > 10 Z^2 eV)."""
    T_eV, n = 1000.0, 1e19
    lm_e = LocalMaxwellian(
        Electron, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0
    )
    lm_i = LocalMaxwellian(
        Hydrogen, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0
    )
    result = float(coulomb_logarithm(lm_e, lm_i))
    expected = _nrl_lnlambda_ei_hot_electron(n, T_eV)
    np.testing.assert_allclose(result, expected, rtol=5e-2)


def test_coulomb_logarithm_ii_nrl():
    """H-H ln Λ at T=1 keV matches NRL ion-ion formula."""
    T_eV, n = 1000.0, 1e19
    lm = LocalMaxwellian(Hydrogen, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0)
    result = float(coulomb_logarithm(lm, lm))
    # mu = 1 amu for protons; Za = Zb = 1
    expected = _nrl_lnlambda_ii(n, T_eV, T_eV, Za=1, Zb=1, mu_a=1.0, mu_b=1.0)
    np.testing.assert_allclose(result, expected, rtol=1e-2)


# ---------------------------------------------------------------------------
# NRL formulary comparisons for collisionality
# ---------------------------------------------------------------------------
# NRL Plasma Formulary 2019, p. 31.


def _nrl_nuD_ee_slow(n_m3, T_eV, lnlambda, energy_eV):
    """NRL e-e perp thermal collision rate (s^{-1}) in slow limit (v/vth)^2 << 1"""
    # NRL includes extra factor of 2 for perpendicular collisionality
    return 5.8e-6 / 2 * (n_m3 * 1e-6) * lnlambda * T_eV ** (-1 / 2) * energy_eV ** (-1)


def _nrl_nupar_ee_slow(n_m3, T_eV, lnlambda, energy_eV):
    """NRL e-e par thermal collision rate (s^{-1}) in slow limit (v/vth)^2 << 1"""
    return 2.9e-6 * (n_m3 * 1e-6) * lnlambda * T_eV ** (-1 / 2) * energy_eV ** (-1)


def _nrl_nuD_ee_fast(n_m3, T_eV, lnlambda, energy_eV):
    """NRL e-e perp thermal collision rate (s^{-1}) in fast limit (v/vth)^2 >> 1"""
    # NRL includes extra factor of 2 for perpendicular collisionality
    return 7.7e-6 / 2 * (n_m3 * 1e-6) * lnlambda * energy_eV ** (-3 / 2)


def _nrl_nupar_ee_fast(n_m3, T_eV, lnlambda, energy_eV):
    """NRL e-e par thermal collision rate (s^{-1}) in slow limit (v/vth)^2 << 1"""
    return 3.9e-6 * (n_m3 * 1e-6) * lnlambda * T_eV * energy_eV ** (-5 / 2)


def _nrl_nuD_ii_slow(n_m3, T_eV, lnlambda, energy_eV, mu1, mu2, Z1, Z2):
    """NRL i-i perp thermal collision rate (s^{-1}) in slow limit (v/vth)^2 << 1"""
    return (
        1.4e-7
        / 2  # NRL includes extra factor of 2 for perpendicular collisionality
        * (n_m3 * 1e-6)
        * lnlambda
        * (Z1**2 * Z2**2)
        * mu2 ** (1 / 2)
        * mu1 ** (-1)
        * T_eV ** (-1 / 2)
        * energy_eV ** (-1)
    )


def _nrl_nupar_ii_slow(n_m3, T_eV, lnlambda, energy_eV, mu1, mu2, Z1, Z2):
    """NRL i-i par thermal collision rate (s^{-1}) in slow limit (v/vth)^2 << 1"""
    return (
        6.8e-8
        * (n_m3 * 1e-6)
        * lnlambda
        * (Z1**2 * Z2**2)
        * mu2 ** (1 / 2)
        * mu1 ** (-1)
        * T_eV ** (-1 / 2)
        * energy_eV ** (-1)
    )


def _nrl_nuD_ii_fast(n_m3, T_eV, lnlambda, energy_eV, mu1, mu2, Z1, Z2):
    """NRL i-i perp thermal collision rate (s^{-1}) in fast limit (v/vth)^2 >> 1"""
    return (
        1.8e-7
        / 2  # NRL includes extra factor of 2 for perpendicular collisionality
        * (n_m3 * 1e-6)
        * lnlambda
        * (Z1**2 * Z2**2)
        * mu1 ** (-1 / 2)
        * energy_eV ** (-3 / 2)
    )


def _nrl_nupar_ii_fast(n_m3, T_eV, lnlambda, energy_eV, mu1, mu2, Z1, Z2):
    """NRL i-i par thermal collision rate (s^{-1}) in slow limit (v/vth)^2 << 1"""
    return (
        9.0e-8
        * (n_m3 * 1e-6)
        * lnlambda
        * (Z1**2 * Z2**2)
        * mu1 ** (1 / 2)
        * mu2 ** (-1)
        * T_eV ** (1)
        * energy_eV ** (-5 / 2)
    )


@pytest.mark.parametrize("speed", ["fast", "slow"])
def test_perpendicular_collisionality_ee_nrl(speed):
    """e-e perp self-collisionality at v<<v_th and v>>v_th agrees with NRL."""
    if speed == "fast":
        fac = 10
        fun1 = _nrl_nuD_ee_fast
    else:
        fac = 0.1
        fun1 = _nrl_nuD_ee_slow

    T_eV, n = 1000.0, 1e19
    lm = LocalMaxwellian(Electron, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0)
    v = fac * lm.v_thermal
    energy_eV = (lm.species.mass / 2 * v**2) / JOULE_PER_EV
    result = float(nuD_ab(lm, lm, v))
    lnl = float(coulomb_logarithm(lm, lm))
    np.testing.assert_allclose(result, fun1(n, T_eV, lnl, energy_eV), rtol=1e-2)


@pytest.mark.parametrize("speed", ["fast", "slow"])
def test_parallel_collisionality_ee_nrl(speed):
    """e-e par self-collisionality at v<<v_th and v>>v_th agrees with NRL."""
    if speed == "fast":
        fac = 10
        fun1 = _nrl_nupar_ee_fast
    else:
        fac = 0.1
        fun1 = _nrl_nupar_ee_slow

    T_eV, n = 1000.0, 1e19
    lm = LocalMaxwellian(Electron, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0)
    v = fac * lm.v_thermal
    energy_eV = (lm.species.mass / 2 * v**2) / JOULE_PER_EV
    result = float(nupar_ab(lm, lm, v))
    lnl = float(coulomb_logarithm(lm, lm))
    np.testing.assert_allclose(result, fun1(n, T_eV, lnl, energy_eV), rtol=1e-2)


@pytest.mark.parametrize("speed", ["fast", "slow"])
def test_perpendicular_collisionality_ii_nrl(speed):
    """H-H perp self-collisionality at agrees with NRL."""
    if speed == "fast":
        fac = 10
        fun1 = _nrl_nuD_ii_fast
    else:
        fac = 0.1
        fun1 = _nrl_nuD_ii_slow

    T_eV, n = 1000.0, 1e19
    lm = LocalMaxwellian(Hydrogen, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0)
    v = fac * lm.v_thermal
    energy_eV = (lm.species.mass / 2 * v**2) / JOULE_PER_EV
    result = float(nuD_ab(lm, lm, v))
    mu1 = mu2 = Z1 = Z2 = 1  # mu = 1 amu for protons; Z1 = Z2 = 1
    lnl = float(coulomb_logarithm(lm, lm))
    np.testing.assert_allclose(
        result, fun1(n, T_eV, lnl, energy_eV, mu1, mu2, Z1, Z2), rtol=5e-2
    )


@pytest.mark.parametrize("speed", ["fast", "slow"])
def test_parallel_collisionality_ii_nrl(speed):
    """H-H par self-collisionality at agrees with NRL."""
    if speed == "fast":
        fac = 10
        fun1 = _nrl_nupar_ii_fast
    else:
        fac = 0.1
        fun1 = _nrl_nupar_ii_slow

    T_eV, n = 1000.0, 1e19
    lm = LocalMaxwellian(Hydrogen, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0)
    v = fac * lm.v_thermal
    energy_eV = (lm.species.mass / 2 * v**2) / JOULE_PER_EV
    result = float(nupar_ab(lm, lm, v))
    mu1 = mu2 = Z1 = Z2 = 1  # mu = 1 amu for protons; Z1 = Z2 = 1
    lnl = float(coulomb_logarithm(lm, lm))
    np.testing.assert_allclose(
        result, fun1(n, T_eV, lnl, energy_eV, mu1, mu2, Z1, Z2), rtol=5e-2
    )


def test_collisionality_density_scaling():
    """Collisionality / ln Λ scales exactly linearly with density."""
    T_eV = 1000.0
    lm1 = LocalMaxwellian(
        Hydrogen, temperature=T_eV, density=1e19, dTdrho=0.0, dndrho=0.0
    )
    lm2 = LocalMaxwellian(
        Hydrogen, temperature=T_eV, density=2e19, dTdrho=0.0, dndrho=0.0
    )
    v = float(lm1.v_thermal)
    # nuD_ab ~ lnΛ * n_b; dividing out lnΛ leaves pure n-scaling
    nu1 = float(collisionality(lm1, v)) / float(coulomb_logarithm(lm1, lm1))
    nu2 = float(collisionality(lm2, v)) / float(coulomb_logarithm(lm2, lm2))
    np.testing.assert_allclose(nu2 / nu1, 2.0, rtol=1e-5)


def test_collisionality_electron_ion_mass_ratio():
    """Electron self-collision is sqrt(m_p/m_e) * (lnΛ_ee/lnΛ_ii) times proton rate."""
    T_eV, n = 1000.0, 1e19
    lm_e = LocalMaxwellian(
        Electron, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0
    )
    lm_i = LocalMaxwellian(
        Hydrogen, temperature=T_eV, density=n, dTdrho=0.0, dndrho=0.0
    )
    nu_e = float(collisionality(lm_e, float(lm_e.v_thermal)))
    nu_i = float(collisionality(lm_i, float(lm_i.v_thermal)))
    lnl_e = float(coulomb_logarithm(lm_e, lm_e))
    lnl_i = float(coulomb_logarithm(lm_i, lm_i))
    # nuD(v_th) ~ Z^4 * lnΛ / m^{1/2} * T^{-3/2}; both Z=1 here
    expected_ratio = (lnl_e / lnl_i) * np.sqrt(
        float(Hydrogen.mass) / float(Electron.mass)
    )
    np.testing.assert_allclose(nu_e / nu_i, expected_ratio, rtol=1e-5)
