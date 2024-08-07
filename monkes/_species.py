import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from scipy.constants import Boltzmann, elementary_charge, epsilon_0, hbar, proton_mass

JOULE_PER_EV = 11606 * Boltzmann
EV_PER_JOULE = 1 / JOULE_PER_EV


def _rescale(a):
    def converter(f):
        if callable(f):

            @functools.wraps(f)
            def wrapped(*args, **kwargs):
                return a * f(*args, **kwargs)

            return wrapped
        else:
            return a * f

    return converter


class Species(eqx.Module):
    """Atomic species of arbitrary charge and mass.

    Parameters
    ----------
    mass : float
        Mass of the species, in units of proton masses. Will be converted to kg.
    charge : float
        Charge of the species, in units of elementary charge. Will be converted to
        Coulombs.

    """

    mass: float = eqx.field(converter=_rescale(proton_mass))
    charge: float = eqx.field(converter=_rescale(elementary_charge))


Electron = Species(1 / 1836.15267343, -1)
Hydrogen = Species(1, 1)
Deuterium = Species(2, 1)
Tritium = Species(3, 1)


class LocalMaxwellian(eqx.Module):
    """Local Maxwellian distribution function on a single surface.

    Parameters
    ----------
    species : Species
        Atomic species of the distribution function.
    temperature : float
        Temperature of the species, in units of eV.
    density : float
        Density of the species, in units of particles/m^3.

    """

    species: Species
    temperature: float  # in units of eV
    density: float  # in units of particles/m^3
    v_thermal: float  # in units of m/s
    dndr: float
    dTdr: float

    def __init__(
        self,
        species: Species,
        temperature: float,
        density: float,
        dndr: float,
        dTdr: float,
    ):
        self.species = species
        self.temperature = temperature
        self.density = density
        self.v_thermal = jnp.sqrt(
            2 * self.temperature * JOULE_PER_EV / self.species.mass
        )
        self.dndr = dndr
        self.dTdr = dTdr

    def __call__(self, v: float) -> float:
        return (
            self.density
            / (jnp.sqrt(jnp.pi) * self.v_thermal) ** 3
            * jnp.exp(-(v**2) / self.v_thermal**2)
        )


class GlobalMaxwellian(eqx.Module):
    """Global Maxwellian distribution function over radius.

    Parameters
    ----------
    species : Species
        Atomic species of the distribution function.
    temperature : callable
        Temperature of the species as a function of radius, in units of eV.
    density : callable
        Density of the species as a function of radius, in units of particles/m^3.

    """

    species: Species
    temperature: callable  # in units of eV
    density: callable  # in units of particles/m^3

    def v_thermal(self, r: float) -> float:
        """float: Thermal speed, in m/s at a given normalized radius r."""
        T = self.temperature(r) * JOULE_PER_EV
        v_thermal = jnp.sqrt(2 * T / self.species.mass)
        return v_thermal

    def localize(self, r: float) -> LocalMaxwellian:
        """The global distribution function evaluated at a particular radius r."""
        n, dndr = jax.value_and_grad(self.density)(r)
        T, dTdr = jax.value_and_grad(self.temperature)(r)
        return LocalMaxwellian(self.species, T, n, dndr, dTdr)

    def __call__(self, r: float, v: float) -> float:
        return (
            self.density(r)
            / (jnp.sqrt(jnp.pi) * self.v_thermal(r)) ** 3
            * jnp.exp(-(v**2) / self.v_thermal(r) ** 2)
        )


def collisionality(
    maxwellian_a: LocalMaxwellian, v: float, *others: LocalMaxwellian
) -> float:
    """Collisionality between species a and others.

    Parameters
    ----------
    maxwellian_a : LocalMaxwellian
        Distribution function of primary species.
    v : float
        Speed being considered.
    *others : LocalMaxwellian
        Distribution functions for background species colliding with primary.

    Returns
    -------
    nu_a : float
        Collisionality of species a against background of others, in units of 1/s
    """
    nu = 0.0
    for ma in others + (maxwellian_a,):
        nu += nuD_ab(maxwellian_a, ma, v)
    return nu


def nuD_ab(
    maxwellian_a: LocalMaxwellian, maxwellian_b: LocalMaxwellian, v: float
) -> float:
    """Pairwise collision freq. for species a colliding with species b at velocity v.

    Parameters
    ----------
    maxwellian_a : LocalMaxwellian
        Distribution function of primary species.
    maxwellian_b : LocalMaxwellian
        Distribution function of background species.
    v : float
        Speed being considered.

    Returns
    -------
    nu_ab : float
        Collisionality of species a against background of b, in units of 1/s

    """
    nb = maxwellian_b.density
    vtb = maxwellian_b.v_thermal
    prefactor = gamma_ab(maxwellian_a, maxwellian_b, v) * nb / v**3
    erf_part = jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb)
    return prefactor * erf_part


def gamma_ab(
    maxwellian_a: LocalMaxwellian, maxwellian_b: LocalMaxwellian, v: float
) -> float:
    """Prefactor for pairwise collisionality."""
    lnlambda = coulomb_logarithm(maxwellian_a, maxwellian_b)
    ea, eb = maxwellian_a.species.charge, maxwellian_b.species.charge
    ma = maxwellian_a.species.mass
    return ea**2 * eb**2 * lnlambda / (4 * jnp.pi * epsilon_0**2 * ma**2)


def nupar_ab(
    maxwellian_a: LocalMaxwellian, maxwellian_b: LocalMaxwellian, v: float
) -> float:
    """Parallel collisionality."""
    nb = maxwellian_b.density
    vtb = maxwellian_b.v_thermal
    return (
        2 * gamma_ab(maxwellian_a, maxwellian_b, v) * nb / v**3 * chandrasekhar(v / vtb)
    )


def coulomb_logarithm(
    maxwellian_a: LocalMaxwellian, maxwellian_b: LocalMaxwellian
) -> float:
    """Coulomb logarithm for collisions between species a and b.

    Parameters
    ----------
    maxwellian_a : LocalMaxwellian
        Distribution function of primary species.
    maxwellian_b : LocalMaxwellian
        Distribution function of background species.

    Returns
    -------
    log(lambda) : float

    """
    bmin, bmax = impact_parameter(maxwellian_a, maxwellian_b)
    return jnp.log(bmax / bmin)


def impact_parameter(
    maxwellian_a: LocalMaxwellian, maxwellian_b: LocalMaxwellian
) -> float:
    """Impact parameters for classical Coulomb collision."""
    bmin = jnp.maximum(
        impact_parameter_perp(maxwellian_a, maxwellian_b),
        debroglie_length(maxwellian_a, maxwellian_b),
    )
    bmax = debye_length(maxwellian_a, maxwellian_b)
    return bmin, bmax


def impact_parameter_perp(
    maxwellian_a: LocalMaxwellian, maxwellian_b: LocalMaxwellian
) -> float:
    """Distance of the closest approach for a 90° Coulomb collision."""
    m_reduced = (
        maxwellian_a.species.mass
        * maxwellian_b.species.mass
        / (maxwellian_a.species.mass + maxwellian_b.species.mass)
    )
    v_th = jnp.sqrt(maxwellian_a.v_thermal * maxwellian_b.v_thermal)
    return (
        maxwellian_a.species.charge
        * maxwellian_a.species.charge
        / (4 * jnp.pi * epsilon_0 * m_reduced * v_th**2)
    )


def debroglie_length(
    maxwellian_a: LocalMaxwellian, maxwellian_b: LocalMaxwellian
) -> float:
    """Thermal DeBroglie wavelength."""
    m_reduced = (
        maxwellian_a.species.mass
        * maxwellian_b.species.mass
        / (maxwellian_a.species.mass + maxwellian_b.species.mass)
    )
    v_th = jnp.sqrt(maxwellian_a.v_thermal * maxwellian_b.v_thermal)
    return hbar / (2 * m_reduced * v_th)


def debye_length(*maxwellians: LocalMaxwellian) -> float:
    """Scale length for charge screening."""
    den = 0
    for m in maxwellians:
        den += m.density / (m.temperature * JOULE_PER_EV) * m.species.charge**2
    return jnp.sqrt(epsilon_0 / den)


def chandrasekhar(x: jax.Array) -> jax.Array:
    """Chandrasekhar function."""
    return (
        jax.scipy.special.erf(x) - 2 * x / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))
    ) / (2 * x**2)


def _dchandrasekhar(x):
    return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2)) - 2 / x * chandrasekhar(x)
