"""Solution objects and computation of output moments."""

import re
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import elementary_charge, epsilon_0, proton_mass

from ._misc import (
    _d3v,
    _dke_thermodynamic_forces,
    _dr,
    angular_momentum_drift,
    radial_drift,
)
from .field import Field
from .rotation import RotatingEquilibrium
from .species import (
    _JOULE_PER_EV,
    LocalMaxwellian,
    _coulomb_logarithm,
    _species_pairs,
)
from .velocity_grids import UniformPitchAngleGrid, _AbstractSpeedGrid

_MDKE_OUTPUTS = {}
_DKE_OUTPUTS = {}

_DKE_DEFAULT_OUTPUT_QTYS = (
    "<heat_flux>",
    "<particle_flux>",
    "<V||B>",
    "<J||B>",
    "J_rho",
)

_SUPERSCRIPT_DIGITS = str.maketrans("0123456789-+=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁼⁽⁾")


def _clean_units(s: str) -> str:
    r"""Render a LaTeX units string from ``_DKE_OUTPUTS`` as plain unicode.

    For example ``"kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}"`` becomes
    ``"kg·m⁻¹·s⁻³ = W·m⁻³"``. Returns the empty string for ``"None"`` or empty
    input.
    """
    if not s or s == "None":
        return ""
    out = s.replace("\\cdot", "·")
    out = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", out)
    out = re.sub(
        r"\^\{([^}]*)\}",
        lambda m: m.group(1).translate(_SUPERSCRIPT_DIGITS),
        out,
    )
    out = re.sub(
        r"\^(-?\d)",
        lambda m: m.group(1).translate(_SUPERSCRIPT_DIGITS),
        out,
    )
    out = re.sub(r"\s*·\s*", "·", out)
    return out.strip()


def _register_mdke_output(
    name: str,
    label: str,
    units: str,
    description: str,
    dim: tuple[int | str, ...],
) -> Callable[[Callable], Callable]:
    """Decorator to wrap a function and add it to the list of things we can compute.

    Parameters
    ----------
    name : str
        Name of the quantity. This will be used as the key used to compute the
        quantity in `get` and its name in the data dictionary.
    label : str
        Title of the quantity in LaTeX format.
    units : str
        Units of the quantity in LaTeX format.
    description : str
        Description of the quantity.
    dim : tuple of int or str
        Shape of the output array, as sizes or names of grid dimensions.
    """

    def _decorator(func):
        d = {
            "label": label,
            "units": units,
            "description": description,
            "fun": func,
            "dim": dim,
        }
        _MDKE_OUTPUTS[name] = d
        return func

    return _decorator


def _register_dke_output(
    name: str,
    label: str,
    units: str,
    description: str,
    dim: tuple[int | str, ...],
) -> Callable[[Callable], Callable]:
    """Decorator to wrap a function and add it to the list of things we can compute.

    Parameters
    ----------
    name : str
        Name of the quantity. This will be used as the key used to compute the
        quantity in `get` and its name in the data dictionary.
    label : str
        Title of the quantity in LaTeX format.
    units : str
        Units of the quantity in LaTeX format.
    description : str
        Description of the quantity.
    dim : tuple of int or str
        Shape of the output array, as sizes or names of grid dimensions.
    """

    def _decorator(func):
        d = {
            "label": label,
            "units": units,
            "description": description,
            "fun": func,
            "dim": dim,
        }
        _DKE_OUTPUTS[name] = d
        return func

    return _decorator


class DKESolution(eqx.Module):
    """Solution to Drift Kinetic Equation.

    Attributes
    ----------
    F0 : jax.Array, shape (ns, nx, 1, 1, 1)
        Maxwellian background distribution, broadcastable against ``f1``. With
        rotation it is that of the local density, shape (ns, nx, 1, nt, nz).
    f1 : jax.Array, shape (ns, nx, na, nt, nz)
        Perturbation solved for by the DKE.
    rhs : jax.Array, shape (ns, nx, na, nt, nz)
        Drive terms for DKE
    field : Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
    EparB : float
        <E||B>, flux surface average of parallel electric field times B.
    background : list[LocalMaxwellian]
        Additional background species to include in the collision operator without
        solving for df.
    rotating : RotatingEquilibrium, optional
        Lowest order state of a toroidally rotating plasma. Velocities are then those
        in the rotating frame, and the fluxes are those of the radial drift including
        the ExB, centrifugal and Coriolis drifts.
    f : jax.Array, shape (ns, nx, na, nt, nz)
        Full distribution function ``F0 + f1``. Computed on access.
    f1_krylov : tuple of jax.Array
        Distribution function and source terms as seen by the Krylov solver, as a
        tuple of ``f1`` flattened, shape (ns*nx*na*nt*nz,), and the (particle, heat)
        source of each species, shape (2*ns,). Computed on access.
    """

    F0: jax.Array
    f1: jax.Array
    rhs: jax.Array
    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: _AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: jax.Array
    EparB: jax.Array
    background: list[LocalMaxwellian]
    rotating: RotatingEquilibrium | None
    _particle_source: jax.Array
    _heat_source: jax.Array

    def __init__(
        self,
        F0: jax.Array,
        f1: jax.Array,
        rhs: jax.Array,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: _AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: jax.Array,
        EparB: jax.Array,
        background: list[LocalMaxwellian],
        rotating: RotatingEquilibrium | None = None,
    ):
        ns = len(species)
        shape = (ns, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta)
        N = np.prod(shape)
        if isinstance(f1, (tuple, list)):
            # (f1, sources) parts, as used by the Krylov solver
            f1, sources = f1
        else:
            f1 = f1.flatten()
            sources = f1[N:] if f1.size == N + 2 * ns else None
            f1 = f1[:N] if f1.size == N + 2 * ns else f1
        if f1.size != N or (sources is not None and sources.size != 2 * ns):
            raise ValueError("got wrong size for f1")
        f1 = f1.reshape(shape)
        if sources is None:
            particle_source = jnp.full(ns, jnp.nan)
            heat_source = jnp.full(ns, jnp.nan)
        else:
            # one (particle, heat) pair per species, matching DKESources columns
            sources = sources.reshape((ns, 2))
            particle_source = sources[:, 0]
            heat_source = sources[:, 1]

        self.F0 = jnp.asarray(F0).reshape(ns, speedgrid.nx, 1, 1, 1)
        if rotating is not None:
            self.F0 = self.F0 * rotating.density[:, None, None]
        self.rotating = rotating
        self.f1 = f1
        self.rhs = rhs.flatten()[:N].reshape(shape)
        self._particle_source = particle_source
        self._heat_source = heat_source
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.background = background
        self.Erho = Erho
        self.EparB = EparB

    @property
    def f(self) -> jax.Array:
        """Full distribution function ``F0 + f1``."""
        return self.F0 + self.f1

    @property
    def f1_krylov(self) -> tuple[jax.Array, jax.Array]:
        """Distribution function and source terms, as seen by krylov solver."""
        f1 = self.f1.flatten()
        sources = jnp.stack([self._particle_source, self._heat_source], axis=1)
        sources = jnp.nan_to_num(sources, nan=0.0).flatten()
        return (f1, sources)

    def get(self, qty: str, **kwargs) -> jax.Array:
        """Compute desired moments of the solution.

        Parameters
        ----------
        qty : str
            Quantity to compute, one of ``qtys_list()``.

        Returns
        -------
        qty : jax.Array
            Desired output quantity as an array.

        """
        assert qty in _DKE_OUTPUTS.keys()
        return _DKE_OUTPUTS[qty]["fun"](self, **kwargs)

    def qtys_list(self) -> list[str]:
        """List of all computable output quantities."""
        return list(_DKE_OUTPUTS.keys())

    def print_summary(self, qtys: tuple[str, ...] = _DKE_DEFAULT_OUTPUT_QTYS) -> None:
        """Print headline output moments with units.

        Per-species quantities (shape ``(ns,)``) render as
        ``[ v0 v1 ... ] (per species, units)``; scalar quantities (sums over
        species, e.g. ``<J||B>`` and ``J_rho``) render as ``v (units)``.
        """
        ns = len(self.species)
        width = max(len(q) for q in qtys)
        for qty in qtys:
            vals = self.get(qty)
            units = _clean_units(_DKE_OUTPUTS[qty].get("units", ""))
            if jnp.ndim(vals) == 0:
                suffix = f" ({units})" if units else ""
                s = f"{qty:<{width}s}: " + "{: .3e}" + suffix
                jax.debug.print(s, vals, ordered=True)
            else:
                suffix = f" (per species, {units})" if units else " (per species)"
                s = f"{qty:<{width}s}: [" + "{: .3e} " * ns + "]" + suffix
                jax.debug.print(s, *vals, ordered=True)


class MDKESolution(eqx.Module):
    """Solution to Monoenergetic Drift Kinetic Equation.

    Attributes
    ----------
    f : jax.Array, shape (3, na, nt, nz)
        Solution of the MDKE for each drive term
    rhs : jax.Array, shape (3, na, nt, nz)
        Drive terms for MDKE
    field : Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v = -∂Φ /∂ρ /v in units of V*s/m.
    nuhat : float
        Monoenergetic collisionality, ν/v in units of 1/m.
    """

    f: jax.Array
    rhs: jax.Array
    field: Field
    pitchgrid: UniformPitchAngleGrid
    nuhat: jax.Array
    erhohat: jax.Array

    def __init__(
        self,
        f: jax.Array,
        rhs: jax.Array,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        nuhat: jax.Array,
        erhohat: jax.Array,
    ):
        self.f = f.reshape(3, pitchgrid.nalpha, field.ntheta, field.nzeta)
        self.rhs = rhs.reshape(3, pitchgrid.nalpha, field.ntheta, field.nzeta)
        self.field = field
        self.pitchgrid = pitchgrid
        self.nuhat = nuhat
        self.erhohat = erhohat

    def get(self, qty: str, **kwargs) -> jax.Array:
        """Compute desired moments of the solution.

        Parameters
        ----------
        qty : str
            Quantity to compute. Currently only "Dij" is supported, to return the
            monoenergetic transport coefficients.

        Returns
        -------
        qty : jax.Array
            Desired output quantity as an array.

        """
        # this is kind of a dummy API for now, but future proofing to if we want
        # to compute more output qtys, rather than keep adding to a dict and possibly
        # computing a lot of wasted stuff we do it on the fly as needed.
        assert qty in _MDKE_OUTPUTS.keys()
        return _MDKE_OUTPUTS[qty]["fun"](self, **kwargs)

    def qtys_list(self) -> list[str]:
        """List of all computable output quantities."""
        return list(_MDKE_OUTPUTS.keys())


@_register_mdke_output(
    name="Dij",
    label="$D_{ij}$",
    units="\\mathrm{Various}",
    description="Monoenergetic transport coefficient matrix",
    dim=(3, 3),
)
def _mdke_Dij(sol, normalization=None, **kwargs):
    """Monoenergetic transport coefficients."""
    f = sol.f.reshape((3, -1))
    s = sol.rhs.reshape((3, -1))
    na, nt, nz = (
        sol.pitchgrid.nalpha,
        sol.field.ntheta,
        sol.field.nzeta,
    )
    sf = s[:, None] * f[None, :]  # shape (3,3,N)
    Dij_itz = sf.reshape((3, 3, na, nt, nz))
    Dij_i = sol.field.flux_surface_average(Dij_itz)  # shape (3,3,na)
    Dij = jnp.sum(Dij_i * sol.pitchgrid.wxi, axis=-1)  # shape (3,3)
    return Dij


@_register_mdke_output(
    name="Dij_DKES",
    label="$D_{ij,DKES}$",
    units="\\mathrm{Various}",
    description="Monoenergetic transport coefficient matrix, scaled to match DKES.",
    dim=(3, 3),
)
def _mdke_Dij_dkes(sol, **kwargs):
    """Monoenergetic transport coefficients."""
    Dij = sol.get("Dij")
    sgn = jnp.sign(sol.field.Psi)
    a = sol.field.a_minor
    sa = sgn * a
    B0 = sol.field.B0
    scale = jnp.array(
        [
            [a**2, sa, sa / B0],
            [sa, sa, sa / B0],
            [sa / B0, sa / B0, 1 / B0**2],
        ]
    )
    return Dij * scale


@_register_dke_output(
    name="<particle_flux>",
    label="\\Gamma = \\langle \\int d^3v f_s \\mathbf{v}_m \\cdot "
    "\\nabla \\rho \\rangle",
    units="m^{-3} \\cdot s^{-1}",
    description="Particle flux for each species. With rotation, the flux of the "
    "radial drift in the rotating frame, here and in the other fluxes.",
    dim=("ns",),
)
def _dke_particle_flux(sol, **kwargs):
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    dr = _dr(sol.field)[None, None, None]
    dr = dr / dr.sum()

    drift = radial_drift(
        sol.field, sol.speedgrid, sol.pitchgrid, sol.species, sol.rotating
    )

    particle_flux = sol.f * drift * d3v * dr
    particle_flux = particle_flux.sum(axis=(-4, -3, -2, -1))
    return particle_flux


@_register_dke_output(
    name="<heat_flux>",
    label="Q = \\langle \\int d^3v 1/2 m v^2 f_s \\mathbf{v}_m "
    "\\cdot \\nabla \\rho \\rangle",
    units="kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}",
    description="Heat flux for each species. With rotation, the energy carried is "
    "the kinetic energy in the rotating frame plus the potential energy relative "
    "to theta=0.",
    dim=("ns",),
)
def _dke_heat_flux(sol, **kwargs):
    vth = jnp.array([sp.v_thermal for sp in sol.species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in sol.species])[:, None, None, None, None]
    x = sol.speedgrid.x[None, :, None, None, None]
    v = vth * x
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    dr = _dr(sol.field)[None, None, None]
    dr = dr / dr.sum()

    drift = radial_drift(
        sol.field, sol.speedgrid, sol.pitchgrid, sol.species, sol.rotating
    )
    energy = ms * v**2 / 2
    if sol.rotating is not None:
        # potential energy in the rotating frame, relative to θ=0
        Ts = jnp.array([sp.temperature for sp in sol.species]) * _JOULE_PER_EV
        energy = energy + (Ts[:, None, None] * sol.rotating.lam)[:, None, None]
    heat_flux = sol.f * energy * drift * d3v * dr
    heat_flux = heat_flux.sum(axis=(-4, -3, -2, -1))
    return heat_flux


@_register_dke_output(
    name="<momentum_flux>",
    label="\\langle \\int d^3v m_s v_{||} B f_s \\mathbf{v}_m \\cdot "
    "\\nabla \\rho \\rangle",
    units="kg \\cdot T \\cdot m^{-2} \\cdot s^{-2}",
    description="Flux of parallel momentum times field strength for each species",
    dim=("ns",),
)
def _dke_momentum_flux(sol, **kwargs):
    vth = jnp.array([sp.v_thermal for sp in sol.species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in sol.species])[:, None, None, None, None]
    xi = sol.pitchgrid.xi[None, None, :, None, None]
    x = sol.speedgrid.x[None, :, None, None, None]
    vpar = x * vth * xi
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    dr = _dr(sol.field)[None, None, None]
    dr = dr / dr.sum()

    drift = radial_drift(
        sol.field, sol.speedgrid, sol.pitchgrid, sol.species, sol.rotating
    )
    momentum_flux = ms * vpar * sol.field.Bmag * sol.f * drift * d3v * dr
    momentum_flux = momentum_flux.sum(axis=(-4, -3, -2, -1))
    return momentum_flux


@_register_dke_output(
    name="V||",
    label="V_{||} = 1/n_s \\int d^3v v_{||} f_s",
    units="m \\cdot s^{-1}",
    description="Parallel flow on surface for each species.",
    dim=("ns", "nt", "nz"),
)
def _dke_Vpar(sol, **kwargs):
    vth = jnp.array([sp.v_thermal for sp in sol.species])[:, None, None, None, None]
    ns = _density(sol)[:, None, None]
    xi = sol.pitchgrid.xi[None, None, :, None, None]
    x = sol.speedgrid.x[None, :, None, None, None]
    vpar = x * vth * xi
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    dr = _dr(sol.field)[None, None, None]
    dr = dr / dr.sum()

    Vpar = 1 / ns * vpar * sol.f * d3v
    Vpar = Vpar.sum(axis=(1, 2))
    return Vpar


@_register_dke_output(
    name="<V||B>",
    label="V_{||} B = 1/n_s \\langle B \\int d^3v v_{||} f_s \\rangle",
    units="T \\cdot m \\cdot s^{-1}",
    description="Flux surface average field*parallel velocity for each species,",
    dim=("ns",),
)
def _dke_VparB(sol, **kwargs):
    dr = _dr(sol.field)
    dr = dr / dr.sum()
    Vpar = sol.get("V||")
    BVpar = sol.field.Bmag[:, :] * Vpar * dr
    BVpar = BVpar.sum(axis=(-2, -1))
    return BVpar


def _density(sol):
    """Lowest order density of each species, shape (ns, 1, 1) or (ns, nt, nz)."""
    ns = jnp.array([sp.density for sp in sol.species])[:, None, None]
    if sol.rotating is not None:
        ns = ns * sol.rotating.density
    return ns


def _perpendicular_rotation(sol):
    """Coefficient of B x grad(rho) / B^2 in the perpendicular flow, shape (ns, nt, nz).

    The perpendicular (diamagnetic and ExB) flow is
    V_perp = B x grad(rho) / B^2 * (1/(q n) dp/drho + dPhi/drho).
    """
    if sol.rotating is not None:
        raise NotImplementedError("Perpendicular flows are not available with rotation")
    # The density is a pseudo-density times the Boltzmann factor in Phi_1. The
    # radial derivative of Phi_1 enters both the Boltzmann factor and the ExB
    # drift, and cancels between them.
    qs = jnp.array([sp.species.charge for sp in sol.species])[:, None, None]
    Ts = jnp.array([sp.temperature for sp in sol.species])[:, None, None]
    dTs = jnp.array([sp.dTdrho for sp in sol.species])[:, None, None]
    ns = jnp.array([sp.density for sp in sol.species])[:, None, None]
    dns = jnp.array([sp.dndrho for sp in sol.species])[:, None, None]
    Ts, dTs = Ts * _JOULE_PER_EV, dTs * _JOULE_PER_EV
    Phi1 = sol.get("Phi_1")
    return Ts / qs * dns / ns + (1 + qs * Phi1 / Ts) * dTs / qs - sol.Erho


@_register_dke_output(
    name="Vperp",
    label="V_{\\perp} = \\frac{|\\nabla \\rho|}{B} \\left(\\frac{1}{q_s n_s} "
    "\\frac{\\partial p_s}{\\partial \\rho} + \\frac{\\partial \\Phi}"
    "{\\partial \\rho}\\right)",
    units="m \\cdot s^{-1}",
    description="Perpendicular (diamagnetic and ExB) flow on surface for each "
    "species, along b x grad(rho), including Phi_1. Requires Field.g_sup_rr.",
    dim=("ns", "nt", "nz"),
)
def _dke_Vperp(sol, **kwargs):
    if sol.field.g_sup_rr is None:
        raise ValueError(
            "Vperp requires Field.g_sup_rr, which is not available for this field."
        )
    return jnp.sqrt(sol.field.g_sup_rr) / sol.field.Bmag * _perpendicular_rotation(sol)


@_register_dke_output(
    name="V^theta",
    label="V^{\\theta} = V_{||} B^{\\theta} / B + \\mathbf{V}_{\\perp} \\cdot "
    "\\nabla \\theta",
    units="s^{-1}",
    description="Contravariant poloidal component of the total flow for each "
    "species, including Phi_1.",
    dim=("ns", "nt", "nz"),
)
def _dke_V_theta(sol, **kwargs):
    field = sol.field
    # (B x grad(rho)) . grad(theta) = B_zeta / sqrt(g)
    Vperp = field.B_sub_z / (field.Bmag**2 * field.sqrtg) * _perpendicular_rotation(sol)
    return sol.get("V||") * field.B_sup_t / field.Bmag + Vperp


@_register_dke_output(
    name="V^zeta",
    label="V^{\\zeta} = V_{||} B^{\\zeta} / B + \\mathbf{V}_{\\perp} \\cdot "
    "\\nabla \\zeta",
    units="s^{-1}",
    description="Contravariant toroidal component of the total flow for each "
    "species, including Phi_1.",
    dim=("ns", "nt", "nz"),
)
def _dke_V_zeta(sol, **kwargs):
    field = sol.field
    # (B x grad(rho)) . grad(zeta) = -B_theta / sqrt(g)
    Vperp = (
        -field.B_sub_t / (field.Bmag**2 * field.sqrtg) * _perpendicular_rotation(sol)
    )
    return sol.get("V||") * field.B_sup_z / field.Bmag + Vperp


@_register_dke_output(
    name="<J||B>",
    label="J_{||}B = \\sum_s q_s/n_s \\langle B \\int d^3v v_{||} f_s \\rangle",
    units="A \\cdot T \\cdot m^{-3}",
    description="Bootstrap current",
    dim=(),
)
def _dke_bootstrap_current(sol, **kwargs):
    if sol.rotating is not None:
        # the densities vary on the surface
        return sol.field.flux_surface_average(sol.field.Bmag * sol.get("J||"))
    ns = jnp.array([sp.density for sp in sol.species])
    qs = jnp.array([sp.species.charge for sp in sol.species])
    BVpar = sol.get("<V||B>")
    bootstrap_current = ns * qs * BVpar
    bootstrap_current = bootstrap_current.sum(axis=(-1))
    return bootstrap_current


@_register_dke_output(
    name="J_rho",
    label="J_{\\rho} = \\sum_s q_s \\Gamma_s",
    units="A \\cdot m^{-3}",
    description="Radial current",
    dim=(),
)
def _dke_radial_current(sol, **kwargs):
    qs = jnp.array([sp.species.charge for sp in sol.species])
    particle_flux = sol.get("<particle_flux>")
    radial_current = particle_flux * qs
    radial_current = radial_current.sum(axis=(-1))
    return radial_current


@_register_dke_output(
    name="J||",
    label="J_{||} = \\sum_s q_s V_{||,s}",
    units="A \\cdot m^{-3}",
    description="Parallel current",
    dim=("nt", "nz"),
)
def _dke_parallel_current(sol, **kwargs):
    ns = _density(sol)
    qs = jnp.array([sp.species.charge for sp in sol.species])[:, None, None]
    Vpar = sol.get("V||")
    Jpar = qs * ns * Vpar
    Jpar = Jpar.sum(axis=(-3))
    return Jpar


def _velocity_moment(sol, f, weight):
    """Integrate weight * f over velocity space at each point on the surface."""
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    return (weight * f * d3v).sum(axis=(1, 2))


def _pressure_weight(sol):
    vth = jnp.array([sp.v_thermal for sp in sol.species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in sol.species])[:, None, None, None, None]
    x = sol.speedgrid.x[None, :, None, None, None]
    return ms * (vth * x) ** 2 / 3


def _boltzmann_factor(sol):
    """Boltzmann response exp(-q_s Phi_1 / T_s), shape (ns, nt, nz)."""
    qs = jnp.array([sp.species.charge for sp in sol.species])[:, None, None]
    Ts = jnp.array([sp.temperature for sp in sol.species])[:, None, None]
    return jnp.exp(-qs * sol.get("Phi_1") / (Ts * _JOULE_PER_EV))


@_register_dke_output(
    name="n1",
    label="\\delta n_s = \\int d^3v f_{s1}",
    units="m^{-3}",
    description="Density perturbation on surface for each species.",
    dim=("ns", "nt", "nz"),
)
def _dke_density_perturbation(sol, **kwargs):
    return _velocity_moment(sol, sol.f1, 1.0)


@_register_dke_output(
    name="n",
    label="n_s = e^{-q_s \\Phi_1 / T_s} \\int d^3v F_{s0} + \\delta n_s",
    units="m^{-3}",
    description="Total density on surface for each species, including the "
    "Boltzmann response to Phi_1.",
    dim=("ns", "nt", "nz"),
)
def _dke_density(sol, **kwargs):
    n0 = _velocity_moment(sol, sol.F0, 1.0)
    return _boltzmann_factor(sol) * n0 + sol.get("n1")


@_register_dke_output(
    name="p1",
    label="\\delta p_s = \\int d^3v 1/3 m_s v^2 f_{s1}",
    units="kg \\cdot m^{-1} \\cdot s^{-2} = Pa",
    description="Pressure perturbation on surface for each species.",
    dim=("ns", "nt", "nz"),
)
def _dke_pressure_perturbation(sol, **kwargs):
    return _velocity_moment(sol, sol.f1, _pressure_weight(sol))


@_register_dke_output(
    name="p",
    label="p_s = e^{-q_s \\Phi_1 / T_s} \\int d^3v 1/3 m_s v^2 F_{s0} + \\delta p_s",
    units="kg \\cdot m^{-1} \\cdot s^{-2} = Pa",
    description="Total pressure on surface for each species, including the "
    "Boltzmann response to Phi_1.",
    dim=("ns", "nt", "nz"),
)
def _dke_pressure(sol, **kwargs):
    p0 = _velocity_moment(sol, sol.F0, _pressure_weight(sol))
    return _boltzmann_factor(sol) * p0 + sol.get("p1")


@_register_dke_output(
    name="Phi_1",
    label="\\Phi_1 = \\sum_s q_s \\delta n_s / \\sum_s (q_s^2 n_s / T_s)",
    units="V",
    description="Variation of electrostatic potential on surface from "
    "quasi-neutrality, with zero flux surface average.",
    dim=("nt", "nz"),
)
def _dke_Phi1(sol, **kwargs):
    # Quasi-neutrality with a linearized Boltzmann response to Phi_1 from every
    # species, including background species which have no kinetic perturbation.
    # The flux surface average of Phi_1 is absorbed into the equilibrium potential,
    # so only the variation of the charge density on the surface is kept.
    qs = jnp.array([sp.species.charge for sp in sol.species])[:, None, None]
    charge = (qs * sol.get("n1")).sum(axis=0)
    charge = charge - sol.field.flux_surface_average(charge)
    all_species = list(sol.species) + list(sol.background)
    density = jnp.ones((len(all_species), 1, 1))
    if sol.rotating is not None:
        density = sol.rotating.field_density
    shielding = sum(
        sp.species.charge**2 * sp.density * n / (sp.temperature * _JOULE_PER_EV)
        for sp, n in zip(all_species, density)
    )
    return charge / shielding


def _classical_fluxes(sol, coulomb_log=None):
    """Classical particle and heat fluxes for each species.

    Friction is evaluated with the Braginskii matrix elements for collisions with
    all species, including background species. Species densities are the
    pseudo-densities of Maxwell-Boltzmann distributions in Phi_1.
    """
    # Eqs. (3.9)-(3.10) of S. Buller et al., J. Plasma Phys. 85, 175850401 (2019),
    # using the linearized Fokker-Planck operator for arbitrary masses and
    # temperatures. The gyrophase dependent part of f_1 that carries
    # the classical flux is -rho.grad(F_0), set by the gradients of the
    # Maxwellians alone, so the friction is independent of the drift kinetic
    # solution. That function lies exactly in the span of the first two Laguerre
    # polynomials, so the Braginskii matrix elements give the friction exactly.
    if sol.rotating is not None:
        raise NotImplementedError("Classical fluxes are not available with rotation")
    if sol.field.g_sup_rr is None:
        raise ValueError(
            "Classical fluxes require Field.g_sup_rr, which is not available "
            "for this field."
        )
    g = sol.field.g_sup_rr / sol.field.Bmag**2
    Phi1 = sol.get("Phi_1")

    def fluxes_ab(a, b):
        ma, qa, na = a.species.mass, a.species.charge, a.density
        mb, qb, nb = b.species.mass, b.species.charge, b.density
        Ta = a.temperature * _JOULE_PER_EV
        Tb = b.temperature * _JOULE_PER_EV
        dTa = a.dTdrho * _JOULE_PER_EV
        dTb = b.dTdrho * _JOULE_PER_EV
        lnlambda = _coulomb_logarithm(a, b) if coulomb_log is None else coulomb_log
        xab2 = ma * Tb / (mb * Ta)
        norm = (1 + xab2) ** 2.5
        M00 = -(1 + ma / mb) * (1 + xab2) / norm
        M01 = -1.5 * (1 + ma / mb) / norm
        M11 = -(13 + 16 * xab2 + 30 * xab2**2) / 4 / norm
        N11 = 27 * ma / (4 * mb) / norm
        # Boltzmann factors of both species weight the friction on the surface.
        # The radial derivative of the variation in the Boltzmann factor gives
        # the extra Phi_1 * dlnT term.
        weight = jnp.exp(-(qa / Ta + qb / Tb) * Phi1)
        geometry1 = sol.field.flux_surface_average(g * weight)
        geometry2 = sol.field.flux_surface_average(g * weight * Phi1)
        dp = Ta * a.dndrho / (na * qa) - Tb * b.dndrho / (nb * qb)
        dlnT = dTa / Ta - dTb / Tb
        coeff = lnlambda * qa * qb**2 * na * nb
        Gamma = coeff * (
            geometry1 * M00 * dp
            + geometry2 * M00 * dlnT
            + geometry1 * (M00 - M01) * dTa / qa
            - geometry1 * (M00 - xab2 * M01) * dTb / qb
        )
        q = coeff * (
            geometry1 * M01 * dp
            + geometry2 * M01 * dlnT
            + geometry1 * (M01 - M11) * dTa / qa
            - geometry1 * (M01 + N11) * dTb / qb
        )
        return Gamma, q

    Gamma, q = _species_pairs(
        fluxes_ab, list(sol.species), list(sol.species) + list(sol.background)
    )
    Gamma, q = Gamma.sum(axis=1), q.sum(axis=1)
    ma = jnp.array([sp.species.mass for sp in sol.species])
    Ta = jnp.array([sp.temperature for sp in sol.species]) * _JOULE_PER_EV
    prefactor = (
        4 * jnp.sqrt(2 * jnp.pi) * jnp.sqrt(ma) / (3 * (4 * jnp.pi * epsilon_0) ** 2)
    ) / Ta**1.5
    Gamma = prefactor * Gamma
    # total energy flux Q = q + 5/2 T Gamma, matching <heat_flux>
    Q = -prefactor * Ta * q + 2.5 * Ta * Gamma
    return Gamma, Q


@_register_dke_output(
    name="<classical_particle_flux>",
    label="\\Gamma^{C} = \\frac{1}{q_s} \\langle \\frac{\\mathbf{B} \\times "
    "\\nabla \\rho}{B^2} \\cdot \\mathbf{R}_s \\rangle",
    units="m^{-3} \\cdot s^{-1}",
    description="Classical particle flux for each species, including Phi_1. "
    "Requires Field.g_sup_rr. Accepts coulomb_log to override the Coulomb "
    "logarithm.",
    dim=("ns",),
)
def _dke_classical_particle_flux(sol, coulomb_log=None, **kwargs):
    return _classical_fluxes(sol, coulomb_log)[0]


@_register_dke_output(
    name="<classical_heat_flux>",
    label="Q^{C} = \\frac{1}{q_s} \\langle \\frac{\\mathbf{B} \\times "
    "\\nabla \\rho}{B^2} \\cdot \\mathbf{G}_s \\rangle",
    units="kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}",
    description="Classical heat flux for each species, including Phi_1. "
    "Requires Field.g_sup_rr. Accepts coulomb_log to override the Coulomb "
    "logarithm.",
    dim=("ns",),
)
def _dke_classical_heat_flux(sol, coulomb_log=None, **kwargs):
    return _classical_fluxes(sol, coulomb_log)[1]


@_register_dke_output(
    name="<angular_momentum_flux>",
    label="\\Pi = \\langle \\int d^3v f_s m_s R V_\\zeta \\mathbf{v}_d "
    "\\cdot \\nabla \\rho \\rangle",
    units="kg \\cdot m^{-1} \\cdot s^{-2}",
    description="Flux of toroidal angular momentum for each species, conjugate to "
    "the rotation shear. Requires an axisymmetric field with R.",
    dim=("ns",),
)
def _dke_angular_momentum_flux(sol, **kwargs):
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    dr = _dr(sol.field)[None, None, None]
    dr = dr / dr.sum()
    momentum = angular_momentum_drift(
        sol.field, sol.speedgrid, sol.pitchgrid, sol.species, sol.rotating
    )
    return (sol.f * momentum * d3v * dr).sum(axis=(-4, -3, -2, -1))


def _energy_flux(sol, particle_flux, heat_flux, angular_momentum_flux, frame):
    if frame not in ("rotating", "lab"):
        raise ValueError(f"frame must be 'rotating' or 'lab', got '{frame}'")
    if sol.rotating is None:
        return heat_flux
    ms = jnp.array([sp.species.mass for sp in sol.species])
    qs = jnp.array([sp.species.charge for sp in sol.species])
    omega = sol.rotating.omega
    # potential energy at θ=0
    Phi = sol.field.flux_surface_average(sol.rotating.Phi)
    energy = -qs * Phi - ms * omega**2 * sol.field.R[0, 0] ** 2 / 2
    energy_flux = heat_flux + energy * particle_flux
    if frame == "lab":
        energy_flux = energy_flux + omega * angular_momentum_flux
    return energy_flux


@_register_dke_output(
    name="<energy_flux>",
    label="\\langle \\int d^3v f_s [1/2 m_s v^2 + q_s (\\Phi_0 - \\langle "
    "\\Phi_0 \\rangle) - 1/2 m_s \\omega^2 R^2] \\mathbf{v}_d \\cdot "
    "\\nabla \\rho \\rangle",
    units="kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}",
    description="Energy flux for each species in a rotating plasma, with the "
    "potential energy measured from the flux surface average of the potential "
    "and the axis of rotation. ``frame='lab'`` gives the flux of the kinetic "
    "energy in the lab frame instead of the rotating frame, which adds omega "
    "times the angular momentum flux. Equal to <heat_flux> without rotation.",
    dim=("ns",),
)
def _dke_energy_flux(sol, frame="rotating", **kwargs):
    momentum = sol.get("<angular_momentum_flux>") if frame == "lab" else 0.0
    return _energy_flux(
        sol, sol.get("<particle_flux>"), sol.get("<heat_flux>"), momentum, frame
    )


def _nondissipative_fluxes(sol):
    """Fluxes of particles, heat and angular momentum that are independent of f1.

    The drive by the rotation shear is -(ω'/T) F dπ/dt = -L[F G] with G = (ω'/T) G₂,
    G₂ = m²/(2 q ψ') [(I v||/B + ω R²)² + μ R² B_p²/(m B)], and L the collisionless
    operator, so f1 contains -F G. The physical fluxes are moments of the collision
    operator, which annihilates F G only to lowest order, and written as moments of
    f1 they are those of the solution plus the moments of the Maxwellian

    Γ = (ω'/T) <∫d³v F G₂ dρ/dt>, Q = (ω'/T) <∫d³v F G₂ (m v²/2 + T λ) dρ/dt>,
    Π = -<∫d³v F [A₁ + (x² + λ) A₂] G₂ dρ/dt>

    These vanish on an up-down symmetric surface.
    """
    ns = len(sol.species)
    if sol.rotating is None:
        return jnp.zeros((3, ns))
    field, rot = sol.field, sol.rotating
    # the velocity integrals are polynomials times the Maxwellian, done exactly with
    # Gauss-Hermite in v|| and Gauss-Laguerre in v_perp²
    s, ws = np.polynomial.hermite.hermgauss(8)
    y, wy = np.polynomial.laguerre.laggauss(5)
    s, ws = s[:, None, None, None], ws[:, None, None, None]
    y, wy = y[None, :, None, None], wy[None, :, None, None]

    def flux(sp, lam, dlamdt, density, A1, A2):
        m, q, vth = sp.species.mass, sp.species.charge, sp.v_thermal
        T = sp.temperature * _JOULE_PER_EV
        B, I, R = field.Bmag, field.B_sub_z, field.R
        u, w2 = vth * s, vth**2 * y
        drift = -m / q * (2 * u**2 + w2) / (2 * B**3) * field.BxgradrhodotgradB + (
            -I * T / q * dlamdt / B + 2 * m / q * rot.omega * u * R * field._dfdt(R)
        ) / (field.sqrtg * B)
        G2 = (
            m**2
            / (2 * q * field.sqrtg * field.B_sup_t)
            * ((I * u / B + rot.omega * R**2) ** 2 + w2 * (R**2 - I**2 / B**2) / 2)
        )
        x2 = s**2 + y
        weight = sp.density * density / jnp.sqrt(jnp.pi) * ws * wy
        g = field.flux_surface_average((weight * G2 * drift).sum(axis=(0, 1)))
        gx = field.flux_surface_average(
            (weight * (x2 + lam) * G2 * drift).sum(axis=(0, 1))
        )
        return jnp.array(
            [rot.domega_drho / T * g, rot.domega_drho * gx, -(A1 * g + A2 * gx)]
        )

    forces = _dke_thermodynamic_forces(sol.species, field, sol.Erho, 0.0, rot)
    fluxes = [
        flux(sp, rot.lam[i], rot.dlamdt[i], rot.density[i], forces[0, i], forces[1, i])
        for i, sp in enumerate(sol.species)
    ]
    return jnp.array(fluxes).T


_NONDISSIPATIVE = (
    "Part of the {} of each species that is independent of the solution f1, to be "
    "added to <{}>. Nonzero in a rotating plasma on a surface without up-down "
    "symmetry. Not including the part driven by E||."
)


@_register_dke_output(
    name="<nondissipative_particle_flux>",
    label="\\Gamma^H",
    units="m^{-3} \\cdot s^{-1}",
    description=_NONDISSIPATIVE.format("particle flux", "particle_flux"),
    dim=("ns",),
)
def _dke_nondissipative_particle_flux(sol, **kwargs):
    return _nondissipative_fluxes(sol)[0]


@_register_dke_output(
    name="<nondissipative_heat_flux>",
    label="Q^H",
    units="kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}",
    description=_NONDISSIPATIVE.format("heat flux", "heat_flux"),
    dim=("ns",),
)
def _dke_nondissipative_heat_flux(sol, **kwargs):
    return _nondissipative_fluxes(sol)[1]


@_register_dke_output(
    name="<nondissipative_angular_momentum_flux>",
    label="\\Pi^H",
    units="kg \\cdot m^{-1} \\cdot s^{-2}",
    description=_NONDISSIPATIVE.format(
        "angular momentum flux", "angular_momentum_flux"
    ),
    dim=("ns",),
)
def _dke_nondissipative_angular_momentum_flux(sol, **kwargs):
    return _nondissipative_fluxes(sol)[2]


@_register_dke_output(
    name="Phi_0",
    label="\\Phi_0",
    units="V",
    description="Variation of the lowest order electrostatic potential on the "
    "surface of a rotating plasma, zero at theta=0.",
    dim=("nt", "nz"),
)
def _dke_Phi0(sol, **kwargs):
    if sol.rotating is None:
        return jnp.zeros((sol.field.ntheta, sol.field.nzeta))
    return sol.rotating.Phi


@_register_dke_output(
    name="n0",
    label="n_{s0}",
    units="m^{-3}",
    description="Lowest order density on surface for each species, which varies "
    "on the surface in a rotating plasma.",
    dim=("ns", "nt", "nz"),
)
def _dke_n0(sol, **kwargs):
    shape = (len(sol.species), sol.field.ntheta, sol.field.nzeta)
    return jnp.broadcast_to(_density(sol), shape)


@_register_dke_output(
    name="particle_source",
    label="S_p",
    units="s^{-1}",
    description="Particle source for solvability of DKE.",
    dim=("ns",),
)
def _dke_particle_source(sol, **kwargs):
    return sol._particle_source


@_register_dke_output(
    name="heat_source",
    label="S_h",
    units="s^{-1}",
    description="Heat source for solvability of DKE.",
    dim=("ns",),
)
def _dke_heat_source(sol, **kwargs):
    return sol._heat_source


@_register_dke_output(
    name="particleFlux_vm_rN_sfincs",
    label="\\mathrm{particleFlux\\_vm\\_rN}",
    units="None",
    description="Particle flux in sfincs normalization.",
    dim=("ns",),
)
def _dke_particleFlux_vm_rN_sfincs(sol, **kwargs):
    Rbar = kwargs.get("Rbar", 1.0)
    mbar = kwargs.get("mbar", 1.0)
    Tbar = kwargs.get("Tbar", 1e3)
    nbar = kwargs.get("nbar", 1e20)
    mbar *= proton_mass
    Tbar *= _JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<particle_flux>") * Rbar / (nbar * vbar)


@_register_dke_output(
    name="heatFlux_vm_rN_sfincs",
    label="\\mathrm{heatFlux\\_vm\\_rN}",
    units="None",
    description="Heat flux in sfincs normalization.",
    dim=("ns",),
)
def _dke_heatFlux_vm_rN_sfincs(sol, **kwargs):
    Rbar = kwargs.get("Rbar", 1.0)
    mbar = kwargs.get("mbar", 1.0)
    Tbar = kwargs.get("Tbar", 1e3)
    nbar = kwargs.get("nbar", 1e20)
    mbar *= proton_mass
    Tbar *= _JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<heat_flux>") * Rbar / (nbar * vbar**3 * mbar)


@_register_dke_output(
    name="flow_sfincs",
    label="\\mathrm{flow}",
    units="None",
    description="Parallel flow in sfincs normalization.",
    dim=("ns", "nt", "nz"),
)
def _dke_flow_sfincs(sol, **kwargs):
    density = jnp.array([sp.density for sp in sol.species])
    mbar = kwargs.get("mbar", 1.0)
    Tbar = kwargs.get("Tbar", 1e3)
    nbar = kwargs.get("nbar", 1e20)
    mbar *= proton_mass
    Tbar *= _JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("V||") * density[:, None, None] / (nbar * vbar)


@_register_dke_output(
    name="FSABFlow_sfincs",
    label="\\mathrm{FSABFlow}",
    units="None",
    description="Flux surface average field*parallel velocity in sfincs normalization.",
    dim=("ns",),
)
def _dke_FSABFlow_sfincs(sol, **kwargs):
    density = jnp.array([sp.density for sp in sol.species])
    mbar = kwargs.get("mbar", 1.0)
    Tbar = kwargs.get("Tbar", 1e3)
    nbar = kwargs.get("nbar", 1e20)
    Bbar = kwargs.get("Bbar", 1.0)
    mbar *= proton_mass
    Tbar *= _JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<V||B>") / (vbar * Bbar) * density / nbar


@_register_dke_output(
    name="FSABjHat_sfincs",
    label="\\mathrm{FSABjHat}",
    units="None",
    description="Bootstrap current in sfincs normalization.",
    dim=(),
)
def _dke_FSABjHat_sfincs(sol, **kwargs):
    mbar = kwargs.get("mbar", 1.0)
    Tbar = kwargs.get("Tbar", 1e3)
    nbar = kwargs.get("nbar", 1e20)
    Bbar = kwargs.get("Bbar", 1.0)
    mbar *= proton_mass
    Tbar *= _JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<J||B>") / (elementary_charge * nbar * vbar * Bbar)


@_register_dke_output(
    name="j_rN_sfincs",
    label="\\mathrm{j_rN}",
    units="None",
    description="Radial current in sfincs normalization.",
    dim=(),
)
def _dke_j_rN_sfincs(sol, **kwargs):
    mbar = kwargs.get("mbar", 1.0)
    Tbar = kwargs.get("Tbar", 1e3)
    nbar = kwargs.get("nbar", 1e20)
    Rbar = kwargs.get("Rbar", 1.0)
    mbar *= proton_mass
    Tbar *= _JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("J_rho") * Rbar / (elementary_charge * nbar * vbar)


@_register_dke_output(
    name="jHat_sfincs",
    label="\\mathrm{jHat}",
    units="None",
    description="Parallel current in sfincs normalization.",
    dim=("nt", "nz"),
)
def _dke_jHat_sfincs(sol, **kwargs):
    mbar = kwargs.get("mbar", 1.0)
    Tbar = kwargs.get("Tbar", 1e3)
    nbar = kwargs.get("nbar", 1e20)
    mbar *= proton_mass
    Tbar *= _JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("J||") / (elementary_charge * nbar * vbar)
