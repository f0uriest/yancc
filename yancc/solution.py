"""Solution objects and computation of output moments."""

import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import elementary_charge, proton_mass

from .field import Field
from .misc import _d3v, _dr, radial_magnetic_drift
from .species import JOULE_PER_EV, LocalMaxwellian
from .velocity_grids import AbstractPitchAngleGrid, AbstractSpeedGrid

MDKE_OUTPUTS = {}
DKE_OUTPUTS = {}

DKE_DEFAULT_OUTPUT_QTYS = (
    "<heat_flux>",
    "<particle_flux>",
    "<V||B>",
    "<J||B>",
    "J_rho",
)

_SUPERSCRIPT_DIGITS = str.maketrans("0123456789-+=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁼⁽⁾")


def clean_units(s: str) -> str:
    r"""Render a LaTeX units string from ``DKE_OUTPUTS`` as plain unicode.

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


def register_mdke_output(name, label, units, description, dim):
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
    dim : int
        Size of output array
    """

    def _decorator(func):
        d = {
            "label": label,
            "units": units,
            "description": description,
            "fun": func,
            "dim": dim,
        }
        MDKE_OUTPUTS[name] = d
        return func

    return _decorator


def register_dke_output(name, label, units, description, dim):
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
    dim : int
        Size of output array
    """

    def _decorator(func):
        d = {
            "label": label,
            "units": units,
            "description": description,
            "fun": func,
            "dim": dim,
        }
        DKE_OUTPUTS[name] = d
        return func

    return _decorator


class DKESolution(eqx.Module):
    """Solution to Drift Kinetic Equation.

    Attributes
    ----------
    F0 : jax.Array, shape (ns, nx, 1, 1, 1)
        Maxwellian background distribution, broadcastable against ``f1``.
    f1 : jax.Array, shape (ns, nx, na, nt, nz)
        Perturbation solved for by the DKE.
    rhs : jax.Array, shape (ns, nx, na, nt, nz)
        Drive terms for DKE
    field : Field
        Magnetic field information.
    pitchgrid : AbstractPitchAngleGrid
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
    f : jax.Array, shape (ns, nx, na, nt, nz)
        Full distribution function ``F0 + f1``. Computed on access.
    f1_krylov : jax.Array
        Distribution function including source terms, as seen by the Krylov solver.
        Computed on access.
    """

    F0: jax.Array
    f1: jax.Array
    rhs: jax.Array
    field: Field
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: jax.Array
    EparB: jax.Array
    background: list[LocalMaxwellian]
    _particle_source: jax.Array
    _heat_source: jax.Array

    def __init__(
        self,
        F0: jax.Array,
        f1: jax.Array,
        rhs: jax.Array,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: jax.Array,
        EparB: jax.Array,
        background: list[LocalMaxwellian],
    ):
        ns = len(species)
        shape = (ns, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta)
        N = np.prod(shape)
        f1 = f1.flatten()
        if f1.size == N:
            f1 = f1.reshape(shape)
            particle_source = jnp.full(ns, jnp.nan)
            heat_source = jnp.full(ns, jnp.nan)
        elif f1.size == N + 2 * ns:
            heat_source = f1[-ns:]
            particle_source = f1[-2 * ns : -ns]
            f1 = f1[:N].reshape(shape)
        else:
            raise ValueError("got wrong size for f1")

        self.F0 = jnp.asarray(F0).reshape(ns, speedgrid.nx, 1, 1, 1)
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
    def f1_krylov(self) -> jax.Array:
        """Distribution function, including source terms, as seen by krylov solver."""
        f1 = self.f1.flatten()
        sources = jnp.concatenate([self._particle_source, self._heat_source])
        sources = jnp.nan_to_num(sources, nan=0.0)
        return jnp.concatenate([f1, sources])

    def get(self, qty, **kwargs):
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
        assert qty in DKE_OUTPUTS.keys()
        return DKE_OUTPUTS[qty]["fun"](self, **kwargs)

    def qtys_list(self) -> list[str]:
        """List of all computable output quantities."""
        return list(DKE_OUTPUTS.keys())

    def print_summary(self, qtys: tuple[str, ...] = DKE_DEFAULT_OUTPUT_QTYS) -> None:
        """Print headline output moments with units.

        Per-species quantities (shape ``(ns,)``) render as
        ``[ v0 v1 ... ] (per species, units)``; scalar quantities (sums over
        species, e.g. ``<J||B>`` and ``J_rho``) render as ``v (units)``.
        """
        ns = len(self.species)
        width = max(len(q) for q in qtys)
        for qty in qtys:
            vals = self.get(qty)
            units = clean_units(DKE_OUTPUTS[qty].get("units", ""))
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
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v = -∂Φ /∂ρ /v in units of V*s/m.
    nuhat : float
        Monoenergetic collisionality, ν/v in units of 1/m.
    """

    f: jax.Array
    rhs: jax.Array
    field: Field
    pitchgrid: AbstractPitchAngleGrid
    nuhat: jax.Array
    erhohat: jax.Array

    def __init__(
        self,
        f: jax.Array,
        rhs: jax.Array,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        nuhat: jax.Array,
        erhohat: jax.Array,
    ):
        self.f = f.reshape(3, pitchgrid.nalpha, field.ntheta, field.nzeta)
        self.rhs = rhs.reshape(3, pitchgrid.nalpha, field.ntheta, field.nzeta)
        self.field = field
        self.pitchgrid = pitchgrid
        self.nuhat = nuhat
        self.erhohat = erhohat

    def get(self, qty, **kwargs):
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
        assert qty in MDKE_OUTPUTS.keys()
        return MDKE_OUTPUTS[qty]["fun"](self, **kwargs)

    def qtys_list(self) -> list[str]:
        """List of all computable output quantities."""
        return list(MDKE_OUTPUTS.keys())


@register_mdke_output(
    name="Dij",
    label="$D_{ij}$",
    units="\\mathrm{Various}",
    description="Monoenergetic transport coefficient matrix",
    dim=(3, 3),
)
def _mdke_Dij(sol, normalization=None, **kwargs):
    """Monoenergetic transport coefficients."""
    f = sol.f.reshape((-1, 3))
    s = sol.rhs.reshape((-1, 3))
    na, nt, nz = (
        sol.pitchgrid.nalpha,
        sol.field.ntheta,
        sol.field.nzeta,
    )
    sf = s.T[:, None] * f.T[None, :]  # shape (3,3,N)
    Dij_itz = sf.reshape((3, 3, na, nt, nz))
    Dij_i = sol.field.flux_surface_average(Dij_itz)  # shape (3,3,na)
    Dij = jnp.sum(Dij_i * sol.pitchgrid.wxi, axis=-1)  # shape (3,3)
    return Dij


@register_mdke_output(
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


@register_dke_output(
    name="<particle_flux>",
    label="\\Gamma = \\langle \\int d^3v f_s \\mathbf{v}_m \\cdot "
    "\\nabla \\rho \\rangle",
    units="m^{-3} \\cdot s^{-1}",
    description="Particle flux for each species",
    dim=("ns",),
)
def _dke_particle_flux(sol, **kwargs):
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    dr = _dr(sol.field)[None, None, None]
    dr = dr / dr.sum()

    radial_drift = radial_magnetic_drift(
        sol.field, sol.speedgrid, sol.pitchgrid, sol.species
    )

    particle_flux = sol.f * radial_drift * d3v * dr
    particle_flux = particle_flux.sum(axis=(-4, -3, -2, -1))
    return particle_flux


@register_dke_output(
    name="<heat_flux>",
    label="Q = \\langle \\int d^3v 1/2 m v^2 f_s \\mathbf{v}_m "
    "\\cdot \\nabla \\rho \\rangle",
    units="kg \\cdot m^{-1} \\cdot s^{-3} = W \\cdot m^{-3}",
    description="Heat flux for each species",
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

    radial_drift = radial_magnetic_drift(
        sol.field, sol.speedgrid, sol.pitchgrid, sol.species
    )
    heat_flux = 1 / 2 * sol.f * ms * v**2 * radial_drift * d3v * dr
    heat_flux = heat_flux.sum(axis=(-4, -3, -2, -1))
    return heat_flux


@register_dke_output(
    name="V||",
    label="V_{||} = 1/n_s \\int d^3v v_{||} f_s",
    units="m \\cdot s^{-1}",
    description="Parallel flow on surface for each species.",
    dim=("ns", "nt", "nz"),
)
def _dke_Vpar(sol, **kwargs):
    vth = jnp.array([sp.v_thermal for sp in sol.species])[:, None, None, None, None]
    ns = jnp.array([sp.density for sp in sol.species])[:, None, None, None, None]
    xi = sol.pitchgrid.xi[None, None, :, None, None]
    x = sol.speedgrid.x[None, :, None, None, None]
    vpar = x * vth * xi
    d3v = _d3v(sol.speedgrid, sol.pitchgrid, sol.species)[..., None, None]
    dr = _dr(sol.field)[None, None, None]
    dr = dr / dr.sum()

    Vpar = 1 / ns * vpar * sol.f * d3v
    Vpar = Vpar.sum(axis=(1, 2))
    return Vpar


@register_dke_output(
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


@register_dke_output(
    name="<J||B>",
    label="J_{||}B = \\sum_s q_s/n_s \\langle B \\int d^3v v_{||} f_s \\rangle",
    units="A \\cdot T \\cdot m^{-3}",
    description="Bootstrap current",
    dim=(),
)
def _dke_bootstrap_current(sol, **kwargs):
    ns = jnp.array([sp.density for sp in sol.species])
    qs = jnp.array([sp.species.charge for sp in sol.species])
    BVpar = sol.get("<V||B>")
    bootstrap_current = ns * qs * BVpar
    bootstrap_current = bootstrap_current.sum(axis=(-1))
    return bootstrap_current


@register_dke_output(
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


@register_dke_output(
    name="J||",
    label="J_{||} = \\sum_s q_s V_{||,s}",
    units="A \\cdot m^{-3}",
    description="Parallel current",
    dim=("nt", "nz"),
)
def _dke_parallel_current(sol, **kwargs):
    ns = jnp.array([sp.density for sp in sol.species])[:, None, None]
    qs = jnp.array([sp.species.charge for sp in sol.species])[:, None, None]
    Vpar = sol.get("V||")
    Jpar = qs * ns * Vpar
    Jpar = Jpar.sum(axis=(-3))
    return Jpar


@register_dke_output(
    name="particle_source",
    label="S_p",
    units="s^{-1}",
    description="Particle source for solvability of DKE.",
    dim=("ns",),
)
def _dke_particle_source(sol, **kwargs):
    return sol._particle_source


@register_dke_output(
    name="heat_source",
    label="S_h",
    units="s^{-1}",
    description="Heat source for solvability of DKE.",
    dim=("ns",),
)
def _dke_heat_source(sol, **kwargs):
    return sol._heat_source


@register_dke_output(
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
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<particle_flux>") * Rbar / (nbar * vbar)


@register_dke_output(
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
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<heat_flux>") * Rbar / (nbar * vbar**3 * mbar)


@register_dke_output(
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
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("V||") * density[:, None, None] / (nbar * vbar)


@register_dke_output(
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
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<V||B>") / (vbar * Bbar) * density / nbar


@register_dke_output(
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
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("<J||B>") / (elementary_charge * nbar * vbar * Bbar)


@register_dke_output(
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
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("J_rho") * Rbar / (elementary_charge * nbar * vbar)


@register_dke_output(
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
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(2 * Tbar / mbar)
    return sol.get("J||") / (elementary_charge * nbar * vbar)
