"""Constraints, sources, RHS, etc."""

import jax
import jax.numpy as jnp
import lineax as lx
from scipy.constants import elementary_charge, proton_mass

from .field import Field
from .species import JOULE_PER_EV, LocalMaxwellian
from .velocity_grids import AbstractSpeedGrid, UniformPitchAngleGrid


class DKESources(lx.MatrixLinearOperator):
    """Fake sources of particles and heat to ensure solvability

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species

        x = speedgrid.x
        F = jnp.array([sp(x * sp.v_thermal) for sp in species])
        # these have shape (ns, nx)
        s1 = (x**2 - 5 / 2) * F
        s2 = (-2 / 3 * x**2 + 1) * F
        # now need to make them broadcast against full distribution function
        # these have shape (ns, nx, nxi, nt, nz)
        s1 = s1[:, :, None, None, None] * jnp.ones(
            (1, 1, pitchgrid.nxi, field.ntheta, field.nzeta)
        )
        s2 = s2[:, :, None, None, None] * jnp.ones(
            (1, 1, pitchgrid.nxi, field.ntheta, field.nzeta)
        )
        # flatten by species
        s1 = s1.reshape((len(species), -1))
        s2 = s2.reshape((len(species), -1))
        # split and recombine to keep species together
        s1a = jnp.split(s1, len(species))
        s2a = jnp.split(s2, len(species))
        sa = [jnp.concatenate([s1s, s2s]).T for s1s, s2s in zip(s1a, s2a)]
        super().__init__(jax.scipy.linalg.block_diag(*sa))


class DKEConstraint(lx.MatrixLinearOperator):
    """Constraints to fix gauge freedom in density and energy.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    normalize : bool
        Whether to ignore factors of v_thermal in the
        integrals. If True, integrals will be dimensionless.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    normalize: bool

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        normalize=True,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.normalize = normalize

        if normalize:
            vth = jnp.ones((len(species), 1, 1))
        else:
            vth = jnp.array([sp.v_thermal for sp in species])[:, None, None]
        dx = (speedgrid.x**2 * speedgrid.wx)[None, :, None]
        x2dx = (speedgrid.x**4 * speedgrid.wx)[None, :, None]
        dxi = pitchgrid.wxi[None, None, :]
        # int f d3v, for particle conservation, shape(ns, nx, nxi)
        d3v = vth**3 * dx * dxi
        # int v^2 f d3v, for energy conservation, shape(ns, nx, nxi)
        v2d3v = vth**5 * x2dx * dxi

        # flux surface average operator
        dt = field.wtheta[:, None]
        dz = field.wzeta[None, :]
        dr = (field.sqrtg * dt * dz) / (field.sqrtg * dt * dz).sum()
        dr = dr.flatten()[None, None, None, :]

        Ip = 2 * jnp.pi * (d3v[..., None] * dr).reshape((len(species), -1))
        Ie = 2 * jnp.pi * (v2d3v[..., None] * dr).reshape((len(species), -1))
        Ipa = jnp.split(Ip, len(species))
        Iea = jnp.split(Ie, len(species))
        Ia = [jnp.concatenate([Ips, Ies]) for Ips, Ies in zip(Ipa, Iea)]
        super().__init__(jax.scipy.linalg.block_diag(*Ia))


def radial_magnetic_drift(
    field: Field,
    speedgrid: AbstractSpeedGrid,
    pitchgrid: UniformPitchAngleGrid,
    species: list[LocalMaxwellian],
) -> jax.Array:
    """Radial magnetic drift 𝐯ₘ ⋅ ∇ ψ

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered

    Returns
    -------
    f : jax.Array, shape(ns, nx, nxi, nt, nz)
        Radial magnetic drift.
    """
    if not isinstance(species, (list, tuple)):
        species = [species]
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in species])[:, None, None, None, None]
    qs = jnp.array([sp.species.charge for sp in species])[:, None, None, None, None]
    xi = pitchgrid.xi[None, None, :, None, None]
    x = speedgrid.x[None, :, None, None, None]
    v = x * vth
    vmadotgradrho = -(
        ms * v**2 / qs * (1 + xi**2) / (2 * field.Bmag**3) * field.BxgradrhodotgradB
    )
    return vmadotgradrho


def dke_rhs(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: AbstractSpeedGrid,
    species: list[LocalMaxwellian],
    Erho: float,
    include_constraints: bool = True,
    normalize: bool = False,
) -> jax.Array:
    """RHS of DKE as solved in SFINCS.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
    include_constraints : bool
        Whether to append zeros to the rhs for constraint equations.
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize

    Returns
    -------
    f : jax.Array
        RHS of linear DKE.
    """
    if not isinstance(species, (list, tuple)):
        species = [species]
    qs = jnp.array([sp.species.charge for sp in species])[:, None, None, None, None]
    ns = jnp.array([sp.density for sp in species])[:, None, None, None, None]
    dns = jnp.array([sp.dndrho for sp in species])[:, None, None, None, None]
    Ts = jnp.array([sp.temperature for sp in species])[:, None, None, None, None]
    dTs = jnp.array([sp.dTdrho for sp in species])[:, None, None, None, None]
    Ln = dns / ns
    LT = dTs / Ts
    x = speedgrid.x[None, :, None, None, None]
    vmadotgradrho = radial_magnetic_drift(field, speedgrid, pitchgrid, species)
    gradients = Ln + qs * Erho / Ts + (x**2 - 3 / 2) * LT
    rhs = -vmadotgradrho * gradients
    if normalize:
        vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
        rhs /= vth
    rhs = rhs.flatten()
    if include_constraints:
        rhs = jnp.concatenate([rhs, jnp.zeros(2 * len(species))])
    return rhs


def mdke_rhs(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
) -> jax.Array:
    """RHS of monoenergetic DKE.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.

    Returns
    -------
    f : jax.Array, shape(N,3)
        RHS of linear monoenergetic DKE.
    """
    xi = pitchgrid.xi[:, None, None]
    s1 = (1 + xi**2) / (2 * field.Bmag**3) * field.BxgradrhodotgradB
    s2 = s1
    s3 = xi * field.Bmag
    rhs = jnp.array([s1, s2, s3]).reshape((3, -1)).T
    return rhs


@jax.jit
def compute_monoenergetic_coefficients(
    f: jax.Array,
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
) -> jax.Array:
    """Compute D_ij coefficients from solution for distribution function f.

    Parameters
    ----------
    f : jax.Array, shape(N,3)
        Solution to monoenergetic drift kinetic equation.
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.

    Returns
    -------
    Dij : jax.Array, shape(3, 3)
        Monoenergetic transport coefficients.
    """
    f = f.reshape((-1, 3))
    nxi, nt, nz = (
        pitchgrid.nxi,
        field.ntheta,
        field.nzeta,
    )
    N = nxi * nt * nz
    # slice out source/constraint terms if present
    f = f[:N]

    s = mdke_rhs(field, pitchgrid)
    s = s.reshape((-1, 3))

    # form monoenergetic coefficients
    sf = s.T[:, None] * f.T[None, :]  # shape (3,3,N)
    Dij_itz = sf.reshape((3, 3, nxi, nt, nz))
    Dij_i = field.flux_surface_average(Dij_itz)
    Dij = jnp.sum(Dij_i * pitchgrid.wxi, axis=-1)
    return Dij


def normalize_dkes(Dij: jax.Array, field: Field, v: float = 1.0):
    """Normalize monoenergetic coefficients to match DKES/MONKES.

    Parameters
    ----------
    Dij : jax.Array, shape(..., 3, 3)
        Array of monoenergetic coefficients.
    field : Field
        Magnetic field information
    v : float
        Speed being considered.

    Returns
    -------
    Dij : jax.Array, shape(..., 3, 3)
        Rescaled monoenergetic coefficients
    """
    sgn = jnp.sign(field.Psi)
    a = field.a_minor
    sa = sgn * a
    B0 = field.B0

    scale = (
        jnp.array(
            [
                [a**2, sa, sa / B0],
                [sa, sa, sa / B0],
                [sa / B0, sa / B0, 1 / B0**2],
            ]
        )
        * v
    )
    return Dij * scale


def compute_transport_matrix(
    Dij: jax.Array,
    speedgrid: AbstractSpeedGrid,
    species: list[LocalMaxwellian],
):
    """Compute the transport matrix for each species from monoenergetic coefficients.

    Parameters
    ----------
    Dij : jax.Array, shape(nspecies, nx, 3, 3)
        Monoenergetic transport coefficient for each species and each speed.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered

    Returns
    -------
    Lij : jax.Array, shape(nspecies, 3, 3)
        Transport marix for each species.
    """
    # TODO: check this, it seems to disagree with the formulas in monkes and beidler
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None]
    x = speedgrid.x[None, :, None, None]
    fM = jnp.concatenate([sp(x * sp.v_thermal) for sp in species], axis=0)
    wx = speedgrid.wx[None, :, None, None]

    wij = jnp.concatenate(
        [
            jnp.concatenate([x**0, x**2, x**1], axis=3),
            jnp.concatenate([x**2, x**4, x**3], axis=3),
            jnp.concatenate([x**1, x**3, x**2], axis=3),
        ],
        axis=2,
    )
    vscale = jnp.concatenate(
        [
            jnp.concatenate([vth**3, vth**3, vth**4], axis=3),
            jnp.concatenate([vth**3, vth**3, vth**4], axis=3),
            jnp.concatenate([vth**4, vth**4, vth**5], axis=3),
        ],
        axis=2,
    )
    integrand = vscale * x**2 * fM * wij * Dij * wx
    return integrand.sum(axis=1)


def compute_fluxes(
    f: jax.Array,
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: AbstractSpeedGrid,
    species: list[LocalMaxwellian],
):
    """Compute output fluxes from solution of DKE.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered

    Returns
    -------
    fluxes: dict of ndarray
        Contains:
        particle_flux : jax.Array, shape(ns)
            Γₐ = Particle flux for each species.
        heat_flux : jax.Array, shape(ns)
            Qₐ = Heat flux for each species.
        Vpar : jax.Array, shape(ns, nt, nz)
            V|| = Parallel velocity for each species
        BVpar: jax.Array, shape(ns)
            〈BV||〉 = Flux surface average field*parallel velocity for each species
        bootstrap_current: float
            〈J||B〉 = Bootstrap current.
        radial_current : float
            Jr = Radial current.
        Jpar : jax.Array, shape(nt, nz)
            J|| = Parallel current density.
    """
    if not isinstance(species, (list, tuple)):
        species = [species]
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in species])[:, None, None, None, None]
    qs = jnp.array([sp.species.charge for sp in species])[:, None, None, None, None]
    ns = jnp.array([sp.density for sp in species])[:, None, None, None, None]
    xi = pitchgrid.xi[None, None, :, None, None]
    x = speedgrid.x[None, :, None, None, None]

    dx = ((speedgrid.x**2 * speedgrid.wx) @ speedgrid.xvander_inv)[
        None, :, None, None, None
    ]
    dxi = pitchgrid.wxi[None, None, :, None, None]
    # int f d3v
    d3v = vth**3 * dx * dxi

    # flux surface average operator
    dt = field.wtheta[None, None, None, :, None]
    dz = field.wzeta[None, None, None, None, :]
    dr = field.sqrtg[None, None, None, :, :] * dt * dz
    dr = dr / dr.sum()
    radial_drift = radial_magnetic_drift(field, speedgrid, pitchgrid, species)
    vpar = x * vth * xi

    particle_flux = f * radial_drift * d3v * dr
    heat_flux = f * ms * vth**2 * x**2 * radial_drift * d3v * dr
    Vpar = 1 / ns * vpar * f * d3v
    BVpar = field.Bmag[None, None, None, :, :] * Vpar * dr
    bootstrap_current = ns * qs * BVpar
    radial_current = particle_flux * qs
    Jpar = qs * ns * Vpar

    particle_flux = particle_flux.sum(axis=(1, 2, 3, 4))
    heat_flux = heat_flux.sum(axis=(1, 2, 3, 4))
    Vpar = Vpar.sum(axis=(1, 2))
    BVpar = BVpar.sum(axis=(1, 2, 3, 4))
    bootstrap_current = bootstrap_current.sum(axis=(0, 1, 2, 3, 4))
    radial_current = radial_current.sum(axis=(0, 1, 2, 3, 4))
    Jpar = Jpar.sum(axis=(0, 1, 2))

    fluxes = {
        "particle_flux": particle_flux,
        "heat_flux": heat_flux,
        "Vpar": Vpar,
        "BVpar": BVpar,
        "bootstrap_current": bootstrap_current,
        "radial_current": radial_current,
        "Jpar": Jpar,
    }

    return fluxes


def normalize_fluxes_sfincs(fluxes, Bbar=1, Rbar=1, nbar=1e19, mbar=1, Tbar=1e3):
    """Normalize fluxes to match SFINCS.

    Parameters
    ----------
    fluxes : dict of array
        Fluxes, as output from ``compute_fluxes``.
    Bbar : float
        Reference magnetic field, in Tesla.
    Rbar : float
        Reference length scale, in meters.
    nbar : float
        Reference density, in 1/meter^3
    mbar : float
        Reference mass, in proton masses.
    Tbar : float
        Reference temperature, in eV.
    """
    mbar *= proton_mass
    Tbar *= JOULE_PER_EV
    vbar = jnp.sqrt(Tbar / mbar)
    fluxes = fluxes.copy()
    fluxes["particle_flux"] *= Rbar / (nbar * vbar)
    fluxes["heat_flux"] *= Rbar / (nbar * vbar**3 * mbar)
    fluxes["Vpar"] *= 1 / (nbar * vbar)
    fluxes["BVpar"] *= 1 / (vbar * Bbar * nbar)
    fluxes["bootstrap_current"] *= 1 / (elementary_charge * nbar * vbar * Bbar)
    fluxes["radial_current"] *= Rbar / (elementary_charge * nbar * vbar)
    fluxes["jpar"] *= 1 / (elementary_charge * nbar * vbar)
    return fluxes
