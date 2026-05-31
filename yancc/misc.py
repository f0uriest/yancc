"""Constraints, sources, RHS, etc."""

from typing import Any, Union

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Float
from scipy.constants import elementary_charge

from .field import Field
from .species import LocalMaxwellian
from .velocity_grids import AbstractSpeedGrid, UniformPitchAngleGrid


def _dr(field):
    """Real space volume element."""
    dt = field.wtheta[:, None]
    dz = field.wzeta[None, :]
    dr = field.sqrtg[:, :] * dt * dz
    dr = dr
    return dr


def _d3v(speedgrid, pitchgrid, species):
    """Velocity space volume element."""
    dx = (speedgrid.x**2 * speedgrid.wx)[None, :, None]
    dxi = pitchgrid.wxi[None, None, :]
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None]
    return 2 * jnp.pi * vth**3 * dxi * dx


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
        # these have shape (ns, nx, na, nt, nz)
        s1 = s1[:, :, None, None, None] * jnp.ones(
            (1, 1, pitchgrid.na, field.ntheta, field.nzeta)
        )
        s2 = s2[:, :, None, None, None] * jnp.ones(
            (1, 1, pitchgrid.na, field.ntheta, field.nzeta)
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

        vth = jnp.array([sp.v_thermal for sp in species])[:, None, None]

        # int f d3v, for particle conservation, shape(ns, nx, na)
        d3v = _d3v(speedgrid, pitchgrid, species)
        # int v^2 f d3v, for energy conservation, shape(ns, nx, na)
        v2d3v = speedgrid.x[None, :, None] ** 2 * vth**2 * d3v

        if normalize:
            d3v /= vth**3
            v2d3v /= vth**5

        # flux surface average operator
        dr = _dr(field).flatten()

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
    f : jax.Array, shape(ns, nx, na, nt, nz)
        Radial magnetic drift.
    """
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
    Erho: Union[float, Float[Any, ""]],
    EparB: Union[float, Float[Any, ""]] = 0.0,
    include_constraints: bool = True,
    single_rhs: bool = True,
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
    EparB : float
        <E||B>, flux surface average of parallel electric field times B.
    include_constraints : bool
        Whether to append zeros to the rhs for constraint equations.
    single_rhs : bool
        If True, return a single combined rhs vector. If False, return ns*3 rhs, each
        unit drive, for computing the transport matrix.

    Returns
    -------
    f : jax.Array
        RHS of linear DKE.
    """
    rhs = _dke_rhs_3(field, pitchgrid, speedgrid, species)
    if single_rhs:
        forces = _dke_thermodynamic_forces(species, field, Erho, EparB)[
            :, :, None, None, None, None, None
        ]
        rhs = (forces * rhs).sum(axis=(0, 1)).reshape((1, -1))
    else:
        rhs = jnp.swapaxes(rhs, 0, 1)
        rhs = rhs.reshape((3 * len(species), -1))

    if include_constraints:
        rhs = jnp.pad(rhs, [(0, 0), (0, 2 * len(species))])
    return rhs.squeeze()


def _dke_thermodynamic_forces(species, field, Erho, EparB):
    qs = jnp.array([sp.species.charge for sp in species]) / elementary_charge
    Ts = jnp.array([sp.temperature for sp in species])

    Ln = jnp.array([-spec.aLn for spec in species])
    LT = jnp.array([-spec.aLT for spec in species])
    A1 = Ln + qs * (-Erho) / Ts - 3 / 2 * LT
    A2 = LT
    A3 = qs / Ts * EparB / field.B2mag_fsa
    forces = jnp.array([A1, A2, A3])
    return forces


def _dke_rhs_3(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: AbstractSpeedGrid,
    species: list[LocalMaxwellian],
) -> Float[jax.Array, "3 ns ns nx na nt nz"]:
    vmadotgradrho = radial_magnetic_drift(field, speedgrid, pitchgrid, species)
    Fs = jax.vmap(jnp.diag, in_axes=1, out_axes=2)(
        jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    )[:, :, :, None, None, None]
    x = speedgrid.x[None, None, :, None, None, None]
    vth = jnp.diag(jnp.array([sp.v_thermal for sp in species]))[
        :, :, None, None, None, None
    ]
    vpar = pitchgrid.xi[None, None, None, :, None, None] * vth * x
    Bvpar = field.Bmag[None, None, None, None, :, :] * vpar

    r1 = -vmadotgradrho * Fs
    r2 = -(x**2) * vmadotgradrho * Fs
    r3 = Bvpar * Fs
    rhs = jnp.array([r1, r2, r3])
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
