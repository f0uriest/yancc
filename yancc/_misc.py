"""Constraints, sources, RHS, etc."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Float
from scipy.constants import elementary_charge

from ._sharding import _shard_state
from .field import Field
from .rotation import RotatingEquilibrium
from .species import LocalMaxwellian
from .velocity_grids import UniformPitchAngleGrid, _AbstractSpeedGrid


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
    speedgrid: _AbstractSpeedGrid
    species: list[LocalMaxwellian]

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: _AbstractSpeedGrid,
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
            (1, 1, pitchgrid.nalpha, field.ntheta, field.nzeta)
        )
        s2 = s2[:, :, None, None, None] * jnp.ones(
            (1, 1, pitchgrid.nalpha, field.ntheta, field.nzeta)
        )
        # flatten by species
        s1 = s1.reshape((len(species), -1))
        s2 = s2.reshape((len(species), -1))
        # split and recombine to keep species together
        s1a = jnp.split(s1, len(species))
        s2a = jnp.split(s2, len(species))
        sa = [jnp.concatenate([s1s, s2s]).T for s1s, s2s in zip(s1a, s2a)]
        super().__init__(jax.scipy.linalg.block_diag(*sa))

    def _shard(self, mesh):
        # rows are the flattened (species, speed, ...) state
        return eqx.tree_at(lambda m: m.matrix, self, _shard_state(self.matrix, mesh, 0))


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
    speedgrid: _AbstractSpeedGrid
    species: list[LocalMaxwellian]
    normalize: bool

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: _AbstractSpeedGrid,
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

    def _shard(self, mesh):
        # columns are the flattened (species, speed, ...) state
        return eqx.tree_at(lambda m: m.matrix, self, _shard_state(self.matrix, mesh, 1))


def radial_magnetic_drift(
    field: Field,
    speedgrid: _AbstractSpeedGrid,
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


def radial_drift(
    field: Field,
    speedgrid: _AbstractSpeedGrid,
    pitchgrid: UniformPitchAngleGrid,
    species: list[LocalMaxwellian],
    rotating: RotatingEquilibrium | None = None,
) -> jax.Array:
    """Radial drift dρ/dt of the guiding centers.

    This is the magnetic drift and, in a rotating plasma, the drift in the gradient of
    the potential energy in the rotating frame (ExB and centrifugal) and the Coriolis
    drift:

    dρ/dt = 𝐯ₘ ⋅ ∇ρ - B_ζ T ∂λ/∂θ / (q √g B²) + 2 m ω v|| R ∂R/∂θ / (q √g B)

    which is m/(q ψ') times the rate of change of I v||/B + ω R² along the
    collisionless trajectories, with ψ' = √g B^θ, from the conservation of canonical
    angular momentum.

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
    rotating : RotatingEquilibrium, optional
        Lowest order state of a toroidally rotating plasma.

    Returns
    -------
    f : jax.Array, shape(ns, nx, na, nt, nz)
        Radial drift.
    """
    drift = radial_magnetic_drift(field, speedgrid, pitchgrid, species)
    if rotating is None:
        return drift
    assert field.R is not None
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in species])[:, None, None, None, None]
    qs = jnp.array([sp.species.charge for sp in species])[:, None, None, None, None]
    Ts = jnp.array([sp.temperature for sp in species])[:, None, None, None, None]
    vpar = speedgrid.x[None, :, None, None, None] * vth * pitchgrid.xi[:, None, None]
    potential = -field.B_sub_z * Ts * elementary_charge / qs / field.Bmag
    potential = potential * rotating.dlamdt[:, None, None]
    coriolis = 2 * ms / qs * rotating.omega * vpar * field.R * field._dfdt(field.R)
    return drift + (potential + coriolis) / (field.sqrtg * field.Bmag)


def angular_momentum_drift(
    field: Field,
    speedgrid: _AbstractSpeedGrid,
    pitchgrid: UniformPitchAngleGrid,
    species: list[LocalMaxwellian],
    rotating: RotatingEquilibrium | None = None,
) -> jax.Array:
    """Toroidal angular momentum carried by the radial drift of the guiding centers.

    This is m²/(2 q ψ') times the rate of change of
    (I v||/B + ω R²)² + μ R² B_p² / (m B) along the collisionless trajectories: the
    toroidal angular momentum of a guiding center times its radial drift, plus the
    part of the gyromotion. It multiplies the rotation shear in the drive of the DKE
    and its moment is the flux of toroidal angular momentum.

    Parameters
    ----------
    field : Field
        Magnetic field information. Must be axisymmetric and have R.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    rotating : RotatingEquilibrium, optional
        Lowest order state of a toroidally rotating plasma.

    Returns
    -------
    f : jax.Array, shape(ns, nx, na, nt, nz)
        Angular momentum times radial drift.
    """
    if field.nzeta != 1 or field.R is None:
        raise ValueError("Angular momentum requires an axisymmetric field with R.")
    drift = radial_drift(field, speedgrid, pitchgrid, species, rotating)
    omega = 0.0 if rotating is None else rotating.omega
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in species])[:, None, None, None, None]
    qs = jnp.array([sp.species.charge for sp in species])[:, None, None, None, None]
    xi = pitchgrid.xi[:, None, None]
    v = speedgrid.x[None, :, None, None, None] * vth
    B, I, R = field.Bmag, field.B_sub_z, field.R
    # ∂/∂θ of R² B_p² / (2 B) = (R² B² - I²) / (2 B)
    dh = R * field._dfdt(R) * B + (R**2 / 2 + I**2 / (2 * B**2)) * field.dBdt
    gyro = ms**2 / (2 * qs) * v**3 * xi * (1 - xi**2) * dh / (field.sqrtg * B**2)
    return ms * (I * v * xi / B + omega * R**2) * drift + gyro


def dke_rhs(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: _AbstractSpeedGrid,
    species: list[LocalMaxwellian],
    Erho: float | Float[Any, ""],
    EparB: float | Float[Any, ""] = 0.0,
    single_rhs: bool = True,
    rotating: RotatingEquilibrium | None = None,
) -> jax.Array:
    """RHS of DKE as solved in SFINCS.

    In a rotating plasma the drive is

    -[A₁ + (x² + λ) A₂] dρ/dt F - (dω/dρ)/T (angular momentum drift) F + A₃ B v|| F

    with F the Maxwellian of the local density, the drifts of ``radial_drift`` and
    ``angular_momentum_drift``, λ the potential energy in the rotating frame over the
    temperature, and A₁ including the radial derivative of λ at θ=0.

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
    single_rhs : bool
        If True, return a single combined rhs vector. If False, return ns*3 rhs, each
        unit drive, for computing the transport matrix. With ``rotating`` these are
        followed by one more, for unit dω/dρ.
    rotating : RotatingEquilibrium, optional
        Lowest order state of a toroidally rotating plasma.

    Returns
    -------
    f : jax.Array
        RHS of linear DKE, with the distribution function flattened in its last dim.
    """
    rhs = _dke_rhs_3(field, pitchgrid, speedgrid, species, rotating)
    if single_rhs:
        forces = _dke_thermodynamic_forces(species, field, Erho, EparB, rotating)[
            :, :, None, None, None, None, None
        ]
        rhs = (forces * rhs).sum(axis=(0, 1))
        if rotating is not None:
            shear = _dke_rhs_shear(field, pitchgrid, speedgrid, species, rotating)
            rhs = rhs + rotating.domega_drho * shear
        rhs = rhs.reshape((1, -1))
    else:
        rhs = jnp.swapaxes(rhs, 0, 1)
        rhs = rhs.reshape((3 * len(species), -1))
        if rotating is not None:
            shear = _dke_rhs_shear(field, pitchgrid, speedgrid, species, rotating)
            rhs = jnp.concatenate([rhs, shear.reshape((1, -1))])

    return rhs.squeeze()


def _dke_thermodynamic_forces(species, field, Erho, EparB, rotating=None):
    qs = jnp.array([sp.species.charge for sp in species]) / elementary_charge
    Ts = jnp.array([sp.temperature for sp in species])

    Ln = jnp.array([-spec.aLn for spec in species])
    LT = jnp.array([-spec.aLT for spec in species])
    A1 = Ln + qs * (-Erho) / Ts - 3 / 2 * LT
    if rotating is not None:
        # the density is that at θ=0, n = N exp(-E/T) with E the potential energy in
        # the rotating frame. This makes A1 + 3/2 A2 the gradient of ln(N).
        A1 = A1 + rotating.dlnNdrho
    A2 = LT
    A3 = qs / Ts * EparB / field.B2mag_fsa
    forces = jnp.array([A1, A2, A3])
    return forces


def _dke_rhs_shear(field, pitchgrid, speedgrid, species, rotating):
    """Drive of the DKE by the rotation shear, for unit dω/dρ. (ns, nx, na, nt, nz)."""
    momentum = angular_momentum_drift(field, speedgrid, pitchgrid, species, rotating)
    Fs = jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    Fs = Fs[:, :, None, None, None] * rotating.density[:, None, None]
    Ts = jnp.array([sp.temperature for sp in species])[:, None, None, None, None]
    return -momentum * Fs / (Ts * elementary_charge)


def _dke_rhs_3(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: _AbstractSpeedGrid,
    species: list[LocalMaxwellian],
    rotating: RotatingEquilibrium | None = None,
) -> Float[jax.Array, "3 ns ns nx na nt nz"]:
    vmadotgradrho = radial_drift(field, speedgrid, pitchgrid, species, rotating)
    Fs = jax.vmap(jnp.diag, in_axes=1, out_axes=2)(
        jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    )[:, :, :, None, None, None]
    x = speedgrid.x[None, None, :, None, None, None]
    energy = x**2
    if rotating is not None:
        # Maxwellian of the local density, and the potential energy
        Fs = Fs * rotating.density[:, None, None]
        energy = energy + rotating.lam[:, None, None]
    vth = jnp.diag(jnp.array([sp.v_thermal for sp in species]))[
        :, :, None, None, None, None
    ]
    vpar = pitchgrid.xi[None, None, None, :, None, None] * vth * x
    Bvpar = field.Bmag[None, None, None, None, :, :] * vpar

    r1 = -vmadotgradrho * Fs
    r2 = -energy * vmadotgradrho * Fs
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
