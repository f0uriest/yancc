"""Constraints, sources, RHS, etc."""

import cola
import jax
import jax.numpy as jnp
from monkes import Field, LocalMaxwellian

from .velocity_grids import PitchAngleGrid, SpeedGrid


class DKESources(cola.ops.Dense):
    """Fake sources of particles and heat to ensure solvability

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
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


class DKEConstraint(cola.ops.Dense):
    """Constraints to fix gauge freedom in density and energy.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    normalize : bool
        Whether to ignore factors of v_thermal in the
        integrals. If True, integrals will be dimensionless.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        normalize=True,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species

        if normalize:
            vth = jnp.ones((len(species), 1, 1))
        else:
            vth = jnp.array([sp.v_thermal for sp in species])[:, None, None]
        # xvander goes from modal -> nodal
        dx = ((speedgrid.x**2 * speedgrid.wx) @ speedgrid.xvander)[None, :, None]
        x2dx = ((speedgrid.x**4 * speedgrid.wx) @ speedgrid.xvander)[None, :, None]
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


def dke_rhs(
    field: Field,
    speedgrid: SpeedGrid,
    pitchgrid: PitchAngleGrid,
    species: list[LocalMaxwellian],
    E_psi: float,
    include_constraints: bool = True,
) -> jax.Array:
    """RHS of DKE as solved in SFINCS.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Radial electric field.
    include_constraints : bool
        Whether to append zeros to the rhs for constraint equations.

    Returns
    -------
    f : jax.Array
        RHS of linear DKE.
    """
    if not isinstance(species, (list, tuple)):
        species = [species]
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in species])[:, None, None, None, None]
    qs = jnp.array([sp.species.charge for sp in species])[:, None, None, None, None]
    ns = jnp.array([sp.density for sp in species])[:, None, None, None, None]
    dns = jnp.array([sp.dndr for sp in species])[:, None, None, None, None]
    Ts = jnp.array([sp.temperature for sp in species])[:, None, None, None, None]
    dTs = jnp.array([sp.dTdr for sp in species])[:, None, None, None, None]
    Ln = dns / ns
    LT = dTs / Ts
    xi = pitchgrid.xi[None, None, :, None, None]
    x = speedgrid.x[None, :, None, None, None]
    vmadotgradpsi = (
        x**2
        * vth**2
        * (1 / 2 + xi**2 / 2)
        * ms
        / qs
        / field.Bmag**2
        * field.BxgradpsidotgradB
    )
    gradients = Ln + qs * E_psi / Ts + (x**2 - 3 / 2) * LT
    rhs = (vmadotgradpsi * gradients).flatten()
    if include_constraints:
        rhs = jnp.concatenate([rhs, jnp.zeros(2 * len(species))])
    return rhs


def mdke_rhs(
    field: Field,
    speedgrid: SpeedGrid,
    pitchgrid: PitchAngleGrid,
    species: list[LocalMaxwellian],
    E_psi: float,
    include_constraints: bool = True,
) -> jax.Array:
    """RHS of monoenergetic DKE.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Radial electric field.
    include_constraints : bool
        Whether to append zeros to the rhs for constraint equations.

    Returns
    -------
    f : jax.Array, shape(N,3)
        RHS of linear monoenergetic DKE.
    """
    if not isinstance(species, (list, tuple)):
        species = [species]
    vth = jnp.array([s.v_thermal for s in species])[:, None, None, None, None]
    x = speedgrid.x[None, :, None, None, None]
    v = vth * x
    xi = pitchgrid.xi[None, None, :, None, None]
    s1 = (1 + xi**2) / (2 * field.Bmag**3) * field.BxgradpsidotgradB
    s2 = s1
    s3 = xi * field.Bmag
    rhs = jnp.array([s1 * v, s2 * v, s3 * v]).reshape((3, -1)).T
    if include_constraints:
        rhs = jnp.concatenate([rhs, jnp.zeros((len(species), 3))])
    return rhs


@jax.jit
def compute_monoenergetic_coefficients(f, s, field, xigrid):
    """Compute D_ij coefficients from solution for distribution function f."""
    # dummy index for x variable
    f = f.reshape((3, xigrid.nxi, 1, field.ntheta, field.nzeta))
    s = s.reshape((3, xigrid.nxi, 1, field.ntheta, field.nzeta))
    sf = s[:, None] * f[None, :]
    sf = jax.vmap(jax.vmap(xigrid._integral))(sf)
    Dij = field.flux_surface_average(sf)
    return Dij.squeeze()
