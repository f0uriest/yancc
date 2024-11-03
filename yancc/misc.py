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


class MDKESources(cola.ops.Dense):
    """Fake sources of particles to ensure solvability of monoenergetic DKE.

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
        N = pitchgrid.nxi * field.ntheta * field.nzeta
        sa = [
            jax.scipy.linalg.block_diag(*[jnp.ones(N) for _ in speedgrid.x])
            for _ in species
        ]
        super().__init__(jax.scipy.linalg.block_diag(*sa))


class MDKEConstraint(cola.ops.Dense):
    """Constraints to fix gauge freedom in density for monoenergetic DKE.

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

        # independent constraints for each x, each species
        vth = jnp.ones((len(species), 1, 1))
        dx = jnp.ones((1, speedgrid.nx, 1))
        dxi = pitchgrid.wxi[None, None, :]
        # int f d3v, for particle conservation, shape(ns, nx, nxi)
        d3v = vth**3 * dx * dxi

        # flux surface average operator
        dt = field.wtheta[:, None]
        dz = field.wzeta[None, :]
        dr = (field.sqrtg * dt * dz) / (field.sqrtg * dt * dz).sum()
        dr = dr.flatten()[None, None, None, :]

        Ip = (
            2 * jnp.pi * (d3v[..., None] * dr).reshape((len(species), speedgrid.nx, -1))
        )
        Ipa = jnp.split(Ip, len(species))
        Ia = [jax.scipy.linalg.block_diag(*Ipxa[0]) for Ipxa in Ipa]
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
def compute_monoenergetic_coefficients(
    f: jax.Array,
    s: jax.Array,
    field: Field,
    speedgrid: SpeedGrid,
    pitchgrid: PitchAngleGrid,
    species: list[LocalMaxwellian],
) -> jax.Array:
    """Compute D_ij coefficients from solution for distribution function f.

    Parameters
    ----------
    f : jax.Array, shape(N,3)
        Solution to monoenergetic drift kinetic equation.
    s : jax.Array, shape(N,3)
        RHS for monoenergetic drift kinetic equation, eg from `mdke_rhs`.
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered

    Returns
    -------
    Dij : jax.Array, shape(nspecies, nx, 3, 3)
        Monoenergetic transport coefficient for each species and each speed.
    """
    f = f.reshape((-1, 3))
    s = s.reshape((-1, 3))
    ns, nx, nxi, nt, nz = (
        len(species),
        speedgrid.nx,
        pitchgrid.nxi,
        field.ntheta,
        field.nzeta,
    )
    N = ns * nx * nxi * nt * nz
    # slice out source/constraint terms if present
    f = f[:N]
    s = s[:N]
    # convert f from k to x in speed
    f = f.reshape((ns, nx, nxi, nt, nz, 3))
    f = jnp.einsum("xk, skitzj->sxitzj", speedgrid.xvander, f)
    f = f.reshape((N, 3))
    # form monoenergetic coefficients
    sf = s.T[:, None] * f.T[None, :]  # shape (3,3,N)
    Dij_sxitz = sf.reshape((3, 3, ns, nx, nxi, nt, nz))
    Dij_sxi = field.flux_surface_average(Dij_sxitz)
    Dij_sx = jnp.sum(Dij_sxi * pitchgrid.wxi, axis=-1)
    Dsx_ij = jnp.moveaxis(Dij_sx, (0, 1), (2, 3))

    return Dsx_ij


def compute_transport_matrix(
    Dij: jax.Array,
    speedgrid: SpeedGrid,
    species: list[LocalMaxwellian],
):
    """Compute the transport matrix for each species from monoenergetic coefficients.

    Parameters
    ----------
    Dij : jax.Array, shape(nspecies, nx, 3, 3)
        Monoenergetic transport coefficient for each species and each speed.
    speedgrid : SpeedGrid
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
