"""Constraints, sources, RHS, etc."""

import equinox as eqx
import jax
import jax.numpy as jnp
import monkes

from .velocity_grids import PitchAngleGrid, SpeedGrid


class SFINCSSources(eqx.Module):
    """Fake sources of particles and momentum to ensure solvability

    Parameters
    ----------
    field : Field
        Magnetic field information
    species : list[LocalMaxwellian]
        Species being considered
    xgrid : SpeedGrid
        Grid of coordinates in speed.
    xigrid : PitchAngleGrid
        Grid of coordinates in pitch angle.

    """

    field: monkes.Field
    species: list[monkes._species.LocalMaxwellian]
    xgrid: SpeedGrid
    xigrid: PitchAngleGrid
    s1: jax.Array
    s2: jax.Array
    F: jax.Array

    def __init__(self, field, species, xgrid, xigrid):
        self.field = field
        self.species = species
        self.xgrid = xgrid
        self.xigrid = xigrid

        x = xgrid.x[None, None, :, None, None] * jnp.ones(
            (len(species), xigrid.nxi, 1, field.ntheta, field.nzeta)
        )
        self.s1 = -(x**2) + 5 / 2
        self.s2 = 2 / 3 * x**2 - 1
        self.F = jnp.array([sp(x[0] * sp.v_thermal) for sp in species])

    def mv(self, S):
        """Matrix vector product."""
        shp = S.shape
        S = S.reshape((2, len(self.species)))
        S1 = S[0, :, None, None, None, None]
        S2 = S[1, :, None, None, None, None]
        out = (self.s1 * S1 + self.s2 * S2) * self.F
        if len(shp) == 1:
            out = out.flatten()
        return out


class SFINCSConstraint(eqx.Module):
    """Constraints to fix gauge freedom in density and momentum.

    Parameters
    ----------
    field : Field
        Magnetic field information
    species : list[LocalMaxwellian]
        Species being considered
    xgrid : SpeedGrid
        Grid of coordinates in speed.
    xigrid : PitchAngleGrid
        Grid of coordinates in pitch angle.

    """

    field: monkes.Field
    species: list[monkes._species.LocalMaxwellian]
    xgrid: SpeedGrid
    xigrid: PitchAngleGrid
    ntheta: int
    nzeta: int
    nxi: int
    nx: int
    ns: int
    vth: jax.Array

    def __init__(self, field, species, xgrid, xigrid):

        self.field = field
        if not isinstance(species, (list, tuple)):
            species = [species]
        self.species = species
        self.ntheta = field.ntheta
        self.nzeta = field.nzeta
        self.nxi = xigrid.nxi
        self.nx = xgrid.nx
        self.ns = len(species)
        self.xgrid = xgrid
        self.xigrid = xigrid
        self.vth = jnp.array([sp.v_thermal for sp in species])

    def mv(self, f):
        """Matrix vector product."""
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        wx = self.xgrid.wx[None, None, :, None, None]
        wxi = self.xigrid.wxi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        vth = self.vth[:, None, None, None, None]
        intf = jnp.sum(f * wx * wxi, axis=(1, 2))
        intv2f = jnp.sum(f * x**2 * vth**2 * wx * wxi, axis=(1, 2))
        out = jnp.array(
            [
                self.field.flux_surface_average(intf),
                self.field.flux_surface_average(intv2f),
            ]
        )
        if len(shp) == 1:
            out = out.flatten()
        return out


def dke_rhs(
    field: monkes.Field,
    species: list[monkes._species.LocalMaxwellian],
    xgrid: SpeedGrid,
    xigrid: PitchAngleGrid,
    Er: float,
) -> jax.Array:
    """RHS of DKE as solved in SFINCS.

    Parameters
    ----------
    field : Field
        Magnetic field information
    species : list[LocalMaxwellian]
        Species being considered
    xgrid : SpeedGrid
        Grid of coordinates in speed.
    xigrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    Er : float
        Radial electric field.

    Returns
    -------
    f : jax.Array
        RHS of linear DKE.
    """
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
    ms = jnp.array([sp.species.mass for sp in species])[:, None, None, None, None]
    qs = jnp.array([sp.species.charge for sp in species])[:, None, None, None, None]
    ns = jnp.array([sp.density for sp in species])[:, None, None, None, None]
    dns = jnp.array([sp.dndr for sp in species])[:, None, None, None, None]
    Ts = jnp.array([sp.temperature for sp in species])[:, None, None, None, None]
    dTs = jnp.array([sp.dTdr for sp in species])[:, None, None, None, None]
    xi = xigrid.xi[None, :, None, None, None]
    x = xgrid.x[None, None, :, None, None]
    vmadotgradpsi = (
        x**2
        * vth**2
        * (1 / 2 + xi**2 / 2)
        * ms
        / qs
        / field.Bmag**2
        * field.Bxgradpsidotgrad(field.Bmag)
    )
    gradients = 1 / ns * dns + qs * Er / Ts + (x**2 - 3 / 2) / Ts * dTs
    return vmadotgradpsi * gradients
