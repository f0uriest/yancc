"""Drift Kinetic Operators without collisions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import monkes

from .velocity_grids import PitchAngleGrid, SpeedGrid


class SFINCSTrajectories(eqx.Module):
    """Collisionless Drift Kinetic operator, using particle trajectories from SFINCS

    AKA, "full trajectories"

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
    """

    field: monkes.Field
    species: list[monkes._species.LocalMaxwellian]
    xgrid: SpeedGrid
    xigrid: PitchAngleGrid
    Er: float
    ntheta: int
    nzeta: int
    nxi: int
    nx: int
    ns: int
    vth: jax.Array

    def __init__(self, field, species, xgrid, xigrid, Er):

        self.field = field
        if not isinstance(species, (list, tuple)):
            species = [species]
        self.species = species
        self.vth = jnp.array([sp.v_thermal for sp in species])
        self.Er = Er
        self.ntheta = field.ntheta
        self.nzeta = field.nzeta
        self.nxi = xigrid.nxi
        self.nx = xgrid.nx
        self.ns = len(species)
        self.xgrid = xgrid
        self.xigrid = xigrid

    def _thetadot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        return (
            va * x * xi * field.B_sup_t / field.Bmag
            + field.B_sub_z / (field.Bmag**2 * field.sqrtg) * self.Er
        )

    def _zetadot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        return (
            va * x * xi * field.B_sup_z / field.Bmag
            - field.B_sub_t / (field.Bmag**2 * field.sqrtg) * self.Er
        )

    def _xidot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        dBdt = field._dfdt(field.Bmag)
        dBdz = field._dfdz(field.Bmag)
        term1 = (
            -(1 - xi**2)
            / (2 * field.Bmag**2)
            * va
            * x
            * (field.B_sup_t * dBdt + field.B_sup_z * dBdz)
        )
        term2 = (
            xi
            * (1 - xi**2)
            / (2 * field.Bmag**3 * field.sqrtg)
            * self.Er
            * (field.B_sub_z * dBdt - field.B_sub_t * dBdz)
        )
        return term1 + term2

    def _xdot(self):
        field = self.field
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        dBdt = field._dfdt(field.Bmag)
        dBdz = field._dfdz(field.Bmag)
        return (
            (1 + xi**2)
            * x
            / (2 * field.Bmag**3 * field.sqrtg)
            * self.Er
            * (field.B_sub_z * dBdt - field.B_sub_t * dBdz)
        )

    def _thetadotdtheta(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._thetadot() * self.field._dfdt(f)
        return out.reshape(shp)

    def _zetadotdzeta(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._zetadot() * self.field._dfdz(f)
        return out.reshape(shp)

    def _xidotdxi(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._xidot() * jax.vmap(self.xigrid._dfdxi)(f)
        return out.reshape(shp)

    def _xdotdx(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._xdot() * jax.vmap(self.xgrid._dfdx)(f)
        return out.reshape(shp)

    def mv(self, f):
        """Action of collisionless Drift Kinetic operator on f.

        rdot * grad(f) + xidot * df/dxi + xdot * df/dx

        """
        out1 = self._thetadotdtheta(f)
        out2 = self._zetadotdzeta(f)
        out3 = self._xidotdxi(f)
        out4 = self._xdotdx(f)
        return out1 + out2 + out3 + out4


class DKESTrajectories(eqx.Module):
    """Collisionless Monoenergetic Drift Kinetic operator, using trajectories from DKES.

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
    """

    field: monkes.Field
    species: list[monkes._species.LocalMaxwellian]
    xigrid: PitchAngleGrid
    xgrid: SpeedGrid
    Er: float
    ntheta: int
    nzeta: int
    nxi: int
    nx: int
    ns: int
    vth: jax.Array

    def __init__(self, field, species, xgrid, xigrid, Er):

        self.field = field
        if not isinstance(species, (list, tuple)):
            species = [species]
        self.species = species
        self.vth = jnp.array([sp.v_thermal for sp in species])
        self.Er = Er
        self.ntheta = field.ntheta
        self.nzeta = field.nzeta
        self.nxi = xigrid.nxi
        self.nx = xgrid.nx
        self.ns = len(species)
        self.xgrid = xgrid
        self.xigrid = xigrid

    def _thetadot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        return (
            va * x * xi * field.B_sup_t / field.Bmag
            + field.B_sub_z / (field.Bmag_fsa**2 * field.sqrtg) * self.Er
        )

    def _zetadot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        return (
            va * x * xi * field.B_sup_z / field.Bmag
            - field.B_sub_t / (field.Bmag_fsa**2 * field.sqrtg) * self.Er
        )

    def _xidot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        dBdt = field._dfdt(field.Bmag)
        dBdz = field._dfdz(field.Bmag)
        out = (
            -(1 - xi**2)
            / (2 * field.Bmag**2)
            * va
            * x
            * (field.B_sup_t * dBdt + field.B_sup_z * dBdz)
        )
        return out

    def _xdot(self):
        return jnp.array(0.0)

    def _thetadotdtheta(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._thetadot() * self.field._dfdt(f)
        return out.reshape(shp)

    def _zetadotdzeta(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._zetadot() * self.field._dfdz(f)
        return out.reshape(shp)

    def _xidotdxi(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._xidot() * jax.vmap(self.xigrid._dfdxi)(f)
        return out.reshape(shp)

    def _xdotdx(self, f):
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._xdot()  # no need for differential operator
        return out.reshape(shp)

    def mv(self, f):
        """Action of collisionless monoenergetic DK operator on f."""
        out1 = self._thetadotdtheta(f)
        out2 = self._zetadotdzeta(f)
        out3 = self._xidotdxi(f)
        out4 = self._xdotdx(f)
        return out1 + out2 + out3 + out4
