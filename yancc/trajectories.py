"""Drift Kinetic Operators without collisions."""

import cola
import equinox as eqx
import jax
import jax.numpy as jnp
import monkes

from .linalg import approx_kron, approx_sum_kron, prodkron2kronprod
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
        term1 = (
            -(1 - xi**2)
            / (2 * field.Bmag**2)
            * va
            * x
            * (field.B_sup_t * self.field.dBdt + field.B_sup_z * self.field.dBdz)
        )
        term2 = (
            xi
            * (1 - xi**2)
            / (2 * field.Bmag**3 * field.sqrtg)
            * self.Er
            * (field.B_sub_z * self.field.dBdt - field.B_sub_t * self.field.dBdz)
        )
        return term1 + term2

    def _xdot(self):
        field = self.field
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.xgrid.x[None, None, :, None, None]
        return (
            (1 + xi**2)
            * x
            / (2 * field.Bmag**3 * field.sqrtg)
            * self.Er
            * (field.B_sub_z * self.field.dBdt - field.B_sub_t * self.field.dBdz)
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
    x : jax.Array
        Values of coordinates in speed.
    xigrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    Er : float
        Radial electric field.
    """

    field: monkes.Field
    species: list[monkes._species.LocalMaxwellian]
    xigrid: PitchAngleGrid
    x: jax.Array
    Er: float
    ntheta: int
    nzeta: int
    nxi: int
    nx: int
    ns: int
    vth: jax.Array

    def __init__(self, field, species, x, xigrid, Er):

        self.field = field
        if not isinstance(species, (list, tuple)):
            species = [species]
        self.species = species
        self.vth = jnp.array([sp.v_thermal for sp in species])
        self.Er = Er
        self.ntheta = field.ntheta
        self.nzeta = field.nzeta
        self.x = jnp.atleast_1d(x)
        self.xigrid = xigrid
        self.nxi = xigrid.nxi
        self.nx = self.x.size
        self.ns = len(species)

    def _thetadot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.x[None, None, :, None, None]
        return (
            va * x * xi * field.B_sup_t / field.Bmag
            + field.B_sub_z / (field.Bmag_fsa**2 * field.sqrtg) * self.Er
        )

    def _zetadot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.x[None, None, :, None, None]
        return (
            va * x * xi * field.B_sup_z / field.Bmag
            - field.B_sub_t / (field.Bmag_fsa**2 * field.sqrtg) * self.Er
        )

    def _xidot(self):
        field = self.field
        va = self.vth[:, None, None, None, None]
        xi = self.xigrid.xi[None, :, None, None, None]
        x = self.x[None, None, :, None, None]
        out = -(1 - xi**2) / (2 * field.Bmag) * va * x * field.bdotgradB
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
        out = self._xdot() * f
        return out.reshape(shp)

    def mv(self, f):
        """Action of collisionless monoenergetic DK operator on f."""
        out1 = self._thetadotdtheta(f)
        out2 = self._zetadotdzeta(f)
        out3 = self._xidotdxi(f)
        out4 = self._xdotdx(f)
        return out1 + out2 + out3 + out4


def full_trajectories(field, speedgrid, pitchgrid, species, E_psi):
    """Full particle trajectories, as used in SFINCS."""
    rdot1 = rdot1_full_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    rdot2 = rdot2_full_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    xidot1 = xidot1_full_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    xidot2 = xidot2_full_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    xdot = xdot_full_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    return rdot1 + rdot2 + xidot1 + xidot2 + xdot


def dkes_trajectories(field, speedgrid, pitchgrid, species, E_psi):
    """Monoenergetic particle trajectories, as used in DKES/MONKES."""
    rdot1 = rdot1_full_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    rdot2 = rdot2_dkes_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    xidot = xidot1_full_trajectories(field, speedgrid, pitchgrid, species, E_psi)
    return rdot1 + rdot2 + xidot


def xdot_full_trajectories(
    field, speedgrid, pitchgrid, species, E_psi, approx_rdot=False
):
    """Term proportional to xdot in the full trajectories."""
    Is = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
    pxi2 = cola.ops.Diagonal(1 + pitchgrid.xi**2)
    xDx = cola.ops.Diagonal(speedgrid.x) @ cola.ops.Dense(
        speedgrid.xvander @ speedgrid.Dx
    )
    if approx_rdot:
        A = field.BxgradpsidotgradB / (2 * field.Bmag**3)
        Ak = approx_kron(A, False)
        xdot = cola.ops.Kronecker(cola.ops.Kronecker(E_psi * Is, xDx), pxi2, *Ak.Ms)
    else:
        BxgradpsidotgradB_over_2B3 = cola.ops.Diagonal(
            (field.BxgradpsidotgradB / (2 * field.Bmag**3)).flatten()
        )
        xdot = cola.ops.Kronecker(
            cola.ops.Kronecker(E_psi * Is, xDx), pxi2, BxgradpsidotgradB_over_2B3
        )
    return xdot


def xidot1_full_trajectories(
    field, speedgrid, pitchgrid, species, E_psi, approx_rdot=False
):
    """First xidot term in the full trajectories, bigger by a factor E*."""
    mxi2 = cola.ops.Diagonal(1 - pitchgrid.xi**2)
    Dxi = cola.ops.Dense(pitchgrid.Dxi_pseudospectral)
    xa = cola.ops.Diagonal(speedgrid.x) @ cola.ops.Dense(speedgrid.xvander)
    vth = cola.ops.Diagonal(jnp.array([s.v_thermal for s in species]))
    v = cola.ops.Kronecker(vth, xa)
    if approx_rdot:
        A = field.bdotgradB / (2 * field.Bmag)
        Ak = approx_kron(A, False)
        xidot1 = cola.ops.Kronecker(-v, mxi2 @ Dxi, *Ak.Ms)
    else:
        bdotgradB_over_2B = cola.ops.Diagonal(
            (field.bdotgradB / (2 * field.Bmag)).flatten()
        )
        xidot1 = cola.ops.Kronecker(-v, mxi2 @ Dxi, bdotgradB_over_2B)
    return xidot1


def xidot2_full_trajectories(
    field, speedgrid, pitchgrid, species, E_psi, approx_rdot=False
):
    """Second xidot term in the full trajectories, smaller by a factor E*."""
    Is = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
    Ix = cola.ops.Dense(speedgrid.xvander)
    E_I = E_psi * cola.ops.Kronecker(Is, Ix)
    xi = cola.ops.Diagonal(pitchgrid.xi)
    mxi2 = cola.ops.Diagonal(1 - pitchgrid.xi**2)
    Dxi = cola.ops.Dense(pitchgrid.Dxi_pseudospectral)
    if approx_rdot:
        A = field.BxgradpsidotgradB / (2 * field.Bmag**3)
        Ak = approx_kron(A, False)
        xidot2 = cola.ops.Kronecker(E_I, xi @ mxi2 @ Dxi, *Ak.Ms)
    else:
        BxgradpsidotgradB_over_2B3 = cola.ops.Diagonal(
            (field.BxgradpsidotgradB / (2 * field.Bmag**3)).flatten()
        )
        xidot2 = cola.ops.Kronecker(E_I, xi @ mxi2 @ Dxi, BxgradpsidotgradB_over_2B3)
    return xidot2


def rdot1_full_trajectories(
    field, speedgrid, pitchgrid, species, E_psi, approx_rdot=False
):
    """First rdot term in the full trajectories, bigger by a factor E*."""
    xa = cola.ops.Diagonal(speedgrid.x) @ cola.ops.Dense(speedgrid.xvander)
    vth = cola.ops.Diagonal(jnp.array([s.v_thermal for s in species]))
    v = cola.ops.Kronecker(vth, xa)
    xi = cola.ops.Diagonal(pitchgrid.xi)
    It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
    Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
    Dt = cola.ops.Dense(field.Dt)
    Dz = cola.ops.Dense(field.Dz)
    if approx_rdot:
        A1 = field.B_sup_t / field.Bmag
        A2 = field.B_sup_z / field.Bmag
        Ak1 = approx_kron(A1, False)
        Ak2 = approx_kron(A2, False)
        B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
        B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
        C = approx_sum_kron(B1, B2, False)
        rdot1 = cola.ops.Kronecker(v, xi, *C.Ms)
    else:
        B_sup_t_over_B = cola.ops.Diagonal((field.B_sup_t / field.Bmag).flatten())
        B_sup_z_over_B = cola.ops.Diagonal((field.B_sup_z / field.Bmag).flatten())

        rdot1_tz = B_sup_t_over_B @ cola.ops.Kronecker(
            Dt, Iz
        ) + B_sup_z_over_B @ cola.ops.Kronecker(It, Dz)

        rdot1 = cola.ops.Kronecker(v, xi, rdot1_tz)
    return rdot1


def rdot2_full_trajectories(
    field, speedgrid, pitchgrid, species, E_psi, approx_rdot=False
):
    """Second rdot term in the full trajectories, smaller by a factor E*."""
    Is = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
    Ix = cola.ops.Dense(speedgrid.xvander)
    Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
    It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
    Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
    Dt = cola.ops.Dense(field.Dt)
    Dz = cola.ops.Dense(field.Dz)
    E_I = E_psi * cola.ops.Kronecker(Is, Ix)

    if approx_rdot:
        A1 = field.B_sub_t / (field.Bmag**2 * field.sqrtg)
        A2 = field.B_sub_z / (field.Bmag**2 * field.sqrtg)
        Ak1 = approx_kron(A1, False)
        Ak2 = approx_kron(A2, False)
        B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
        B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
        C = approx_sum_kron(B1, B2, False)
        rdot2 = cola.ops.Kronecker(E_I, Ixi, *C.Ms)
    else:
        B_sub_t_over_B2 = cola.ops.Diagonal(
            (field.B_sub_t / (field.Bmag**2 * field.sqrtg)).flatten()
        )
        B_sub_z_over_B2 = cola.ops.Diagonal(
            (field.B_sub_z / (field.Bmag**2 * field.sqrtg)).flatten()
        )
        rdot2_tz = B_sub_z_over_B2 @ cola.ops.Kronecker(
            Dt, Iz
        ) - B_sub_t_over_B2 @ cola.ops.Kronecker(It, Dz)

        rdot2 = cola.ops.Kronecker(E_I, Ixi, rdot2_tz)
    return rdot2


def rdot2_dkes_trajectories(
    field, speedgrid, pitchgrid, species, E_psi, approx_rdot=False
):
    """Second rdot term in the DKES trajectories, smaller by a factor E*."""
    Is = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
    Ix = cola.ops.Dense(speedgrid.xvander)
    Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
    It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
    Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
    Dt = cola.ops.Dense(field.Dt)
    Dz = cola.ops.Dense(field.Dz)
    E_I = E_psi * cola.ops.Kronecker(Is, Ix)
    if approx_rdot:
        A1 = field.B_sub_t / (field.Bmag_fsa**2 * field.sqrtg)
        A2 = field.B_sub_z / (field.Bmag_fsa**2 * field.sqrtg)
        Ak1 = approx_kron(A1, False)
        Ak2 = approx_kron(A2, False)
        B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
        B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
        C = approx_sum_kron(B1, B2, False)
        rdot2 = cola.ops.Kronecker(E_I, Ixi, *C.Ms)
    else:
        B_sub_t_over_B2f = cola.ops.Diagonal(
            (field.B_sub_t / (field.Bmag_fsa**2 * field.sqrtg)).flatten()
        )
        B_sub_z_over_B2f = cola.ops.Diagonal(
            (field.B_sub_z / (field.Bmag_fsa**2 * field.sqrtg)).flatten()
        )
        rdot2_tz = B_sub_z_over_B2f @ cola.ops.Kronecker(
            Dt, Iz
        ) - B_sub_t_over_B2f @ cola.ops.Kronecker(It, Dz)

        rdot2 = cola.ops.Kronecker(E_I, Ixi, rdot2_tz)
    return rdot2
