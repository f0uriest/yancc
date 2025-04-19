"""Drift Kinetic Operators without collisions."""

import functools

import cola
import jax
import jax.numpy as jnp
import numpy as np
from monkes import LocalMaxwellian

from .field import Field
from .finite_diff import fd2, fd_coeffs, fdbwd, fdfwd
from .linalg import approx_kron_diag2d, approx_sum_kron, prodkron2kronprod
from .velocity_grids import PitchAngleGrid, SpeedGrid


class FullTrajectories(cola.ops.Sum):
    """Collisionless Drift Kinetic operator, using particle trajectories from SFINCS

    AKA, "full trajectories"

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
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        normalize: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.normalize = normalize

        rdot1 = FullTrajectoriesSurface1(
            field, speedgrid, pitchgrid, species, E_psi, normalize=normalize
        )
        rdot2 = FullTrajectoriesSurface2(
            field, speedgrid, pitchgrid, species, E_psi, normalize=normalize
        )
        xidot1 = FullTrajectoriesPitch1(
            field, speedgrid, pitchgrid, species, E_psi, normalize=normalize
        )
        xidot2 = FullTrajectoriesPitch2(
            field, speedgrid, pitchgrid, species, E_psi, normalize=normalize
        )
        xdot = FullTrajectoriesSpeed(
            field, speedgrid, pitchgrid, species, E_psi, normalize=normalize
        )
        super().__init__(rdot1, rdot2, xidot1, xidot2, xdot)


class DKESTrajectories(cola.ops.Sum):
    """Collisionless Monoenergetic Drift Kinetic operator, using trajectories from DKES.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : LocalMaxwellian
        Species being considered
    v : float
        Speed being considered.
    E_psi : float
        Radial electric field.
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
        normalize: bool = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi
        self.normalize = normalize

        rdot1 = DKESTrajectoriesSurface1(
            field, pitchgrid, species, v, E_psi, normalize=normalize
        )
        rdot2 = DKESTrajectoriesSurface2(
            field, pitchgrid, species, v, E_psi, normalize=normalize
        )
        xidot = DKESTrajectoriesPitch(
            field, pitchgrid, species, v, E_psi, normalize=normalize
        )
        super().__init__(rdot1, rdot2, xidot)


class FullTrajectoriesSpeed(cola.ops.Kronecker):
    """Term including df/dx in the full trajectories.

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
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        if normalize:
            vth = jnp.array([sp.v_thermal for sp in species])
        else:
            vth = jnp.ones(len(species))
        Is = cola.ops.Diagonal(E_psi / vth)
        pxi2 = cola.ops.Diagonal(1 + pitchgrid.xi**2)
        xDx = cola.ops.Diagonal(speedgrid.x) @ cola.ops.Dense(
            speedgrid.xvander @ speedgrid.Dx
        )
        if approx_rdot:
            A = field.BxgradpsidotgradB / (2 * field.Bmag**3)
            Ak = approx_kron_diag2d(A.flatten(), *A.shape)
            Ms = (cola.ops.Kronecker(E_psi * Is, xDx), pxi2, *Ak.Ms)
        else:
            BxgradpsidotgradB_over_2B3 = cola.ops.Diagonal(
                (field.BxgradpsidotgradB / (2 * field.Bmag**3)).flatten()
            )
            Ms = (cola.ops.Kronecker(Is, xDx), pxi2, BxgradpsidotgradB_over_2B3)
        super().__init__(*Ms)


class FullTrajectoriesPitch1(cola.ops.Kronecker):
    """First df/dxi term in the full trajectories, bigger by a factor E*.

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
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        mxi2 = cola.ops.Diagonal(1 - pitchgrid.xi**2)
        Dxi = cola.ops.Dense(pitchgrid.Dxi_pseudospectral)
        xa = cola.ops.Diagonal(speedgrid.x) @ cola.ops.Dense(speedgrid.xvander)
        if normalize:
            vth = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
        else:
            vth = cola.ops.Diagonal(jnp.array([s.v_thermal for s in species]))
        v = cola.ops.Kronecker(vth, xa)
        if approx_rdot:
            A = field.bdotgradB / (2 * field.Bmag)
            Ak = approx_kron_diag2d(A.flatten(), *A.shape)
            Ms = (-v, mxi2 @ Dxi, *Ak.Ms)
        else:
            bdotgradB_over_2B = cola.ops.Diagonal(
                (field.bdotgradB / (2 * field.Bmag)).flatten()
            )
            Ms = (-v, mxi2 @ Dxi, bdotgradB_over_2B)
        super().__init__(*Ms)


class FullTrajectoriesPitch2(cola.ops.Kronecker):
    """Second df/dxi term in the full trajectories, smaller by a factor E*.

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
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        if normalize:
            vth = jnp.array([sp.v_thermal for sp in species])
        else:
            vth = jnp.ones(len(species))
        Is = cola.ops.Diagonal(E_psi / vth)
        Ix = cola.ops.Dense(speedgrid.xvander)
        E_I = cola.ops.Kronecker(Is, Ix)
        xi = cola.ops.Diagonal(pitchgrid.xi)
        mxi2 = cola.ops.Diagonal(1 - pitchgrid.xi**2)
        Dxi = cola.ops.Dense(pitchgrid.Dxi_pseudospectral)
        if approx_rdot:
            A = field.BxgradpsidotgradB / (2 * field.Bmag**3)
            Ak = approx_kron_diag2d(A.flatten(), *A.shape)
            Ms = (E_I, xi @ mxi2 @ Dxi, *Ak.Ms)
        else:
            BxgradpsidotgradB_over_2B3 = cola.ops.Diagonal(
                (field.BxgradpsidotgradB / (2 * field.Bmag**3)).flatten()
            )
            Ms = (E_I, xi @ mxi2 @ Dxi, BxgradpsidotgradB_over_2B3)
        super().__init__(*Ms)


class FullTrajectoriesSurface1(cola.ops.Kronecker):
    """First df/dr term in the full trajectories, bigger by a factor E*.

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
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        xa = cola.ops.Diagonal(speedgrid.x) @ cola.ops.Dense(speedgrid.xvander)
        if normalize:
            vth = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
        else:
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
            Ak1 = approx_kron_diag2d(A1.flatten(), *A1.shape)
            Ak2 = approx_kron_diag2d(A2.flatten(), *A2.shape)
            B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
            B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
            C = approx_sum_kron((B1, B2))
            Ms = (v, xi, *C.Ms)
        else:
            B_sup_t_over_B = cola.ops.Diagonal((field.B_sup_t / field.Bmag).flatten())
            B_sup_z_over_B = cola.ops.Diagonal((field.B_sup_z / field.Bmag).flatten())

            rdot1_tz = B_sup_t_over_B @ cola.ops.Kronecker(
                Dt, Iz
            ) + B_sup_z_over_B @ cola.ops.Kronecker(It, Dz)

            Ms = (v, xi, rdot1_tz)
        super().__init__(*Ms)


class FullTrajectoriesSurface2(cola.ops.Kronecker):
    """Second df/dr term in the full trajectories, smaller by a factor E*.

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
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by thermal speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        if normalize:
            vth = jnp.array([sp.v_thermal for sp in species])
        else:
            vth = jnp.ones(len(species))
        Is = cola.ops.Diagonal(E_psi / vth)
        Ix = cola.ops.Dense(speedgrid.xvander)
        Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        Dt = cola.ops.Dense(field.Dt)
        Dz = cola.ops.Dense(field.Dz)
        E_I = cola.ops.Kronecker(Is, Ix)

        if approx_rdot:
            A1 = field.B_sub_t / (field.Bmag**2 * field.sqrtg)
            A2 = field.B_sub_z / (field.Bmag**2 * field.sqrtg)
            Ak1 = approx_kron_diag2d(A1.flatten(), *A1.shape)
            Ak2 = approx_kron_diag2d(A2.flatten(), *A2.shape)
            B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
            B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
            C = approx_sum_kron((B1, B2))
            Ms = (E_I, Ixi, *C.Ms)
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

            Ms = (E_I, Ixi, rdot2_tz)
        super().__init__(*Ms)


class DKESTrajectoriesPitch(cola.ops.Kronecker):
    """Term including df/dxi in the DKES trajectories.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : LocalMaxwellian
        Species being considered.
    v : float
        Speed being considered.
    E_psi : float
        Radial electric field.
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        if normalize:
            V = cola.ops.Dense(-jnp.atleast_2d(jnp.ones_like(v)))
        else:
            V = cola.ops.Dense(-jnp.atleast_2d(v))
        mxi2 = cola.ops.Diagonal(1 - pitchgrid.xi**2)
        Dxi = cola.ops.Dense(pitchgrid.Dxi_pseudospectral)
        if approx_rdot:
            A = field.bdotgradB / (2 * field.Bmag)
            Ak = approx_kron_diag2d(A.flatten(), *A.shape)
            Ms = (V, mxi2 @ Dxi, *Ak.Ms)
        else:
            bdotgradB_over_2B = cola.ops.Diagonal(
                (field.bdotgradB / (2 * field.Bmag)).flatten()
            )
            Ms = (V, mxi2 @ Dxi, bdotgradB_over_2B)
        super().__init__(*Ms)


class DKESTrajectoriesSurface1(cola.ops.Kronecker):
    """First df/dr term in the DKES trajectories, larger by a factor E*.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : LocalMaxwellian
        Species being considered.
    v : float
        Speed being considered.
    E_psi : float
        Radial electric field.
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        if normalize:
            V = cola.ops.Dense(jnp.atleast_2d(jnp.ones_like(v)))
        else:
            V = cola.ops.Dense(jnp.atleast_2d(v))
        xi = cola.ops.Diagonal(pitchgrid.xi)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        Dt = cola.ops.Dense(field.Dt)
        Dz = cola.ops.Dense(field.Dz)
        if approx_rdot:
            A1 = field.B_sup_t / field.Bmag
            A2 = field.B_sup_z / field.Bmag
            Ak1 = approx_kron_diag2d(A1.flatten(), *A1.shape)
            Ak2 = approx_kron_diag2d(A2.flatten(), *A2.shape)
            B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
            B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
            C = approx_sum_kron((B1, B2))
            Ms = (V, xi, *C.Ms)
        else:
            B_sup_t_over_B = cola.ops.Diagonal((field.B_sup_t / field.Bmag).flatten())
            B_sup_z_over_B = cola.ops.Diagonal((field.B_sup_z / field.Bmag).flatten())

            rdot1_tz = B_sup_t_over_B @ cola.ops.Kronecker(
                Dt, Iz
            ) + B_sup_z_over_B @ cola.ops.Kronecker(It, Dz)

            Ms = (V, xi, rdot1_tz)
        super().__init__(*Ms)


class DKESTrajectoriesSurface2(cola.ops.Kronecker):
    """Second df/dr term in the DKES trajectories, smaller by a factor E*.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : LocalMaxwellian
        Species being considered.
    v : float
        Speed being considered.
    E_psi : float
        Radial electric field.
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.
    normalize : bool
        Whether to divide equations by speed to non-dimensionalize
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
        approx_rdot: bool = False,
        normalize: bool = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot
        self.normalize = normalize

        if normalize:
            E = cola.ops.Dense(-E_psi / jnp.atleast_2d(v))
        else:
            E = cola.ops.Dense(-E_psi / jnp.atleast_2d(jnp.ones_like(v)))
        Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        Dt = cola.ops.Dense(field.Dt)
        Dz = cola.ops.Dense(field.Dz)
        if approx_rdot:
            A1 = field.B_sub_t / (field.B2mag_fsa * field.sqrtg)
            A2 = field.B_sub_z / (field.B2mag_fsa * field.sqrtg)
            Ak1 = approx_kron_diag2d(A1.flatten(), *A1.shape)
            Ak2 = approx_kron_diag2d(A2.flatten(), *A2.shape)
            B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
            B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
            C = approx_sum_kron((B1, B2))
            Ms = (E, Ixi, *C.Ms)
        else:
            B_sub_t_over_B2f = cola.ops.Diagonal(
                (field.B_sub_t / (field.B2mag_fsa * field.sqrtg)).flatten()
            )
            B_sub_z_over_B2f = cola.ops.Diagonal(
                (field.B_sub_z / (field.B2mag_fsa * field.sqrtg)).flatten()
            )
            rdot2_tz = B_sub_z_over_B2f @ cola.ops.Kronecker(
                Dt, Iz
            ) - B_sub_t_over_B2f @ cola.ops.Kronecker(It, Dz)

            Ms = (E, Ixi, rdot2_tz)
        super().__init__(*Ms)


def _parse_axorder_shape(nt, nz, na, axorder):
    shape = np.arange(3)
    shape[axorder.index("a")] = na
    shape[axorder.index("t")] = nt
    shape[axorder.index("z")] = nz
    caxorder = (axorder.index("a"), axorder.index("t"), axorder.index("z"))
    return tuple(shape), caxorder


def w_theta(field, pitchgrid, E_psi):
    """Wind in theta direction for MDKE."""
    w = (
        field.B_sup_t / field.Bmag * pitchgrid.xi[:, None, None]
        + field.B_sub_z / field.B2mag_fsa / field.sqrtg * E_psi
    )
    return w


def w_zeta(field, pitchgrid, E_psi):
    """Wind in zeta direction for MDKE."""
    w = (
        field.B_sup_z / field.Bmag * pitchgrid.xi[:, None, None]
        - field.B_sub_t / field.B2mag_fsa / field.sqrtg * E_psi
    )
    return w


def w_pitch(field, pitchgrid, nu):
    """Wind in xi/pitch direction for MDKE, including first order scattering term."""
    sina = jnp.sqrt(1 - pitchgrid.xi**2)
    cosa = -pitchgrid.xi
    w = (
        -field.bdotgradB
        / (2 * field.Bmag)
        * (1 - pitchgrid.xi[:, None, None] ** 2)
        / sina[:, None, None]
    )
    w -= (nu * cosa / sina)[:, None, None]
    return w


@functools.partial(jax.jit, static_argnames=["axorder", "p"])
def dfdtheta(
    f,
    field,
    pitchgrid,
    E_psi,
    axorder="atz",
    p="1a",
    diag=False,
    flip=False,
    gauge=False,
):
    """Advection operator in theta direction.

    Parameters
    ----------
    f : jax.Array
        Distribution function.
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    diag : bool
        If True, only apply the diagonal part of the operator.
    flip : bool
        If True, assume f is ordered backwards in each coordinate.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    dfdtheta : jax.Array
        Distribution function advected along theta.
    """
    assert field.ntheta > fd_coeffs[1][p].size
    shp = f.shape
    shape, caxorder = _parse_axorder_shape(
        field.ntheta, field.nzeta, pitchgrid.nxi, axorder
    )
    f = f.reshape(shape)
    f = jnp.where(flip, f[..., ::-1], f)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2))
    w = w_theta(field, pitchgrid, E_psi)
    h = 2 * np.pi / field.ntheta

    fd_diag = (
        lambda f: jnp.diag(jax.jacfwd(fdfwd)(f[0, :, 0], p, h=h, bc="periodic"))[
            None, :, None
        ]
        * f
    )
    bd_diag = (
        lambda f: jnp.diag(jax.jacfwd(fdbwd)(f[0, :, 0], p, h=h, bc="periodic"))[
            None, :, None
        ]
        * f
    )
    fd_full = lambda f: fdfwd(f, p, h=h, bc="periodic", axis=1)
    bd_full = lambda f: fdbwd(f, p, h=h, bc="periodic", axis=1)
    fd = jax.lax.cond(diag, fd_diag, fd_full, f)
    bd = jax.lax.cond(diag, bd_diag, bd_full, f)
    # get only L or U by only taking forward or backward diff? + diagonal correction
    df = w * ((w > 0) * bd + (w <= 0) * fd)
    idx = pitchgrid.nxi // 2
    df = jnp.where(gauge, df.at[idx, 0, 0].set(f[idx, 0, 0]), df)
    df = jnp.moveaxis(df, (0, 1, 2), caxorder)
    df = jnp.where(flip, df[..., ::-1], df)
    return df.reshape(shp)


@functools.partial(jax.jit, static_argnames=["axorder", "p"])
def dfdzeta(
    f,
    field,
    pitchgrid,
    E_psi,
    axorder="atz",
    p="1a",
    diag=False,
    flip=False,
    gauge=False,
):
    """Advection operator in zeta direction.

    Parameters
    ----------
    f : jax.Array
        Distribution function.
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    diag : bool
        If True, only apply the diagonal part of the operator.
    flip : bool
        If True, assume f is ordered backwards in each coordinate.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    dfdzeta : jax.Array
        Distribution function advected along zeta.
    """
    assert field.nzeta > fd_coeffs[1][p].size
    shp = f.shape
    shape, caxorder = _parse_axorder_shape(
        field.ntheta, field.nzeta, pitchgrid.nxi, axorder
    )
    f = f.reshape(shape)
    f = jnp.where(flip, f[..., ::-1], f)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2))
    w = w_zeta(field, pitchgrid, E_psi)
    h = 2 * np.pi / field.nzeta / field.NFP
    fd_diag = (
        lambda f: jnp.diag(jax.jacfwd(fdfwd)(f[0, 0, :], p, h=h, bc="periodic"))[
            None, None, :
        ]
        * f
    )
    bd_diag = (
        lambda f: jnp.diag(jax.jacfwd(fdbwd)(f[0, 0, :], p, h=h, bc="periodic"))[
            None, None, :
        ]
        * f
    )
    fd_full = lambda f: fdfwd(f, p, h=h, bc="periodic", axis=2)
    bd_full = lambda f: fdbwd(f, p, h=h, bc="periodic", axis=2)
    fd = jax.lax.cond(diag, fd_diag, fd_full, f)
    bd = jax.lax.cond(diag, bd_diag, bd_full, f)
    df = w * ((w > 0) * bd + (w <= 0) * fd)
    idx = pitchgrid.nxi // 2
    df = jnp.where(gauge, df.at[idx, 0, 0].set(f[idx, 0, 0]), df)
    df = jnp.moveaxis(df, (0, 1, 2), caxorder)
    df = jnp.where(flip, df[..., ::-1], df)
    return df.reshape(shp)


@functools.partial(jax.jit, static_argnames=["axorder", "p"])
def dfdxi(
    f,
    field,
    pitchgrid,
    nu,
    axorder="atz",
    p="1a",
    diag=False,
    flip=False,
    gauge=False,
):
    """Advection operator in xi/pitch direction.

    Parameters
    ----------
    f : jax.Array
        Distribution function.
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    nu : float
        Normalized collisionality, nu/v
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    diag : bool
        If True, only apply the diagonal part of the operator.
    flip : bool
        If True, assume f is ordered backwards in each coordinate.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    dfdxi : jax.Array
        Distribution function advected along xi.
    """
    assert pitchgrid.nxi > fd_coeffs[1][p].size
    shp = f.shape
    shape, caxorder = _parse_axorder_shape(
        field.ntheta, field.nzeta, pitchgrid.nxi, axorder
    )
    f = f.reshape(shape)
    f = jnp.where(flip, f[..., ::-1], f)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2))
    w = w_pitch(field, pitchgrid, nu)
    h = np.pi / pitchgrid.nxi

    fd_diag = (
        lambda f: jnp.diag(jax.jacfwd(fdfwd)(f[:, 0, 0], p, h=h, bc="symmetric"))[
            :, None, None
        ]
        * f
    )
    bd_diag = (
        lambda f: jnp.diag(jax.jacfwd(fdbwd)(f[:, 0, 0], p, h=h, bc="symmetric"))[
            :, None, None
        ]
        * f
    )
    fd_full = lambda f: fdfwd(f, p, h=h, bc="symmetric", axis=0)
    bd_full = lambda f: fdbwd(f, p, h=h, bc="symmetric", axis=0)
    fd = jax.lax.cond(diag, fd_diag, fd_full, f)
    bd = jax.lax.cond(diag, bd_diag, bd_full, f)
    df = w * ((w > 0) * bd + (w <= 0) * fd)
    idx = pitchgrid.nxi // 2
    df = jnp.where(gauge, df.at[idx, 0, 0].set(f[idx, 0, 0]), df)
    df = jnp.moveaxis(df, (0, 1, 2), caxorder)
    df = jnp.where(flip, df[..., ::-1], df)
    return df.reshape(shp)


@functools.partial(jax.jit, static_argnames=["axorder", "p"])
def dfdpitch(
    f,
    field,
    pitchgrid,
    nu,
    axorder="atz",
    p=2,
    diag=False,
    flip=False,
    gauge=False,
):
    """Diffusion operator in xi/pitch direction.

    Parameters
    ----------
    f : jax.Array
        Distribution function.
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    nu : float
        Normalized collisionality, nu/v
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p : int
        Order of approximation for derivatives.
    diag : bool
        If True, only apply the diagonal part of the operator.
    flip : bool
        If True, assume f is ordered backwards in each coordinate.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    dfdpitch : jax.Array
        Distribution function diffused along xi/pitch.
    """
    assert pitchgrid.nxi > p
    shp = f.shape
    shape, caxorder = _parse_axorder_shape(
        field.ntheta, field.nzeta, pitchgrid.nxi, axorder
    )
    f = f.reshape(shape)
    f = jnp.where(flip, f[..., ::-1], f)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2))
    h = np.pi / pitchgrid.nxi

    fd_diag = (
        lambda f: jnp.diag(jax.jacfwd(fd2)(f[:, 0, 0], p, h=h, bc="symmetric"))[
            :, None, None
        ]
        * f
    )
    fd_full = lambda f: fd2(f, p, h=h, bc="symmetric", axis=0)
    ddf = jax.lax.cond(diag, fd_diag, fd_full, f)
    idx = pitchgrid.nxi // 2
    ddf = jnp.where(gauge, ddf.at[idx, 0, 0].set(-f[idx, 0, 0]), ddf)
    ddf = jnp.moveaxis(ddf, (0, 1, 2), caxorder)
    ddf = jnp.where(flip, ddf[..., ::-1], ddf)
    return -nu * ddf.reshape(shp)


@functools.partial(jax.jit, static_argnames=["axorder", "p1", "p2"])
def mdke(
    f,
    field,
    pitchgrid,
    E_psi,
    nu,
    axorder="atz",
    p1="1a",
    p2=2,
    flip=False,
    gauge=False,
):
    """MDKE operator.

    Parameters
    ----------
    f : jax.Array
        Distribution function.
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    nu : float
        Normalized collisionality, nu/v
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    flip : bool
        If True, assume f is ordered backwards in each coordinate.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    df : jax.Array

    """
    dt = dfdtheta(f, field, pitchgrid, E_psi, axorder, p1, flip=flip, gauge=gauge)
    dz = dfdzeta(f, field, pitchgrid, E_psi, axorder, p1, flip=flip, gauge=gauge)
    di = dfdxi(f, field, pitchgrid, nu, axorder, p1, flip=flip, gauge=gauge)
    dp = dfdpitch(f, field, pitchgrid, nu, axorder, p2, flip=flip, gauge=gauge)
    return dt + dz + di + dp
