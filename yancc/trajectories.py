"""Drift Kinetic Operators without collisions."""

import cola
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from .field import Field
from .finite_diff import fd2, fd_coeffs, fdbwd, fdfwd
from .linalg import approx_kron_diag2d, approx_sum_kron, prodkron2kronprod
from .species import LocalMaxwellian
from .velocity_grids import LegendrePitchAngleGrid, SpeedGrid, UniformPitchAngleGrid


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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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
        pitchgrid: LegendrePitchAngleGrid,
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


def w_pitch(field, pitchgrid):
    """Wind in xi/pitch direction for MDKE."""
    sina = jnp.sqrt(1 - pitchgrid.xi**2)
    w = (
        -field.bdotgradB
        / (2 * field.Bmag)
        * (1 - pitchgrid.xi[:, None, None] ** 2)
        / sina[:, None, None]
    )
    return w


class MDKETheta(lx.AbstractLinearOperator):
    """Advection operator in theta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    nu : float
        Normalized collisionality, nu/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: float
    nu: float
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: bool
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field,
        pitchgrid,
        E_psi,
        nu,
        p1="1a",
        p2=2,
        axorder="atz",
        gauge=True,
    ):
        assert field.ntheta > fd_coeffs[1][p1].size // 2
        assert field.ntheta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.nu = jnp.array(nu)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, f):
        """Matrix vector product."""
        shp = f.shape
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        w = w_theta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.ntheta

        fd = fdfwd(f, self.p1, h=h, bc="periodic", axis=1)
        bd = fdbwd(f, self.p1, h=h, bc="periodic", axis=1)
        # get only L or U by only taking forward or backward diff? + diagonal correction
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = w_theta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.ntheta
        fd = jnp.diag(jax.jacfwd(fdfwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = w_theta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.ntheta
        fd = (jax.jacfwd(fdfwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None, :
        ]
        bd = (jax.jacfwd(fdbwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None, :
        ]
        w = w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, 0].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.field.ntheta, self.field.ntheta))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class MDKEZeta(lx.AbstractLinearOperator):
    """Advection operator in zeta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    nu : float
        Normalized collisionality, nu/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: float
    nu: float
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: bool
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field,
        pitchgrid,
        E_psi,
        nu,
        p1="1a",
        p2=2,
        axorder="atz",
        gauge=True,
    ):
        assert field.nzeta > fd_coeffs[1][p1].size // 2
        assert field.nzeta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.nu = jnp.array(nu)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, f):
        """Matrix vector product."""
        shp = f.shape
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        w = w_zeta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.nzeta / self.field.NFP

        fd = fdfwd(f, self.p1, h=h, bc="periodic", axis=2)
        bd = fdbwd(f, self.p1, h=h, bc="periodic", axis=2)
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = w_zeta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.nzeta / self.field.NFP
        fd = jnp.diag(jax.jacfwd(fdfwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = w_zeta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.nzeta / self.field.NFP
        fd = (jax.jacfwd(fdfwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :, :
        ]
        bd = (jax.jacfwd(fdbwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :, :
        ]
        w = w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, 0].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.field.nzeta, self.field.nzeta))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class MDKEPitch(lx.AbstractLinearOperator):
    """Advection operator in pitch angle direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    nu : float
        Normalized collisionality, nu/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: float
    nu: float
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: bool
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field,
        pitchgrid,
        E_psi,
        nu,
        p1="1a",
        p2=2,
        axorder="atz",
        gauge=True,
    ):
        assert pitchgrid.nxi > fd_coeffs[1][p1].size // 2
        assert pitchgrid.nxi > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.nu = jnp.array(nu)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, f):
        """Matrix vector product."""
        shp = f.shape
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        w = w_pitch(self.field, self.pitchgrid)
        h = np.pi / self.pitchgrid.nxi

        fd = fdfwd(f, self.p1, h=h, bc="symmetric", axis=0)
        bd = fdbwd(f, self.p1, h=h, bc="symmetric", axis=0)
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = w_pitch(self.field, self.pitchgrid)
        h = np.pi / self.pitchgrid.nxi
        fd = jnp.diag(jax.jacfwd(fdfwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = w_pitch(self.field, self.pitchgrid)
        h = np.pi / self.pitchgrid.nxi
        fd = (jax.jacfwd(fdfwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None, :
        ]
        bd = (jax.jacfwd(fdbwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None, :
        ]
        w = w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, idx].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.pitchgrid.nxi, self.pitchgrid.nxi))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class MDKEPitchAngleScattering(lx.AbstractLinearOperator):
    """Diffusion operator in xi direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    nu : float
        Normalized collisionality, nu/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: float
    nu: float
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: bool
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field,
        pitchgrid,
        E_psi,
        nu,
        p1="1a",
        p2=2,
        axorder="atz",
        gauge=True,
    ):
        assert pitchgrid.nxi > fd_coeffs[1][p1].size // 2
        assert pitchgrid.nxi > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.nu = jnp.array(nu)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, f):
        """Matrix vector product."""
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi
        shp = f.shape
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        h = np.pi / self.pitchgrid.nxi

        f1 = fdfwd(f, str(self.p2) + "z", h=h, bc="symmetric", axis=0)
        f1 *= -(self.nu * cosa / sina)[:, None, None]
        f2 = fd2(f, self.p2, h=h, bc="symmetric", axis=0)
        f2 *= -self.nu
        df = f1 + f2

        idx = self.pitchgrid.nxi // 2
        scale = self.nu / h**2
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi

        h = np.pi / self.pitchgrid.nxi

        f1 = jnp.diag(
            jax.jacfwd(fdfwd)(
                f[:, 0, 0], str(self.p2) + "z", h=h, bc="symmetric", axis=0
            )
        )[:, None, None]
        f1 *= -(self.nu * cosa / sina)[:, None, None]
        f2 = jnp.diag(
            jax.jacfwd(fd2)(f[:, 0, 0], self.p2, h=h, bc="symmetric", axis=0)
        )[:, None, None]
        f2 *= -self.nu
        df = f1 + f2
        df = jnp.tile(df, (1, self.field.ntheta, self.field.nzeta))

        idx = self.pitchgrid.nxi // 2
        scale = self.nu / h**2
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        shape, caxorder = _parse_axorder_shape(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi

        h = np.pi / self.pitchgrid.nxi

        f1 = jax.jacfwd(fdfwd)(
            f[:, 0, 0], str(self.p2) + "z", h=h, bc="symmetric", axis=0
        )[:, None, None, :]
        f1 *= -(self.nu * cosa / sina)[:, None, None, None]
        f2 = jax.jacfwd(fd2)(f[:, 0, 0], self.p2, h=h, bc="symmetric", axis=0)[
            :, None, None, :
        ]
        f2 *= -self.nu
        df = f1 + f2
        df = jnp.tile(df, (1, self.field.ntheta, self.field.nzeta, 1))

        idx = self.pitchgrid.nxi // 2
        scale = self.nu / h**2
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, idx].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.pitchgrid.nxi, self.pitchgrid.nxi))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class MDKE(lx.AbstractLinearOperator):
    """Monoenergetic Drift Kinetic Equation operator.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
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
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: float
    nu: float
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: bool
    axorder: str = eqx.field(static=True)
    operators: list[lx.AbstractLinearOperator]

    def __init__(
        self,
        field,
        pitchgrid,
        E_psi,
        nu,
        p1="1a",
        p2=2,
        axorder="atz",
        gauge=True,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.nu = jnp.array(nu)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        dtheta = MDKETheta(field, pitchgrid, E_psi, nu, p1, p2, axorder, gauge)
        dzeta = MDKEZeta(field, pitchgrid, E_psi, nu, p1, p2, axorder, gauge)
        dpitch = MDKEPitch(field, pitchgrid, E_psi, nu, p1, p2, axorder, gauge)
        dscatter = MDKEPitchAngleScattering(
            field, pitchgrid, E_psi, nu, p1, p2, axorder, gauge
        )
        self.operators = [dtheta, dzeta, dpitch, dscatter]

    @eqx.filter_jit
    def mv(self, x):
        """Matrix vector product."""
        f0 = self.operators[0].mv(x)
        f1 = self.operators[1].mv(x)
        f2 = self.operators[2].mv(x)
        f3 = self.operators[3].mv(x)
        return f0 + f1 + f2 + f3

    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        d0 = self.operators[0].diagonal()
        d1 = self.operators[1].diagonal()
        d2 = self.operators[2].diagonal()
        d3 = self.operators[3].diagonal()
        return d0 + d1 + d2 + d3

    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        d0 = self.operators[0].block_diagonal()
        d1 = self.operators[1].block_diagonal()
        d2 = self.operators[2].block_diagonal()
        d3 = self.operators[3].block_diagonal()
        return d0 + d1 + d2 + d3

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(MDKE)
@lx.is_diagonal.register(MDKE)
@lx.is_tridiagonal.register(MDKE)
@lx.is_symmetric.register(MDKETheta)
@lx.is_diagonal.register(MDKETheta)
@lx.is_tridiagonal.register(MDKETheta)
@lx.is_symmetric.register(MDKEZeta)
@lx.is_diagonal.register(MDKEZeta)
@lx.is_tridiagonal.register(MDKEZeta)
@lx.is_symmetric.register(MDKEPitch)
@lx.is_diagonal.register(MDKEPitch)
@lx.is_tridiagonal.register(MDKEPitch)
@lx.is_symmetric.register(MDKEPitchAngleScattering)
@lx.is_diagonal.register(MDKEPitchAngleScattering)
@lx.is_tridiagonal.register(MDKEPitchAngleScattering)
def _(operator):
    return False
