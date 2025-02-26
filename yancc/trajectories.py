"""Drift Kinetic Operators without collisions."""

import cola
import jax.numpy as jnp
from monkes import Field, LocalMaxwellian

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
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi

        rdot1 = FullTrajectoriesSurface1(field, speedgrid, pitchgrid, species, E_psi)
        rdot2 = FullTrajectoriesSurface2(field, speedgrid, pitchgrid, species, E_psi)
        xidot1 = FullTrajectoriesPitch1(field, speedgrid, pitchgrid, species, E_psi)
        xidot2 = FullTrajectoriesPitch2(field, speedgrid, pitchgrid, species, E_psi)
        xdot = FullTrajectoriesSpeed(field, speedgrid, pitchgrid, species, E_psi)
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
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi

        rdot1 = DKESTrajectoriesSurface1(field, pitchgrid, species, v, E_psi)
        rdot2 = DKESTrajectoriesSurface2(field, pitchgrid, species, v, E_psi)
        xidot = DKESTrajectoriesPitch(field, pitchgrid, species, v, E_psi)
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
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

        Is = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
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
            Ms = (cola.ops.Kronecker(E_psi * Is, xDx), pxi2, BxgradpsidotgradB_over_2B3)
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
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

        mxi2 = cola.ops.Diagonal(1 - pitchgrid.xi**2)
        Dxi = cola.ops.Dense(pitchgrid.Dxi_pseudospectral)
        xa = cola.ops.Diagonal(speedgrid.x) @ cola.ops.Dense(speedgrid.xvander)
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
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

        Is = cola.ops.Identity((len(species), len(species)), speedgrid.x.dtype)
        Ix = cola.ops.Dense(speedgrid.xvander)
        E_I = E_psi * cola.ops.Kronecker(Is, Ix)
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
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

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
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

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
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

        mxi2 = cola.ops.Diagonal(1 - pitchgrid.xi**2)
        Dxi = cola.ops.Dense(pitchgrid.Dxi_pseudospectral)
        if approx_rdot:
            A = field.bdotgradB / (2 * field.Bmag)
            Ak = approx_kron_diag2d(A.flatten(), *A.shape)
            Ms = (-v * mxi2 @ Dxi, *Ak.Ms)
        else:
            bdotgradB_over_2B = cola.ops.Diagonal(
                (field.bdotgradB / (2 * field.Bmag)).flatten()
            )
            Ms = (-v * mxi2 @ Dxi, bdotgradB_over_2B)
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
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

        xi = v * cola.ops.Diagonal(pitchgrid.xi)
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

            Ms = (xi, rdot1_tz)
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
    """

    def __init__(
        self,
        field: Field,
        pitchgrid: PitchAngleGrid,
        species: LocalMaxwellian,
        v: float,
        E_psi: float,
        approx_rdot: bool = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.species = species
        self.v = v
        self.E_psi = E_psi
        self.approx_rdot = approx_rdot

        Exi = E_psi * cola.ops.Identity(
            (pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype
        )
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        Dt = cola.ops.Dense(field.Dt)
        Dz = cola.ops.Dense(field.Dz)
        if approx_rdot:
            A1 = field.B_sub_t / (field.Bmag_fsa**2 * field.sqrtg)
            A2 = field.B_sub_z / (field.Bmag_fsa**2 * field.sqrtg)
            Ak1 = approx_kron_diag2d(A1.flatten(), *A1.shape)
            Ak2 = approx_kron_diag2d(A2.flatten(), *A2.shape)
            B1 = prodkron2kronprod(Ak1 @ cola.ops.Kronecker(Dt, Iz))
            B2 = prodkron2kronprod(Ak2 @ cola.ops.Kronecker(It, Dz))
            C = approx_sum_kron((B1, B2))
            Ms = (Exi, *C.Ms)
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

            Ms = (Exi, rdot2_tz)
        super().__init__(*Ms)
