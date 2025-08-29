"""Stuff for preconditioners."""

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from .collisions import RosenbluthPotentials
from .field import Field
from .multigrid import (
    MultigridOperator,
    get_dke_jacobi_smoothers,
    get_dke_operators,
    get_fields_grids,
    get_mdke_jacobi_smoothers,
    get_mdke_operators,
)
from .species import LocalMaxwellian
from .velocity_grids import AbstractSpeedGrid, UniformPitchAngleGrid


class MDKEPreconditioner(MultigridOperator):
    """Preconditioner for the MDKE.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized radial electric field.
    nu : float
        Normalized collisionality.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: Float[Array, ""]
    nu: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        E_psi: Float[ArrayLike, ""],
        nu: Float[ArrayLike, ""],
        **options
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.asarray(E_psi)
        self.nu = jnp.asarray(nu)
        self.p1 = options.pop("p1m", "2d")
        self.p2 = options.pop("p2m", 2)
        gauge = options.pop("gauge", True)
        smooth_solver = options.pop("smooth_solver", "dense")
        smooth_weights = options.pop("smooth_weights", None)
        coarsening = options.pop("coarsening", 2)
        coarse_N = options.pop("coarse_N", 8000)
        min_nt = options.pop("min_nt", 5)
        min_nz = options.pop("min_nz", 5)
        min_na = options.pop("min_na", 5)
        coarse_overweight = options.pop("coarse_overweight", 1)
        interp_method = options.pop("interp_method", "linear")
        smooth_method = options.pop("smooth_method", "standard")
        verbose = options.pop("verbose", False)
        v1 = options.pop("v1", 3)
        v2 = options.pop("v2", 3)
        cycle_index = options.pop("cycle_index", 3)

        assert len(options) == 0, "got unknown option " + str(options)

        fields, grids = get_fields_grids(
            field=field,
            nt=field.ntheta,
            nz=field.nzeta,
            na=pitchgrid.nxi,
            coarsening_factor=coarsening,
            min_N=coarse_N,
            min_nt=min_nt,
            min_nz=min_nz,
            min_na=min_na,
        )

        operators = get_mdke_operators(
            fields=fields,
            pitchgrids=grids,
            E_psi=E_psi,
            nu=nu,
            p1=self.p1,
            p2=self.p2,
            gauge=gauge,
        )
        smoothers = get_mdke_jacobi_smoothers(
            fields=fields,
            pitchgrids=grids,
            E_psi=E_psi,
            nu=nu,
            p1=self.p1,
            p2=self.p2,
            gauge=gauge,
            smooth_solver=smooth_solver,
            weight=smooth_weights,
        )

        super().__init__(
            operators=operators[::-1],
            smoothers=smoothers[::-1],
            x0=None,
            cycle_index=cycle_index,
            v1=v1,
            v2=v2,
            interp_method=interp_method,
            smooth_method=smooth_method,
            coarse_opinv=None,
            coarse_overweight=coarse_overweight,
            verbose=verbose,
        )


@lx.is_symmetric.register(MDKEPreconditioner)
@lx.is_diagonal.register(MDKEPreconditioner)
@lx.is_tridiagonal.register(MDKEPreconditioner)
def _(operator):
    return False


class DKEPreconditioner(MultigridOperator):
    """Preconditioner for the DKE using block diagonal MDKE preconditioners."""

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        potentials: RosenbluthPotentials,
        **options
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.E_psi = jnp.asarray(E_psi)

        self.p1 = options.pop("p1", "2d")
        self.p2 = options.pop("p2", 2)
        gauge = options.pop("gauge", True)
        smooth_solver = options.pop("smooth_solver", "dense")
        smooth_weights = options.pop("smooth_weights", None)
        coarsening = options.pop("coarsening", 2)
        coarse_N = options.pop("coarse_N", 8000)
        min_nt = options.pop("min_nt", 5)
        min_nz = options.pop("min_nz", 5)
        min_na = options.pop("min_na", 5)
        coarse_overweight = options.pop("coarse_overweight", 1)
        interp_method = options.pop("interp_method", "linear")
        smooth_method = options.pop("smooth_method", "standard")
        verbose = options.pop("verbose", False)
        v1 = options.pop("v1", 3)
        v2 = options.pop("v2", 3)
        cycle_index = options.pop("cycle_index", 3)

        assert len(options) == 0, "got unknown option " + str(options)

        fields, grids = get_fields_grids(
            field=field,
            nt=field.ntheta,
            nz=field.nzeta,
            na=pitchgrid.nxi,
            coarsening_factor=coarsening,
            min_N=coarse_N,
            min_nt=min_nt,
            min_nz=min_nz,
            min_na=min_na,
            nx=speedgrid.nx,
            ns=len(species),
        )

        operators = get_dke_operators(
            fields=fields,
            pitchgrids=grids,
            speedgrid=speedgrid,
            species=species,
            E_psi=E_psi,
            potentials=potentials,
            p1=self.p1,
            p2=self.p2,
            gauge=gauge,
        )
        smoothers = get_dke_jacobi_smoothers(
            fields=fields,
            pitchgrids=grids,
            speedgrid=speedgrid,
            species=species,
            E_psi=E_psi,
            potentials=potentials,
            p1=self.p1,
            p2=self.p2,
            gauge=gauge,
            smooth_solver=smooth_solver,
            weight=smooth_weights,
        )

        super().__init__(
            operators=operators[::-1],
            smoothers=smoothers[::-1],
            x0=None,
            cycle_index=cycle_index,
            v1=v1,
            v2=v2,
            interp_method=interp_method,
            smooth_method=smooth_method,
            coarse_opinv=None,
            coarse_overweight=coarse_overweight,
            verbose=verbose,
        )


@lx.is_symmetric.register(DKEPreconditioner)
@lx.is_diagonal.register(DKEPreconditioner)
@lx.is_tridiagonal.register(DKEPreconditioner)
def _(operator):
    return False
