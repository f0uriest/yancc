"""Stuff for preconditioners."""

from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from .collisions import RosenbluthPotentials
from .field import Field
from .multigrid import (
    MultigridOperator,
    get_dke_jacobi2_smoothers,
    get_dke_jacobi_smoothers,
    get_dke_operators,
    get_fields_grids,
    get_grid_resolutions,
    get_mdke_jacobi_smoothers,
    get_mdke_operators,
)
from .species import LocalMaxwellian, collisionality
from .velocity_grids import AbstractSpeedGrid, UniformPitchAngleGrid


class MDKEPreconditioner(MultigridOperator):
    """Preconditioner for the MDKE.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v in units of V*s/m
    nuhat : float
        Monoenergetic collisionality, nu/v in units of 1/m
    verbose : int
        Level of verbosity:
          - 0: no into printed.
          - 1: print initialization info.
          - 2: also print residuals at each multigrid level before and after smoothing.
          - 3: also print residuals within smoothing iterations.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    erhohat: Float[Array, ""]
    nuhat: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        nuhat: Float[ArrayLike, ""],
        verbose: Union[bool, int] = False,
        **options,
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.erhohat = jnp.asarray(erhohat)
        self.nuhat = jnp.asarray(nuhat)
        self.p1 = options.pop("p1", "2d")
        self.p2 = options.pop("p2", 2)
        gauge = options.pop("gauge", True)
        ress = options.pop("ress", None)
        max_grids = options.pop("max_grids", None)
        coarsening_factor = options.pop("coarsening_factor", None)
        coarse_N = options.pop("coarse_N", 8000)
        min_nt = options.pop("min_nt", 5)
        min_nz = options.pop("min_nz", 5)
        min_na = options.pop("min_na", 5)
        smooth_solver = options.pop("smooth_solver", "dense")
        smooth_weights = options.pop("smooth_weights", None)
        smooth_method = options.pop("smooth_method", "standard")
        coarse_method = options.pop("coarse_method", "standard")
        interp_method = options.pop("interp_method", "linear")
        v1 = options.pop("v1", 3)
        v2 = options.pop("v2", 3)
        cycle_index = options.pop("cycle_index", 3)

        assert len(options) == 0, "MDKEPreconditioner got unknown option " + str(
            options
        )

        if ress is None:
            ress = get_grid_resolutions(
                ns=1,
                nx=1,
                na=pitchgrid.nxi,
                nt=field.ntheta,
                nz=field.nzeta,
                coarse_N=coarse_N,
                min_na=min_na,
                min_nt=min_nt,
                min_nz=min_nz,
                max_grids=max_grids,
                coarsening_factor=coarsening_factor,
            )

        if verbose:
            for i, res in enumerate(ress):
                ns, nx, na, nt, nz = res
                # these values aren't traced so we can use regular print
                print(
                    f"Grid {i}: na={na:4d}, "
                    f"nt={nt:4d}, "
                    f"nz={nz:4d}, "
                    f"N={ns*nx*na*nt*nz}"
                )

        fields, grids = get_fields_grids(
            field=field,
            pitchgrid=pitchgrid,
            ress=ress,
        )

        operators = get_mdke_operators(
            fields=fields,
            pitchgrids=grids,
            erhohat=erhohat,
            nuhat=nuhat,
            p1=self.p1,
            p2=self.p2,
            gauge=gauge,
        )
        smoothers = get_mdke_jacobi_smoothers(
            fields=fields,
            pitchgrids=grids,
            erhohat=erhohat,
            nuhat=nuhat,
            p1=self.p1,
            p2=self.p2,
            gauge=gauge,
            smooth_solver=smooth_solver,
            weight=smooth_weights,
        )

        super().__init__(
            operators=operators,
            smoothers=smoothers,
            x0=None,
            cycle_index=cycle_index,
            v1=v1,
            v2=v2,
            interp_method=interp_method,
            smooth_method=smooth_method,
            coarse_opinv=None,
            coarse_method=coarse_method,
            verbose=max(0, verbose - 2),
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
    Erho: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        potentials: RosenbluthPotentials,
        verbose: Union[bool, int] = False,
        **options,
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.Erho = jnp.asarray(Erho)

        self.p1 = options.pop("p1", "2d")
        self.p2 = options.pop("p2", 2)
        gauge = options.pop("gauge", True)
        ress = options.pop("ress", None)
        coarsening_factor = options.pop("coarsening_factor", None)
        max_grids = options.pop("max_grids", None)
        coarse_N = options.pop("coarse_N", 8000)
        min_nt = options.pop("min_nt", 5)
        min_nz = options.pop("min_nz", 5)
        min_na = options.pop("min_na", 5)
        smooth_solver = options.pop("smooth_solver", "dense")
        smooth_weights = options.pop("smooth_weights", None)
        smooth_method = options.pop("smooth_method", "standard")
        smooth_type = options.pop("smooth_type", 1)
        coarse_method = options.pop("coarse_method", "standard")
        interp_method = options.pop("interp_method", "linear")
        v1 = options.pop("v1", 3)
        v2 = options.pop("v2", 3)
        cycle_index = options.pop("cycle_index", 1)
        operator_weights = options.pop("operator_weights", jnp.ones(8).at[-1].set(0))
        smoother_weights = options.pop("smoother_weights", operator_weights)

        assert len(options) == 0, "DKEPreconditioner got unknown option " + str(options)

        if ress is None:
            ress = get_grid_resolutions(
                ns=len(species),
                nx=speedgrid.nx,
                na=pitchgrid.nxi,
                nt=field.ntheta,
                nz=field.nzeta,
                coarse_N=coarse_N,
                min_na=min_na,
                min_nt=min_nt,
                min_nz=min_nz,
                max_grids=max_grids,
                coarsening_factor=coarsening_factor,
            )

        fields, grids = get_fields_grids(field=field, pitchgrid=pitchgrid, ress=ress)

        operators = get_dke_operators(
            fields=fields,
            pitchgrids=grids,
            speedgrid=speedgrid,
            species=species,
            Erho=Erho,
            potentials=potentials,
            p1=self.p1,
            p2=self.p2,
            gauge=gauge,
            operator_weights=operator_weights,
            **options,
        )
        if smooth_type == 1:
            smoothers = get_dke_jacobi_smoothers(
                fields=fields,
                pitchgrids=grids,
                speedgrid=speedgrid,
                species=species,
                Erho=Erho,
                potentials=potentials,
                p1=self.p1,
                p2=self.p2,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=smooth_weights,
                operator_weights=smoother_weights,
                **options,
            )
        else:
            smoothers = get_dke_jacobi2_smoothers(
                fields=fields,
                pitchgrids=grids,
                speedgrid=speedgrid,
                species=species,
                Erho=Erho,
                potentials=potentials,
                p1=self.p1,
                p2=self.p2,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=smooth_weights,
                operator_weights=smoother_weights,
                **options,
            )
        super().__init__(
            operators=operators,
            smoothers=smoothers,
            x0=None,
            cycle_index=cycle_index,
            v1=v1,
            v2=v2,
            interp_method=interp_method,
            smooth_method=smooth_method,
            coarse_opinv=None,
            coarse_method=coarse_method,
            verbose=max(0, verbose - 2),
        )


@lx.is_symmetric.register(DKEPreconditioner)
@lx.is_diagonal.register(DKEPreconditioner)
@lx.is_tridiagonal.register(DKEPreconditioner)
def _(operator):
    return False


class DKEMPreconditioner(lx.AbstractLinearOperator):
    """Preconditioner for the DKE using block diagonal MDKE preconditioners."""

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    M: MultigridOperator
    vs: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        **options,
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.Erho = jnp.asarray(Erho)

        Ers = []
        nus = []
        vs = []
        for spec in species:
            temp_nu = []
            temp_Er = []
            temp_vs = []
            for x in speedgrid.x:
                v = x * spec.v_thermal
                nu = collisionality(spec, v, *species)
                Erhat = Erho / v
                nuhat = nu / v
                temp_Er.append(Erhat)
                temp_nu.append(nuhat)
                temp_vs.append(v)

            Ers.append(temp_Er)
            nus.append(temp_nu)
            vs.append(temp_vs)

        Ers = jnp.array(Ers)
        nus = jnp.array(nus)
        self.vs = jnp.array(vs)

        def get_mdke_precond(nu, Er):
            return MDKEPreconditioner(
                field=field, pitchgrid=pitchgrid, erhohat=Er, nuhat=nu, **options
            )

        self.M = jax.vmap(jax.vmap(get_mdke_precond))(nus, Ers)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix-vector product."""
        vector = vector.reshape((len(self.species), self.speedgrid.nx, -1))

        def _mv(M, v):
            return M.mv(v)

        out = jax.vmap(jax.vmap(_mv))(self.M, vector)
        out = out / self.vs[:, :, None]
        return out.flatten()

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(DKEMPreconditioner)
@lx.is_diagonal.register(DKEMPreconditioner)
@lx.is_tridiagonal.register(DKEMPreconditioner)
def _(operator):
    return False
