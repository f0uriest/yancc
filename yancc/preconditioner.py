"""Stuff for preconditioners."""

from typing import Optional, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from .collisions import RosenbluthPotentials
from .field import Field
from .linalg import InverseLinearOperator
from .multigrid import (
    MultigridOperator,
    get_dke_jacobi2_smoothers,
    get_dke_jacobi_smoothers,
    get_dke_operators,
    get_fields_grids,
    get_grid_resolutions,
    get_mdke_jacobi_smoothers,
    get_mdke_operators,
    get_prolongations,
    get_restrictions,
)
from .species import LocalMaxwellian, collisionality
from .trajectories import DKE, MDKE
from .velocity_grids import AbstractPitchAngleGrid, AbstractSpeedGrid


class MDKEPreconditioner(MultigridOperator):
    """Preconditioner for the MDKE.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : AbstractPitchAngleGrid
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
    pitchgrid: AbstractPitchAngleGrid
    erhohat: Float[Array, ""]
    nuhat: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
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
        resolutions = options.pop("resolutions", None)
        max_grids = options.pop("max_grids", None)
        coarsening_factor = options.pop("coarsening_factor", None)
        coarse_N = options.pop("coarse_N", 8000)
        min_nt = options.pop("min_nt", 5)
        min_nz = options.pop("min_nz", 5)
        min_na = options.pop("min_na", 5)
        smooth_solver = options.pop("smooth_solver", None)
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

        if resolutions is None:
            resolutions = get_grid_resolutions(
                ns=1,
                nx=1,
                na=pitchgrid.na,
                nt=field.ntheta,
                nz=field.nzeta,
                coarse_N=coarse_N,
                min_na=min_na,
                min_nt=min_nt,
                min_nz=min_nz,
                max_grids=max_grids,
                coarsening_factor=coarsening_factor,
            )

        fields, grids = get_fields_grids(
            field=field,
            pitchgrid=pitchgrid,
            resolutions=resolutions,
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
        prolongations = get_prolongations(
            fields=fields, pitchgrids=grids, prefix_size=1, method=interp_method
        )
        restrictions = get_restrictions(
            fields=fields, pitchgrids=grids, prefix_size=1, method=interp_method
        )

        super().__init__(
            operators=operators,
            smoothers=smoothers,
            prolongations=prolongations,
            restrictions=restrictions,
            x0=None,
            cycle_index=cycle_index,
            v1=v1,
            v2=v2,
            smooth_method=smooth_method,
            coarse_opinv=None,
            coarse_method=coarse_method,
            verbose=max(0, verbose - 2),
        )

    def print_resolution_summary(self) -> None:
        """Print one ``Grid i: ...`` line per multigrid level."""
        for i, op in enumerate(self.operators):
            # cast is a no-op at runtime; just narrows the declared
            # AbstractLinearOperator type to MDKE for pyright.
            op = cast(MDKE, op)
            jax.debug.print(
                f"Grid {i}: na={op.pitchgrid.na:4d}, "
                f"nt={op.field.ntheta:4d}, "
                f"nz={op.field.nzeta:4d}, "
                f"N={op.pitchgrid.na * op.field.ntheta * op.field.nzeta:,d}",
                ordered=True,
            )


@lx.is_symmetric.register(MDKEPreconditioner)
@lx.is_diagonal.register(MDKEPreconditioner)
@lx.is_tridiagonal.register(MDKEPreconditioner)
def _(operator):
    return False


class DKEPreconditioner(MultigridOperator):
    """Preconditioner for the DKE.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Speed grid data.
    species : list of LocalMaxwellian
        Plasma species.
    Erho : float
        Radial electric field, Erho = -∂Φ/∂ρ, in Volts (ρ dimensionless).
    background : list of LocalMaxwellian, optional
        Background species for inter-species collisions.
    """

    field: Field
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    background: list[LocalMaxwellian]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: Optional[list[LocalMaxwellian]],
        potentials: RosenbluthPotentials,
        verbose: Union[bool, int] = False,
        **options,
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        self.Erho = jnp.asarray(Erho)

        self.p1 = options.pop("p1", "2d")
        self.p2 = options.pop("p2", 2)
        gauge = options.pop("gauge", True)
        resolutions = options.pop("resolutions", None)
        coarsening_factor = options.pop("coarsening_factor", None)
        max_grids = options.pop("max_grids", None)
        coarse_N = options.pop("coarse_N", 8000)
        min_nt = options.pop("min_nt", 5)
        min_nz = options.pop("min_nz", 5)
        min_na = options.pop("min_na", 5)
        smooth_solver = options.pop("smooth_solver", None)
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

        if resolutions is None:
            resolutions = get_grid_resolutions(
                ns=len(species),
                nx=speedgrid.nx,
                na=pitchgrid.na,
                nt=field.ntheta,
                nz=field.nzeta,
                coarse_N=coarse_N,
                min_na=min_na,
                min_nt=min_nt,
                min_nz=min_nz,
                max_grids=max_grids,
                coarsening_factor=coarsening_factor,
            )

        fields, grids = get_fields_grids(
            field=field, pitchgrid=pitchgrid, resolutions=resolutions
        )
        operators = get_dke_operators(
            fields=fields,
            pitchgrids=grids,
            speedgrid=speedgrid,
            species=species,
            Erho=Erho,
            background=background,
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
                background=background,
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
                background=background,
                potentials=potentials,
                p1=self.p1,
                p2=self.p2,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=smooth_weights,
                operator_weights=smoother_weights,
                **options,
            )
        coarse_opinv = InverseLinearOperator(operators[0], lx.LU(), throw=False)
        prefix_size = len(species) * speedgrid.nx
        prolongations = get_prolongations(
            fields=fields,
            pitchgrids=grids,
            prefix_size=prefix_size,
            method=interp_method,
        )
        restrictions = get_restrictions(
            fields=fields,
            pitchgrids=grids,
            prefix_size=prefix_size,
            method=interp_method,
        )

        super().__init__(
            operators=operators,
            smoothers=smoothers,
            prolongations=prolongations,
            restrictions=restrictions,
            x0=None,
            cycle_index=cycle_index,
            v1=v1,
            v2=v2,
            smooth_method=smooth_method,
            coarse_opinv=coarse_opinv,
            coarse_method=coarse_method,
            verbose=max(0, verbose - 2),
        )

    def print_resolution_summary(self) -> None:
        """Print one ``Grid i: ...`` line per multigrid level."""
        ns = len(self.species)
        nx = self.speedgrid.nx
        for i, op in enumerate(self.operators):
            # cast is a no-op at runtime; just narrows the declared
            # AbstractLinearOperator type to DKE for pyright.
            op = cast(DKE, op)
            na = op.pitchgrid.na
            nt = op.field.ntheta
            nz = op.field.nzeta
            jax.debug.print(
                f"Grid {i}: nx={nx:4d}, "
                f"na={na:4d}, "
                f"nt={nt:4d}, "
                f"nz={nz:4d}, "
                f"N={ns * nx * na * nt * nz:,d}",
                ordered=True,
            )


@lx.is_symmetric.register(DKEPreconditioner)
@lx.is_diagonal.register(DKEPreconditioner)
@lx.is_tridiagonal.register(DKEPreconditioner)
def _(operator):
    return False


class DKEMPreconditioner(lx.AbstractLinearOperator):
    """Preconditioner for the DKE using block diagonal MDKE preconditioners.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Speed grid data.
    species : list of LocalMaxwellian
        Plasma species.
    Erho : float
        Radial electric field, Erho = -∂Φ/∂ρ, in Volts (ρ dimensionless).
    background : list of LocalMaxwellian, optional
        Background species for inter-species collisions.
    """

    field: Field
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    background: list[LocalMaxwellian]
    M: MultigridOperator
    vs: jax.Array
    smooth_method: str = eqx.field(static=True)
    coarse_method: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: Optional[list[LocalMaxwellian]] = None,
        **options,
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        self.Erho = jnp.asarray(Erho)
        self.smooth_method = options.get("smooth_method", "standard")
        self.coarse_method = options.get("coarse_method", "standard")

        erhohats = []
        nuhats = []
        vs = []
        for i, spec in enumerate(species):
            temp_nuhat = []
            temp_erhohat = []
            temp_vs = []
            others = species[:i] + species[i + 1 :] + background
            for x in speedgrid.x:
                v = x * spec.v_thermal
                nu = collisionality(spec, v, *others)
                erhohat = Erho / v
                nuhat = nu / v
                temp_erhohat.append(erhohat)
                temp_nuhat.append(nuhat)
                temp_vs.append(v)

            erhohats.append(temp_erhohat)
            nuhats.append(temp_nuhat)
            vs.append(temp_vs)

        erhohats = jnp.array(erhohats)
        nuhats = jnp.array(nuhats)
        self.vs = jnp.array(vs)

        def get_mdke_precond(nuhat, erhohat):
            return MDKEPreconditioner(
                field=field,
                pitchgrid=pitchgrid,
                erhohat=erhohat,
                nuhat=nuhat,
                **options,
            )

        self.M = jax.vmap(jax.vmap(get_mdke_precond))(nuhats, erhohats)

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
                * self.pitchgrid.na
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
                * self.pitchgrid.na
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator.

        ``mv`` is ``(M @ v) / vs`` (per ``(species, x)`` block), so its adjoint
        is ``M^T @ (u / vs)``. Closed-form so we don't reverse-mode through the
        underlying multigrid ``while_loop``.
        """
        ns = len(self.species)
        nx = self.speedgrid.nx
        vs = self.vs
        M = self.M

        def _mv(u):
            u = u.reshape((ns, nx, -1)) / vs[:, :, None]
            out = jax.vmap(jax.vmap(lambda Mi, v: Mi.transpose().mv(v)))(M, u)
            return out.flatten()

        return lx.FunctionLinearOperator(_mv, jnp.zeros(self.in_size()))

    def print_resolution_summary(self) -> None:
        """Print one ``Grid i: ...`` line per multigrid level. The same grid
        stack is shared across all (species, x) pairs; only the underlying
        ``nuhat`` / ``erhohat`` coefficients vary.
        """
        ns = len(self.species)
        nx = self.speedgrid.nx
        for i, op in enumerate(self.M.operators):
            # cast is a no-op at runtime; just narrows the declared
            # AbstractLinearOperator type to MDKE for pyright.
            op = cast(MDKE, op)
            na = op.pitchgrid.na
            nt = op.field.ntheta
            nz = op.field.nzeta
            jax.debug.print(
                f"Grid {i}: nx={nx:4d}, "
                f"na={na:4d}, "
                f"nt={nt:4d}, "
                f"nz={nz:4d}, "
                f"N={ns * nx * na * nt * nz:,d}",
                ordered=True,
            )


@lx.is_symmetric.register(DKEMPreconditioner)
@lx.is_diagonal.register(DKEMPreconditioner)
@lx.is_tridiagonal.register(DKEMPreconditioner)
def _(operator):
    return False
