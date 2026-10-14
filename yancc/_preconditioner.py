"""Stuff for preconditioners."""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.sharding import Mesh
from jaxtyping import Array, ArrayLike, Float

from ._collisions import RosenbluthPotentials
from ._finite_diff import DEFAULT_P1M, DEFAULT_P2M, fd_coeffs
from ._linalg import (
    AbstractYanccOperator,
    DenseLUInverseOperator,
    LowRankUpdateOperator,
)
from ._misc import DKEConstraint, DKESources
from ._multigrid import (
    MultigridOperator,
    get_dke_operators,
    get_dke_smoothers,
    get_fields_grids,
    get_grid_resolutions,
    get_mdke_operators,
    get_mdke_smoothers,
    get_prolongations,
    get_restrictions,
)
from ._sharding import _validate_mesh
from ._trajectories import DKE, MDKE
from .field import Field
from .species import LocalMaxwellian, _collisionality
from .velocity_grids import UniformPitchAngleGrid, _AbstractSpeedGrid


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
        verbose: bool | int = False,
        **options,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.erhohat = jnp.asarray(erhohat)
        self.nuhat = jnp.asarray(nuhat)
        self.p1 = options.pop("p1", DEFAULT_P1M)
        self.p2 = options.pop("p2", DEFAULT_P2M)
        gauge = options.pop("gauge", True)
        resolutions = options.pop("resolutions", None)
        max_grids = options.pop("max_grids", None)
        coarsening_factor = options.pop("coarsening_factor", None)
        coarse_N = options.pop("coarse_N", 8000)
        # The coarsest grid must still fit the FD stencils: every axis needs
        # n > stencil_width // 2 (periodic/symmetric BCs). A grid too coarse to
        # hold the stencil should error (via the operator asserts), so we floor
        # theta/a at min_n rather than capping at the given size. The exception
        # is an axisymmetric (tokamak) field, nz=1: d/dzeta == 0, no zeta
        # stencil, so min_nz collapses to 1 and zeta is never coarsened.
        min_n = (
            max(fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2) + 1
        )
        min_nt = options.pop("min_nt", min_n)
        min_nz = options.pop("min_nz", 1 if field.nzeta == 1 else min_n)
        min_na = options.pop("min_na", min_n)
        # Smoother FD order, independent of the coarse-operator order (p1/p2).
        smooth_p1 = options.pop("smooth_p1", self.p1)
        smooth_p2 = options.pop("smooth_p2", self.p2)
        smooth_solver = options.pop("smooth_solver", None)
        smooth_weights = options.pop("smooth_weights", None)
        smooth_method = options.pop("smooth_method", "standard")
        smooth_type = options.pop("smooth_type", "plane,a,t,z")
        coarse_method = options.pop("coarse_method", "standard")
        coarse_weight = options.pop("coarse_weight", 1.0)
        interp_method = options.pop("interp_method", "linear")
        v1 = options.pop("v1", 3)
        v2 = options.pop("v2", 3)
        cycle_index = options.pop("cycle_index", 3)
        as_matrix_chunk = options.pop("as_matrix_chunk", 512)

        assert len(options) == 0, "MDKEPreconditioner got unknown option " + str(
            options
        )

        if resolutions is None:
            resolutions = get_grid_resolutions(
                ns=1,
                nx=1,
                na=pitchgrid.nalpha,
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
        smoothers = get_mdke_smoothers(
            fields=fields,
            pitchgrids=grids,
            erhohat=erhohat,
            nuhat=nuhat,
            p1=smooth_p1,
            p2=smooth_p2,
            gauge=gauge,
            smooth_type=smooth_type,
            smooth_solver=smooth_solver,
            weight=smooth_weights,
            # the level operators can be shared when the smoothers use the same ones
            operators=(
                operators if (smooth_p1, smooth_p2) == (self.p1, self.p2) else None
            ),
        )
        # The MDKE has no species mass ratios to badly scale the coarse operator, so
        # its LU factorization is never the ill-conditioned case iterative refinement
        # is for; skip it, along with the condition number estimate that decides it.
        # The dense coarse matrix is built a chunk of columns at a time, which keeps
        # peak memory near the size of the matrix itself rather than that times the
        # number of intermediates in a matrix vector product.
        coarse_opinv = DenseLUInverseOperator(
            operators[0], refine=0, batch_size=as_matrix_chunk
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
            coarse_opinv=coarse_opinv,
            coarse_method=coarse_method,
            coarse_weight=coarse_weight,
            verbose=max(0, verbose - 2),
        )

    def print_resolution_summary(self) -> None:
        """Print one ``Grid i: ...`` line per multigrid level."""
        for i, op in enumerate(self.operators):
            # cast is a no-op at runtime; just narrows the declared
            # AbstractLinearOperator type to MDKE for pyright.
            op = cast(MDKE, op)
            jax.debug.print(
                f"Grid {i}: nα={op.pitchgrid.nalpha:4d}, "
                f"nθ={op.field.ntheta:4d}, "
                f"nζ={op.field.nzeta:4d}, "
                f"N={op.pitchgrid.nalpha * op.field.ntheta * op.field.nzeta:,d}",
                ordered=True,
            )


def _dke_resolutions(field, pitchgrid, speedgrid, species, p1, p2, options):
    """Resolutions of the multigrid levels of a DKEPreconditioner, coarse to fine.

    Pops the options that set them from ``options``.
    """
    resolutions = options.pop("resolutions", None)
    coarsening_factor = options.pop("coarsening_factor", None)
    max_grids = options.pop("max_grids", None)
    coarse_N = options.pop("coarse_N", 8000)
    # The coarsest grid must still fit the FD stencils: every axis needs
    # n > stencil_width // 2 (periodic/symmetric BCs). A grid too coarse to
    # hold the stencil should error (via the operator asserts), so we floor
    # theta/a at min_n rather than capping at the given size. The exception
    # is an axisymmetric (tokamak) field, nz=1: d/dzeta == 0, no zeta
    # stencil, so min_nz collapses to 1 and zeta is never coarsened.
    min_n = max(fd_coeffs[1][p1].size // 2, fd_coeffs[2][p2].size // 2) + 1
    min_nt = options.pop("min_nt", min_n)
    min_nz = options.pop("min_nz", 1 if field.nzeta == 1 else min_n)
    min_na = options.pop("min_na", min_n)
    if resolutions is None:
        resolutions = get_grid_resolutions(
            ns=len(species),
            nx=speedgrid.nx,
            na=pitchgrid.nalpha,
            nt=field.ntheta,
            nz=field.nzeta,
            coarse_N=coarse_N,
            min_na=min_na,
            min_nt=min_nt,
            min_nz=min_nz,
            max_grids=max_grids,
            coarsening_factor=coarsening_factor,
        )
    return resolutions


def _print_grid_levels(resolutions) -> None:
    """Print one ``Grid i: ...`` line per multigrid level, from (ns, nx, na, nt, nz)."""
    for i, (ns, nx, na, nt, nz) in enumerate(resolutions):
        jax.debug.print(
            f"Grid {i}: nx={nx:4d}, "
            f"nα={na:4d}, "
            f"nθ={nt:4d}, "
            f"nζ={nz:4d}, "
            f"N={ns * nx * na * nt * nz:,d}",
            ordered=True,
        )


def _print_dke_resolution_summary(
    field, pitchgrid, speedgrid, species, multigrid_options
) -> None:
    """Print the multigrid levels a DKEPreconditioner would have, without building it.

    The levels only depend on the grids and on the multigrid options that set the
    coarsening, so they can be shown before, or without, building the preconditioner.
    """
    options = dict(multigrid_options)
    p1 = options.pop("p1", DEFAULT_P1M)
    p2 = options.pop("p2", DEFAULT_P2M)
    _print_grid_levels(
        _dke_resolutions(field, pitchgrid, speedgrid, species, p1, p2, options)
    )


class DKEPreconditioner(MultigridOperator):
    """Preconditioner for the DKE.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Speed grid data.
    species : list of LocalMaxwellian
        Plasma species.
    Erho : float
        Radial electric field, Erho = -∂Φ/∂ρ, in Volts (ρ dimensionless).
    background : list of LocalMaxwellian, optional
        Background species for inter-species collisions.
    potentials : RosenbluthPotentials
        Rosenbluth potentials for the collision operator.
    mesh : jax.sharding.Mesh, optional
        Devices to split the preconditioner across, with axes named ``"species"``
        and/or ``"speed"``.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: _AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    background: list[LocalMaxwellian]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: _AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: list[LocalMaxwellian] | None,
        potentials: RosenbluthPotentials,
        verbose: bool | int = False,
        mesh: Mesh | None = None,
        **options,
    ):
        _validate_mesh(mesh, len(species), speedgrid.nx)
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        self.Erho = jnp.asarray(Erho)

        self.p1 = options.pop("p1", DEFAULT_P1M)
        self.p2 = options.pop("p2", DEFAULT_P2M)
        gauge = options.pop("gauge", "shift")
        if isinstance(gauge, str):
            if gauge != "shift":
                raise ValueError(f"gauge must be True, False or 'shift', got {gauge!r}")
        else:
            gauge = bool(gauge)
        resolutions = _dke_resolutions(
            field, pitchgrid, speedgrid, species, self.p1, self.p2, options
        )
        # Smoother FD order, independent of the coarse-operator order (p1/p2).
        smooth_p1 = options.pop("smooth_p1", self.p1)
        smooth_p2 = options.pop("smooth_p2", self.p2)
        smooth_solver = options.pop("smooth_solver", None)
        smooth_weights = options.pop("smooth_weights", None)
        smooth_method = options.pop("smooth_method", "standard")
        smooth_type = options.pop("smooth_type", "plane,s,x,a,l01t,l01z")
        coarse_method = options.pop("coarse_method", "standard")
        coarse_weight = options.pop("coarse_weight", 1.0)
        interp_method = options.pop("interp_method", "linear")
        v1 = options.pop("v1", 3)
        v2 = options.pop("v2", 3)
        cycle_index = options.pop("cycle_index", 1)
        operator_weights = options.pop("operator_weights", jnp.ones(8).at[-1].set(0))
        smoother_weights = options.pop("smoother_weights", operator_weights)
        coulomb_log = options.pop("coulomb_log", None)
        as_matrix_chunk = options.pop("as_matrix_chunk", 512)

        assert len(options) == 0, "DKEPreconditioner got unknown option " + str(options)

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
            gauge=gauge is True,
            operator_weights=operator_weights,
            coulomb_log=coulomb_log,
            mesh=mesh,
        )
        smoothers = get_dke_smoothers(
            fields=fields,
            pitchgrids=grids,
            speedgrid=speedgrid,
            species=species,
            Erho=Erho,
            background=background,
            potentials=potentials,
            p1=smooth_p1,
            p2=smooth_p2,
            gauge=gauge is True,
            smooth_type=smooth_type,
            smooth_solver=smooth_solver,
            weight=smooth_weights,
            operator_weights=smoother_weights,
            coulomb_log=coulomb_log,
            # the level operators can be shared when the smoothers use the same ones
            operators=(
                operators
                if (smooth_p1, smooth_p2) == (self.p1, self.p2)
                and smoother_weights is operator_weights
                else None
            ),
            mesh=mesh,
        )
        # With gauge="shift" the level operators keep the null space of the DKE (a
        # density and an energy mode for each species) rather than replacing equations
        # at a grid point to remove it (gauge=True). Every level then has the same null
        # space, constant in pitch and on the flux surface, which prolongation maps
        # exactly between levels, and the outer bordered solve removes those
        # components from the preconditioner's input and output. Replacing equations
        # at a point instead leaves modes that differ from a null mode only near that
        # point, which are nearly singular when the point is weakly coupled to its
        # neighbors, and their shape depends on the grid, so a coarse correction
        # along them does not match the finer levels. The direct solve on the coarsest
        # grid needs a nonsingular matrix, so it factors the coarse operator with the
        # null space shifted away from zero. With gauge=False the coarse matrix is
        # singular, and its factorization is only useful for verification.
        #
        # It also needs the operator as a dense matrix. Building it a chunk of columns
        # at a time keeps peak memory near the size of the matrix itself, rather than
        # that times the number of intermediates in a matrix vector product, at the
        # cost of a little speed. The matrix is factored after row/column
        # equilibration. With several species its entries span many orders of
        # magnitude, and an unscaled LU has an error floor large enough to leave the
        # nearly singular heavy-species modes with no correct digits, which stalls the
        # outer Krylov solve at a residual that depends on floating point details of
        # the hardware. One step of refinement against the coarse operator, applied
        # when the matrix is badly enough conditioned to need it, removes most of the
        # remaining error.
        coarse_op = operators[0]
        if gauge == "shift":
            coarse_op = _shift_dke_nullspace(coarse_op)
        coarse_opinv = DenseLUInverseOperator(coarse_op, batch_size=as_matrix_chunk)
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
            coarse_weight=coarse_weight,
            verbose=max(0, verbose - 2),
        )

    def print_resolution_summary(self) -> None:
        """Print one ``Grid i: ...`` line per multigrid level."""
        ns = len(self.species)
        nx = self.speedgrid.nx
        # cast is a no-op at runtime; just narrows the declared
        # AbstractLinearOperator type to DKE for pyright.
        ops = [cast(DKE, op) for op in self.operators]
        _print_grid_levels(
            [
                (ns, nx, op.pitchgrid.nalpha, op.field.ntheta, op.field.nzeta)
                for op in ops
            ]
        )


def _shift_dke_nullspace(operator: DKE) -> LowRankUpdateOperator:
    """Ungauged DKE operator with its density and energy null space shifted from zero.

    Returns ``A + L diag(s) R^T``, where the columns of ``R`` span the density and
    energy modes of each species (the null space of ``A``), the columns of ``L`` span
    the density and energy moments (approximately the left null space), and ``s`` is
    the mean magnitude of the diagonal of each species' block of ``A``.
    """
    args = (operator.field, operator.pitchgrid, operator.speedgrid, operator.species)
    # Both have one block of two columns per species, so the orthonormal bases from
    # QR keep the species separate and each species can be shifted by its own scale.
    R = jnp.linalg.qr(DKESources(*args).as_matrix())[0]
    L = jnp.linalg.qr(DKEConstraint(*args, True).as_matrix().T)[0]
    # For a right hand side in the range of A the solution of the shifted system
    # solves A x = b, with no component along R, whatever the shift is, as long as L
    # is not orthogonal to the left null space. So the shift only needs to lift the
    # null modes well above the small nonzero singular values of A without exceeding
    # its largest ones. The smallest singular values of the shifted operator stop
    # changing once the shift is a few orders of magnitude above the nonzero ones,
    # and the mean diagonal of each species is far above those and below the largest.
    ns = len(operator.species)
    scale = jnp.abs(operator.diagonal()).reshape((ns, -1)).mean(axis=1)
    return LowRankUpdateOperator(operator, L, R, jnp.repeat(scale, 2))


class DKEMPreconditioner(AbstractYanccOperator):
    """Preconditioner for the DKE using block diagonal MDKE preconditioners.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
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
    pitchgrid: UniformPitchAngleGrid
    speedgrid: _AbstractSpeedGrid
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
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: _AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: list[LocalMaxwellian] | None = None,
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
                nu = _collisionality(spec, v, *others)
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
                * self.pitchgrid.nalpha
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
            na = op.pitchgrid.nalpha
            nt = op.field.ntheta
            nz = op.field.nzeta
            jax.debug.print(
                f"Grid {i}: nx={nx:4d}, "
                f"nα={na:4d}, "
                f"nθ={nt:4d}, "
                f"nζ={nz:4d}, "
                f"N={ns * nx * na * nt * nz:,d}",
                ordered=True,
            )
