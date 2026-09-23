"""Main interface for solving drift kinetic equations in yancc."""

import copy
from collections.abc import Sequence
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from scipy.constants import elementary_charge, proton_mass

from .collisions import RosenbluthPotentials
from .field import FIELD_SOURCE_NAMES, Field
from .finite_diff import DEFAULT_P1A, DEFAULT_P2A
from .krylov import gcrotmk
from .linalg import BorderedOperator, InverseBorderedOperator
from .misc import (
    DKEConstraint,
    DKESources,
    _dke_thermodynamic_forces,
    dke_rhs,
    mdke_rhs,
)
from .preconditioner import (
    DKEPreconditioner,
    MDKEPreconditioner,
    _print_dke_resolution_summary,
)
from .root import deflated_root_scalar
from .solution import DKESolution, MDKESolution
from .species import Estar, LocalMaxwellian, nustar, poloidal_mach
from .trajectories import DKE, MDKE
from .velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def _freeze_preconditioner(M):
    """Detach a preconditioner from autodiff.

    Krylov solvers use lax.custom_linear_solve so the preconditioner isn't in the AD
    path but its construction happens in the traced region before solve so JAX computes
    tangents/cotangents through those factorizations and then discards them at the
    krylov boundary. stop_gradient on the array leaves makes that explicit, freeing AD
    from building the (unused) factorization derivatives.
    """
    arrays, static = eqx.partition(M, eqx.is_inexact_array)
    arrays = jax.lax.stop_gradient(arrays)
    return eqx.combine(arrays, static)


def _preconditioner_is_linear(M) -> bool:
    """True if a multigrid preconditioner is a linear operator.

    Standard pre/post-smoothing and standard coarse-grid correction are linear.
    Krylov-projected smoothers (krylov*/krylov*s) and the residual-driven
    adaptive smoother are not. Returns False if we can't tell, in which case
    the caller should assume nonlinearity (i.e. use flexible GMRES).
    """
    smooth = getattr(M, "smooth_method", None)
    coarse = getattr(M, "coarse_method", None)
    return smooth == "standard" and coarse == "standard"


def _build_dke_preconditioner(
    field,
    pitchgrid,
    speedgrid,
    species,
    background,
    Erho,
    potentials,
    operator_weights,
    coulomb_log,
    verbose,
    multigrid_options,
):
    """Build the default multigrid preconditioner for the DKE at a given Erho."""
    multigrid_options = copy.copy(multigrid_options)
    multigrid_options.setdefault("operator_weights", operator_weights)
    multigrid_options.setdefault("field", field)
    multigrid_options.setdefault("pitchgrid", pitchgrid)
    multigrid_options.setdefault("speedgrid", speedgrid)
    multigrid_options.setdefault("species", species)
    multigrid_options.setdefault("background", background)
    multigrid_options.setdefault("Erho", Erho)
    multigrid_options.setdefault("potentials", potentials)
    multigrid_options.setdefault("gauge", True)
    multigrid_options.setdefault("verbose", verbose)
    multigrid_options.setdefault("coulomb_log", coulomb_log)
    return DKEPreconditioner(**multigrid_options)


def _make_dke_solution(
    f1, rhs, field, pitchgrid, speedgrid, species, Erho, EparB, background
):
    """Wrap a Krylov solution vector as a DKESolution."""
    F0 = jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    F0 = F0[:, :, None, None, None]
    return DKESolution(
        F0=F0,
        f1=f1,
        rhs=rhs,
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        Erho=Erho,
        EparB=EparB,
        background=background,
    )


def solve_mdke(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    erhohat: float | Float[Any, ""],
    nuhat: float | Float[Any, ""],
    verbose: bool | int = False,
    multigrid_options: dict | None = None,
    throw: bool = False,
    **options,
) -> tuple[MDKESolution, dict[str, jax.Array]]:
    """Solve the mono-energetic drift kinetic equation, giving 3x3 transport matrix.

    Parameters
    ----------
    field : Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v = -∂Φ /∂ρ /v in units of V*s/m.
    nuhat : float
        Monoenergetic collisionality, ν/v in units of 1/m.
    verbose: bool, int
        Level of verbosity:

          - 0: no info printed.
          - 1: print initialization info.
          - 2: print info from krylov solver at each iteration. Frequency can be
            controlled by also passing `print_every=<int>`
          - 3: also print residuals at each multigrid level before and after smoothing.
          - 4: also print residuals within smoothing iterations.

        Note that verbose > 2 may slow things down as additional diagnostic info is
        calculated at each step.
    multigrid_options : dict, optional
        Optional parameters to control behavior of multigrid preconditioner.
    throw : bool, optional
        If True, raise a runtime error if the Krylov solver fails to converge
        (forward solve or tangent solve). Default False.

    Returns
    -------
    sol : MDKESolution
        Solution object containing distribution function and drive terms and methods
        for computing moments.
    info : dict
        Info about the solve, such as number of iterations, number of matrix-vector
        products, final residual etc.

    """
    # create a copy so we don't modify user input for repeated calls
    multigrid_options = (
        {} if multigrid_options is None else copy.copy(multigrid_options)
    )

    p1 = options.pop("p1", DEFAULT_P1A)
    p2 = options.pop("p2", DEFAULT_P2A)
    rtol = jnp.asarray(options.pop("rtol", 1e-5))
    atol = jnp.asarray(options.pop("atol", 0.0))
    m = options.pop("m", 150)
    k = options.pop("k", 10)
    maxiter = options.pop("maxiter", 10)
    print_every = options.pop("print_every", 10)
    U1 = options.pop("U1", None)
    U2 = options.pop("U2", None)
    f1 = options.pop("f1", None)
    f2 = options.pop("f2", None)

    assert len(options) == 0, "solve_mdke got unknown option " + str(options)

    nuhat = jnp.asarray(nuhat)
    erhohat = jnp.asarray(erhohat)

    if verbose:
        _print_field_summary(field)
        jax.debug.print("ν` = {nuhat: .3e}", nuhat=nuhat)
        jax.debug.print("E` = {erhohat: .3e}", erhohat=erhohat)

    M = MDKEPreconditioner(
        field=field,
        pitchgrid=pitchgrid,
        nuhat=nuhat,
        erhohat=erhohat,
        verbose=verbose,
        **multigrid_options,
    )
    M = _freeze_preconditioner(M)
    if verbose:
        M.print_resolution_summary()
    flexible = not _preconditioner_is_linear(M)
    A = MDKE(
        field,
        pitchgrid,
        erhohat,
        nuhat,
        p1=p1,
        p2=p2,
        gauge=True,
    )
    rhs = mdke_rhs(field, pitchgrid)

    if f1 is None:
        f1 = jnp.zeros_like(rhs[:, 0])
    f1, j1, nmv1, res1, success1, C1, U1 = gcrotmk(
        A,
        rhs[:, 0],
        x0=f1,
        MR=M,
        m=m,
        k=k,
        rtol=jnp.asarray(rtol),
        atol=jnp.asarray(atol),
        maxiter=jnp.asarray(maxiter),
        verbose=verbose > 1,
        print_every_inner=jnp.asarray(print_every),
        U=U1,
        flexible=flexible,
        throw=throw,
    )
    if f2 is None:
        f2 = jnp.zeros_like(rhs[:, 0])
    f2, j2, nmv2, res2, success2, C2, U2 = gcrotmk(
        A,
        rhs[:, 2],
        x0=f2,
        MR=M,
        m=m,
        k=k,
        rtol=jnp.asarray(rtol),
        atol=jnp.asarray(atol),
        maxiter=jnp.asarray(maxiter),
        verbose=verbose > 1,
        print_every_inner=jnp.asarray(print_every),
        U=U2,
        flexible=flexible,
        throw=throw,
    )
    info = {
        "j1": j1,
        "nmv1": nmv1,
        "res1": res1 / jnp.linalg.norm(rhs[:, 0]),
        "success1": success1,
        "j2": j2,
        "nmv2": nmv2,
        "res2": res2 / jnp.linalg.norm(rhs[:, 2]),
        "success2": success2,
        "U1": U1,
        "C1": C1,
        "U2": U2,
        "C2": C2,
    }
    if verbose:
        jax.debug.print(
            "Finished krylov (1st rhs): nmv={nmv:4d}, "
            "n_restarts={j:3d}, residual={res:.3e}",
            nmv=nmv1,
            j=j1,
            res=info["res1"],
            ordered=True,
        )
        jax.debug.print(
            "Finished krylov (2nd rhs): nmv={nmv:4d}, "
            "n_restarts={j:3d}, residual={res:.3e}",
            nmv=nmv2,
            j=j2,
            res=info["res2"],
            ordered=True,
        )
    f = jnp.array([f1, f1, f2])
    sol = MDKESolution(f, rhs.T, field, pitchgrid, nuhat, erhohat)
    return (
        sol,
        info,
    )


def solve_dke(  # noqa: C901
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: MaxwellSpeedGrid,
    species: list[LocalMaxwellian],
    Erho: float | Float[Any, ""],
    EparB: float | Float[Any, ""] = 0.0,
    background: list[LocalMaxwellian] | None = None,
    verbose: bool | int = False,
    multigrid_options: dict | None = None,
    throw: bool = False,
    **options,
) -> tuple[DKESolution, dict[str, jax.Array]]:
    """Solve the drift kinetic equation, giving fluxes.

    Parameters
    ----------
    field : Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Speed grid data.
    species : list[LocalMaxwellian]
        Species information.
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
    EparB : float
        <E||B>, flux surface average of parallel electric field times B.
    background : list[LocalMaxwellian]
        Additional background species to include in the collision operator without
        solving for df.
    verbose: bool, int
        Level of verbosity:

          - 0: no info printed.
          - 1: print initialization info.
          - 2: print info from krylov solver at each iteration. Frequency can be
            controlled by also passing `print_every=<int>`
          - 3: also print residuals at each multigrid level before and after smoothing.
          - 4: also print residuals within smoothing iterations.

        Note that verbose > 2 may slow things down as additional diagnostic info is
        calculated at each step.
    multigrid_options : dict, optional
        Optional parameters to control behavior of multigrid preconditioner.
    throw : bool, optional
        If True, raise a runtime error if the Krylov solver fails to converge
        (forward solve or tangent solve). Default False.

    Returns
    -------
    sol : DKESolution
        Solution object containing distribution function and drive terms and methods
        for computing fluxes and other moments.
    info : dict
        Info about the solve, such as number of iterations, number of matrix-vector
        products, final residual etc.

    """
    # create a copy so we don't modify user input for repeated calls
    multigrid_options = (
        {} if multigrid_options is None else copy.copy(multigrid_options)
    )

    p1 = options.pop("p1", DEFAULT_P1A)
    p2 = options.pop("p2", DEFAULT_P2A)
    rtol = jnp.asarray(options.pop("rtol", 1e-5))
    atol = jnp.asarray(options.pop("atol", 0.0))
    m = options.pop("m", 150)
    k = options.pop("k", 10)
    maxiter = options.pop("maxiter", 10)
    print_every = options.pop("print_every", 10)
    operator_weights = options.pop("operator_weights", jnp.ones(8).at[-1].set(0))
    nL = options.pop("nL", 8)
    quad = options.pop("quad", False)
    skip_init_print = options.pop("skip_init_print", False)
    potentials = options.pop("potentials", None)
    M = options.pop("M", None)
    B = options.pop("B", None)
    C = options.pop("C", None)
    U = options.pop("U", None)
    f1 = options.pop("f1", None)
    coulomb_log = options.pop("coulomb_log", None)
    entropy_norm = options.pop("entropy_norm", True)

    assert len(options) == 0, "solve_dke got unknown option " + str(options)

    if background is None:
        background = []

    Erho = jnp.asarray(Erho)
    EparB = jnp.asarray(EparB)

    if verbose and not skip_init_print:
        _print_field_summary(field)
        _print_species_summary(species, field, speedgrid, background, coulomb_log)
        _print_er_summary(species, field, Erho, EparB)
        _print_thermodynamic_forces(species, field, Erho, EparB)

    if potentials is None:
        potentials = RosenbluthPotentials(speedgrid, species, nL=nL, quad=quad)

    if M is None:
        M = _build_dke_preconditioner(
            field,
            pitchgrid,
            speedgrid,
            species,
            background,
            Erho,
            potentials,
            operator_weights,
            coulomb_log,
            verbose,
            multigrid_options,
        )
    M = _freeze_preconditioner(M)

    if verbose and not skip_init_print:
        M.print_resolution_summary()

    if B is None:
        B = DKESources(field, pitchgrid, speedgrid, species)
    if C is None:
        C = DKEConstraint(field, pitchgrid, speedgrid, species, True)

    A = DKE(
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        Erho=Erho,
        background=background,
        potentials=potentials,
        p1=p1,
        p2=p2,
        gauge=False,
        operator_weights=operator_weights,
        coulomb_log=coulomb_log,
    )

    operator = BorderedOperator(A, B, C)
    preconditioner = InverseBorderedOperator(M, B, C)
    flexible = not _preconditioner_is_linear(M)

    rhs = dke_rhs(field, pitchgrid, speedgrid, species, Erho, EparB, True, True)
    shape = (len(species), speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta)
    size = np.prod(shape)
    if f1 is None:
        f1 = jnp.zeros(size + 2 * len(species))
    else:
        f1 = f1.flatten()
        assert (f1.shape[0] == size) or (f1.shape[0] == (size + 2 * len(species)))
        # maybe pad with zeros for sources
        f1 = jnp.pad(f1, [(0, size + 2 * len(species) - f1.shape[0])])
    if U is None:
        U = jnp.zeros((size + 2 * len(species), k))
    else:
        assert (U.shape[0] == size) or (U.shape[0] == (size + 2 * len(species)))
        U = U.reshape((U.shape[0], -1))
        # maybe pad with zeros for sources
        U = jnp.pad(U, [(0, size + 2 * len(species) - U.shape[0]), (0, 0)])

    if entropy_norm:
        weights = _dke_entropy_weights(species, speedgrid, pitchgrid, field)
    else:
        weights = jnp.ones_like(rhs)

    f1, j1, nmv1, res1, success, C1, U1 = gcrotmk(
        operator,
        rhs,
        x0=f1,
        MR=preconditioner,
        m=m,
        k=k,
        rtol=jnp.asarray(rtol),
        atol=jnp.asarray(atol),
        maxiter=jnp.asarray(maxiter),
        verbose=verbose > 1,
        print_every_inner=jnp.asarray(print_every),
        U=U,
        flexible=flexible,
        throw=throw,
        weights=weights if entropy_norm else None,
    )
    info = {
        "niter": j1,
        "nmv": nmv1,
        "res": res1 / jnp.linalg.norm(jnp.sqrt(weights) * rhs),
        "success": success,
        "C": C1,
        "U": U1,
    }
    if verbose:
        jax.debug.print(
            "Finished krylov: nmv={nmv:4d}, n_restarts={j:3d}, residual={res:.3e}",
            nmv=nmv1,
            j=j1,
            res=info["res"],
            ordered=True,
        )

    sol = _make_dke_solution(
        f1, rhs, field, pitchgrid, speedgrid, species, Erho, EparB, background
    )

    if verbose:
        sol.print_summary()

    return (
        sol,
        info,
    )


def _dke_entropy_weights(species, speedgrid, pitchgrid, field):
    """Residual weights for the bordered DKE system in the entropy norm."""
    # The linearized collision operator is self-adjoint (and streaming/drifts are
    # anti-self-adjoint) in the inner product sum_s T_s int d^3v f_s g_s / F_Ms.
    # Keeping only the species-dependent constant of that weight, T_s vth_s^6 / n_s,
    # makes each species' residual measure its perturbation relative to its own
    # Maxwellian, so the norm is not dominated by the species with the largest
    # F_M. The velocity dependence exp(x^2) is dropped since it would weight the
    # poorly resolved tail most heavily. Constraint rows get the same per-species
    # weight, which keeps the bordered system consistent with the scaled f.
    T = jnp.array([sp.temperature for sp in species])
    n = jnp.array([sp.density for sp in species])
    vth = jnp.array([sp.v_thermal for sp in species])
    ws = T * vth**6 / n
    ws = ws / ws.max()
    nf = speedgrid.nx * pitchgrid.nalpha * field.ntheta * field.nzeta
    return jnp.concatenate([jnp.repeat(ws, nf), jnp.repeat(ws, 2)])


# Loosest linear solve tolerance used while the radial current is far from zero, and
# the factor relating that tolerance to the size of the current. The error in the
# fluxes can be 10-100x larger than the residual of the the linear solve, so the
# factor is small enough that the error stays well under the current itself. Together
# with a floor of 1e-2 * ftol it also means the tolerance reaches that floor once the
# current is within 10x of ftol, so the steps that decide convergence are solved as
# accurately as the root is required to be.
_AMBIPOLAR_RTOL_MAX = 1e-4
_AMBIPOLAR_RTOL_FACTOR = 1e-3

# Half-width of the default search bounds, in units of E* for species[0] at x=1.
_AMBIPOLAR_ESTAR_BOUND = 0.1


class _AmbipolarResidual(eqx.Module):
    """Normalized radial current as a stateful function of Erho, for root finding.

    The state carries the Krylov solution and recycled subspace between calls so that
    each solve is warm started from the previous one, along with running solver
    statistics, the scale the radial current is normalized by and its last value.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    background: list[LocalMaxwellian]
    EparB: jax.Array
    multigrid_options: dict
    options: dict
    rtol: jax.Array
    verbose: int = eqx.field(static=True)
    throw: bool = eqx.field(static=True)
    adaptive_rtol: bool = eqx.field(static=True)

    def solve(self, Erho, f1, U, rtol, verbose):
        return solve_dke(
            self.field,
            self.pitchgrid,
            self.speedgrid,
            self.species,
            Erho,
            self.EparB,
            background=self.background,
            verbose=verbose,
            multigrid_options=self.multigrid_options,
            throw=self.throw,
            f1=f1,
            U=U,
            rtol=rtol,
            **self.options,
        )

    def __call__(self, Erho, state):
        f1, U, niter, nmv, _, scale, fprev = state
        # Inexact Newton: the linear solve only has to locate the next Newton step, so
        # it is solved loosely while the radial current is far from zero and tightened
        # as the root is approached, down to the tolerance that sets how accurately
        # the root itself is resolved.
        rtol = self.rtol
        if self.adaptive_rtol:
            rtol = jnp.clip(
                _AMBIPOLAR_RTOL_FACTOR * jnp.abs(fprev), rtol, _AMBIPOLAR_RTOL_MAX
            )
        sol, info = self.solve(Erho, f1, U, rtol, max(int(self.verbose) - 1, 0))
        qs = jnp.array([sp.species.charge for sp in self.species])
        currents = qs * sol.get("<particle_flux>")
        # The scale is a constant, set before the search. A dynamic scale like
        # sum |q*particle_flux| is a bad choice bc it can leave the normalized value at
        # exactly 1 over a range, leaving no gradient for a search to follow and no
        # sign change to bracket.
        f = currents.sum() / scale
        new_state = (
            sol.f1_krylov,
            info["U"],
            (niter + info["niter"]).astype(niter.dtype),
            (nmv + info["nmv"]).astype(nmv.dtype),
            info["res"],
            jax.lax.stop_gradient(scale),
            jax.lax.stop_gradient(jnp.abs(f)),
        )
        return f, new_state


def solve_dke_ambipolar(  # noqa: C901
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: MaxwellSpeedGrid,
    species: list[LocalMaxwellian],
    num_roots: int,
    *,
    bounds: tuple[float | jax.Array, float | jax.Array] | None = None,
    Erho0: float | Sequence[float] | jax.Array | None = None,
    EparB: float | Float[Any, ""] = 0.0,
    background: list[LocalMaxwellian] | None = None,
    verbose: bool | int = False,
    multigrid_options: dict | None = None,
    throw: bool = False,
    reuse_preconditioner: bool = False,
    scale: str | float = "auto",
    adaptive_rtol: bool = True,
    root_options: dict | None = None,
    **options,
) -> tuple[jax.Array, list[DKESolution], dict[str, jax.Array]]:
    """Find ambipolar radial electric fields and the corresponding DKE solutions.

    Searches for values of Erho = -∂Φ/∂ρ where the radial current J_ρ = Σ_s q_s Γ_s
    vanishes, using a Newton type method with deflation to find multiple roots.

    Parameters
    ----------
    field : Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Speed grid data.
    species : list[LocalMaxwellian]
        Species information.
    num_roots : int
        Maximum number of roots to search for.
    bounds : tuple of float, optional
        Lower and upper bounds on Erho for the search, in Volts. Must be finite: the
        bounds are sampled to bracket the roots, and their separation sets the length
        scale used to deflate the roots already found. By default, the bounds are
        E* = ±0.1, where E* = Erho / (a v_th <B>) is the normalized electric field of
        ``species[0]`` at the thermal speed, with a the minor radius, v_th the thermal
        speed and <B> the flux surface averaged field strength. As E* -> 1, the DKE
        model can break down leading to un-physical roots, so it is recommended to
        keep the default bounds, or tighten them if you know where roots are.
    Erho0 : float or array of float, optional
        Initial guesses for Erho, in Volts, one per search, used before the search
        falls back to looking for sign changes and sampling the bounds. Defaults to
        the midpoint of ``bounds``. Guesses much further from a root than the roots are
        from each other can be worse than none, since each one still costs a search and
        several of them can lead to the same root.
    EparB : float
        <E||B>, flux surface average of parallel electric field times B.
    background : list[LocalMaxwellian]
        Additional background species to include in the collision operator without
        solving for df.
    verbose: bool, int
        Level of verbosity:

          - 0: no info printed.
          - 1: print initialization info, progress of the root finding and a summary
            of the solution at each root.
          - 2: also print Newton iterations and info from krylov solver at each
            iteration.
          - 3+: also print multigrid diagnostics, as in ``solve_dke``.

    multigrid_options : dict, optional
        Optional parameters to control behavior of multigrid preconditioner.
    throw : bool, optional
        If True, raise a runtime error if a Krylov solve fails to converge.
    reuse_preconditioner : bool, optional
        If True, build the preconditioner once at the initial guess and use it for all
        solves. This avoids rebuilding it at every Newton step, at the cost of a less
        effective preconditioner far from the initial guess. If False, the
        preconditioner is rebuilt at each value of Erho.
    scale : "auto" or float, optional
        Scale J used to normalize the radial current, so that roots are found for the
        residual Σ_s q_s Γ_s / J. With "auto", J = Σ_s |q_s Γ_s| at the initial guess.
        Otherwise the value given is used, in A·m⁻³. Either way J is a constant, so
        the residual is proportional to the radial current; ``ftol`` is then measured
        relative to the rescaled value.
    adaptive_rtol : bool, optional
        Whether to solve the DKE more loosely while the radial current is far from
        zero, tightening to ``rtol`` as a root is approached.
    root_options : dict, optional
        Options passed to ``deflated_root_scalar``, such as ``ftol`` (stopping
        tolerance on the normalized radial current, default ``1e-4``,
        ``xrtol``/``xatol`` (stopping tolerance on the Newton step in Erho, ``xatol``
        in Volts), ``maxiter`` (maximum iterations per search), ``method`` (``"secant"``
        or ``"newton"``), ``max_stall``, ``probe_steps``, ``interior_samples``,
        ``best_searches`` and ``history_size``. See that function for details; the
        defaults suit most cases.
    **options : dict, optional
        Additional options passed to ``solve_dke``. The Krylov tolerance ``rtol``
        defaults to ``1e-2 * ftol``, and sets the accuracy of the converged roots. The
        size of the recycled Krylov subspace ``k`` defaults to ``10*num_roots``, since
        it is carried across all the solves of the search.

    Returns
    -------
    Erho : jax.Array, shape (num_roots,)
        Ambipolar values of Erho = -∂Φ /∂ρ, in Volts. Roots that were not found are
        set to inf.
    sols : list[DKESolution]
        Solution at each root. Solutions for roots that were not found contain nan.
    info : dict
        Info about the search, containing, for each root:

          - "success": whether the root was found.
          - "residual": normalized radial current at the root.
          - "newton_steps": number of iterations of the search that found the root.
          - "niter": number of Krylov restarts since the previous root was found.
          - "nmv": number of matrix-vector products since the previous root was found.
          - "res": relative Krylov residual of the solve at the root.

        and, for the search as a whole, "scale", the scale J used to normalize the
        radial current, in A·m⁻³, and "nmv_total", the total number of matrix-vector
        products, including unsuccessful searches.

    """
    multigrid_options = (
        {} if multigrid_options is None else copy.copy(multigrid_options)
    )
    for key in ["f1", "U", "skip_init_print"]:
        if key in options:
            raise ValueError(f"solve_dke_ambipolar does not accept option '{key}'")
    root_options = {} if root_options is None else copy.copy(root_options)
    root_options.setdefault("ftol", 1e-4)
    for key in [
        "args",
        "bounds",
        "carry_state",
        "full_output",
        "state_filter",
        "verbose",
    ]:
        if key in root_options:
            raise ValueError(f"root_options does not accept option '{key}'")
    if background is None:
        background = []

    if bounds is None:
        # E* = Erho / Escale, normalized to the thermal speed of species[0] at x = 1
        Escale = field.a_minor * species[0].v_thermal * field.Bmag_fsa
        bounds = (-_AMBIPOLAR_ESTAR_BOUND * Escale, _AMBIPOLAR_ESTAR_BOUND * Escale)
    lower, upper = (jnp.asarray(b, dtype=float) for b in bounds)
    lower, upper = cast(
        tuple[jax.Array, jax.Array],
        eqx.error_if(
            (lower, upper), ~(lower < upper), "bounds must satisfy lower < upper"
        ),
    )
    lower, upper = cast(
        tuple[jax.Array, jax.Array],
        eqx.error_if(
            (lower, upper),
            ~(jnp.isfinite(lower) & jnp.isfinite(upper)),
            "bounds must be finite",
        ),
    )
    if Erho0 is None:
        Erho0 = (lower + upper) / 2
    starts = jnp.atleast_1d(jnp.asarray(Erho0, dtype=float))
    starts = cast(
        jax.Array,
        eqx.error_if(
            starts,
            ~jnp.all((starts >= lower) & (starts <= upper)),
            "Erho0 must be within bounds",
        ),
    )
    scale0 = None
    if isinstance(scale, str):
        if scale != "auto":
            raise ValueError(f"scale must be 'auto' or a float, got '{scale}'")
    else:
        scale0 = jnp.asarray(scale, dtype=float)
        scale0 = cast(
            jax.Array, eqx.error_if(scale0, ~(scale0 > 0), "scale must be positive")
        )
    # the preconditioner and the automatic scale are built at the first guess
    x0 = starts[0]
    EparB = jnp.asarray(EparB)

    rtol = options.pop("rtol", 1e-2 * root_options["ftol"])
    k = options.setdefault("k", 10 * num_roots)
    options.setdefault("operator_weights", jnp.ones(8).at[-1].set(0))
    # these all don't depend on Er so we can build once and amortize
    nL = options.pop("nL", 8)
    quad = options.pop("quad", False)
    if options.get("potentials") is None:
        options["potentials"] = RosenbluthPotentials(
            speedgrid, species, nL=nL, quad=quad
        )
    if options.get("B") is None:
        options["B"] = DKESources(field, pitchgrid, speedgrid, species)
    if options.get("C") is None:
        options["C"] = DKEConstraint(field, pitchgrid, speedgrid, species, True)

    # the preconditioner depends on Er but only needs to be approximate, so in some
    # cases its cheaper to re-use it.
    given_preconditioner = options.get("M") is not None
    if reuse_preconditioner and not given_preconditioner:
        options["M"] = _freeze_preconditioner(
            _build_dke_preconditioner(
                field,
                pitchgrid,
                speedgrid,
                species,
                background,
                x0,
                options["potentials"],
                options["operator_weights"],
                options.get("coulomb_log"),
                verbose,
                multigrid_options,
            )
        )
    options["skip_init_print"] = True

    if verbose:
        _print_field_summary(field)
        _print_species_summary(
            species, field, speedgrid, background, options.get("coulomb_log")
        )
        if given_preconditioner:
            options["M"].print_resolution_summary()
        else:
            _print_dke_resolution_summary(
                field, pitchgrid, speedgrid, species, multigrid_options
            )

    residual = _AmbipolarResidual(
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        background=background,
        EparB=EparB,
        multigrid_options=multigrid_options,
        options=options,
        rtol=jnp.asarray(rtol),
        verbose=int(verbose),
        throw=throw,
        adaptive_rtol=adaptive_rtol,
    )

    ns = len(species)
    size = ns * speedgrid.nx * pitchgrid.nalpha * field.ntheta * field.nzeta + 2 * ns
    f10 = jnp.zeros(size)
    U0 = jnp.zeros((size, k))
    nmv0 = jnp.array(0)
    if scale0 is None:
        # The scale must be the same for every search, so an automatic scale has to be
        # computed before the search rather than taken from wherever it first
        # evaluates. That solve also warm starts the search, which would otherwise
        # start from zero, so it is not simply an extra cost.
        sol0, info0 = residual.solve(
            x0, f10, U0, jnp.asarray(rtol), max(int(verbose) - 1, 0)
        )
        qs = jnp.array([sp.species.charge for sp in species])
        scale0 = jnp.abs(qs * sol0.get("<particle_flux>")).sum()
        f10, U0, nmv0 = sol0.f1_krylov, info0["U"], info0["nmv"]
    state0 = (
        f10,
        U0,
        jnp.array(0),
        jnp.array(0),
        jnp.array(jnp.nan),
        jax.lax.stop_gradient(scale0),
        jnp.array(jnp.inf),
    )

    Erhos, (_, success, fs, newton_steps, states, searches) = deflated_root_scalar(
        residual,
        starts,
        num_roots,
        args=state0,
        bounds=(lower, upper),
        carry_state=True,
        full_output=True,
        # The state holds the solution and recycled subspace, each the size of the
        # problem, so only what is used below is kept.
        state_filter=lambda state: (state[0], state[2], state[3], state[4]),
        verbose=verbose,
        **root_options,
    )
    f1s, niter, nmv, res = states
    _, (_, _, nmv_last, _) = searches

    sols = []
    for j in range(num_roots):
        rhs = dke_rhs(field, pitchgrid, speedgrid, species, Erhos[j], EparB, True, True)
        f1 = jnp.where(success[j], f1s[j], jnp.nan)
        sols.append(
            _make_dke_solution(
                f1,
                rhs,
                field,
                pitchgrid,
                speedgrid,
                species,
                Erhos[j],
                EparB,
                background,
            )
        )

    # The state is carried from one search to the next, so its counters are running
    # totals, and roots are found in order, so the cost of each root is the difference
    # from the one before.
    niter = cast(jax.Array, jnp.where(success, niter, 0))
    nmv = cast(jax.Array, jnp.where(success, nmv, 0))
    info = {
        "success": success,
        "residual": fs,
        "newton_steps": newton_steps,
        "niter": jnp.where(success, jnp.diff(niter, prepend=0), 0),
        "nmv": jnp.where(success, jnp.diff(nmv, prepend=0), 0),
        "res": jnp.where(success, res, jnp.nan),
        "scale": scale0,
        "nmv_total": nmv_last + nmv0,
    }
    if verbose:
        for j in range(num_roots):
            jax.debug.print(
                "Root {j:2d}: success={s},  Eᵨ={Erho: .6e} (V),  "
                "residual={f: .3e},  newton steps={k:3d},  nmv={nmv:5d}",
                j=j,
                s=success[j],
                Erho=Erhos[j],
                f=fs[j],
                k=newton_steps[j],
                nmv=info["nmv"][j],
                ordered=True,
            )
            # branches are wrapped in lambdas since cond hashes them, and modules with
            # array fields are unhashable
            jax.lax.cond(success[j], lambda: sols[j].print_summary(), lambda: None)

    return Erhos, sols, info


def _print_species_summary(species, field, speedgrid, background, coulomb_log=None):
    for si, spec in enumerate(species):
        jax.debug.print(
            "Species {si:2d}:  "
            + "m={mass: .2e} (mₚ)      "
            + "q={charge: .2e} (qₚ)\n"
            + " " * 13
            + "n={dens: .2e} (m⁻³)  "
            + "a/Lₙ={L_n: .2e}\n"
            + " " * 13
            + "T={temp: .2e} (eV)   "
            + "a/Lᴛ={L_T: .2e}  ",
            si=si,
            mass=spec.species.mass / proton_mass,
            charge=spec.species.charge / elementary_charge,
            dens=spec.density,
            temp=spec.temperature,
            L_n=spec.aLn,
            L_T=spec.aLT,
            ordered=True,
        )
        others = species[:si] + species[si + 1 :] + background
        tempx = jnp.array([speedgrid.x[0], 1.0, speedgrid.x[-1]])
        nustars = nustar(spec, field, tempx, *others, lnlambda=coulomb_log)
        for nu, x in zip(nustars, tempx):
            jax.debug.print(
                " " * 13 + "ν* (x={x:.2e}): {nu: .3e}", x=x, nu=nu, ordered=True
            )


def _print_er_summary(species, field, Erho, EparB):
    jax.debug.print("<E||B> : {EparB: .2e} (V*T/m)", EparB=EparB)
    jax.debug.print("Eᵨ = -∂Φ /∂ρ: {Erho: .2e} (V)", Erho=Erho)
    erstars = jnp.array([Estar(spec, field, Erho, 1.0) for spec in species])
    s = "E* (x=1.0): [" + "{: .3e} " * len(species) + "] (per species)"
    jax.debug.print(s, *erstars, ordered=True)
    machs = jnp.array([poloidal_mach(spec, field, Erho, 1.0) for spec in species])
    s = "Mₚ (x=1.0): [" + "{: .3e} " * len(species) + "] (per species)"
    jax.debug.print(s, *machs, ordered=True)


def _print_thermodynamic_forces(species, field, Erho, EparB):
    forces = _dke_thermodynamic_forces(species, field, Erho, EparB)
    s = "A₁: [" + "{: .3e} " * len(species) + "] (per species)"
    jax.debug.print(s, *forces[0], ordered=True)
    s = "A₂: [" + "{: .3e} " * len(species) + "] (per species)"
    jax.debug.print(s, *forces[1], ordered=True)
    s = "A₃: [" + "{: .3e} " * len(species) + "] (per species)"
    jax.debug.print(s, *forces[2], ordered=True)


def _effective_trapped_fraction(field: Field, nlam: int = 64) -> Float[Any, ""]:
    """Effective trapped-particle fraction f_t = 1 - f_c."""
    Bmax = field.Bmag.max()
    lam = jnp.linspace(0.0, 1.0 / Bmax, nlam)
    denom: jax.Array = jax.vmap(
        lambda l: field.flux_surface_average(
            jnp.sqrt(jnp.maximum(1.0 - l * field.Bmag, 0.0))
        )
    )(lam)
    integrand = jnp.where(denom > 0, lam / denom, jnp.array(0.0))
    fc = 0.75 * field.B2mag_fsa * jnp.trapezoid(integrand, lam)
    return 1.0 - fc


def _print_field_summary(field: Field) -> None:
    # relative RMS variation of |B| on the surface: a ripple/trapping proxy
    # that, unlike Bmax/Bmin, sees the whole |B| landscape rather than extremes
    ripple = jnp.sqrt(field.B2mag_fsa / field.Bmag_fsa**2 - 1.0)
    mirror = field.Bmag.max() / field.Bmag.min()
    ftrap = _effective_trapped_fraction(field)
    # source is an int-code enum leaf (jit/trace transparent); bake each label
    # into its own format string and select with lax.switch so it renders
    # whether the field is concrete or traced under an outer jit.
    body = (
        "):\n"
        "    ρ         = {rho: .3f}              ι         = {iota: .3e}\n"
        "    <B>       = {Bavg: .3e} T        δ_B       = {ripple: .3e}\n"
        "    Bmax/Bmin = {mirror: .3e}          f_trapped = {ftrap: .3e}\n"
        "    I         = {I: .3e} T·m      G         = {G: .3e} T·m"
    )

    def _printer(name: str):
        return lambda: jax.debug.print(
            "Field info (source: " + name + body,
            rho=field.rho,
            iota=field.iota,
            Bavg=field.Bmag_fsa,
            mirror=mirror,
            ripple=ripple,
            ftrap=ftrap,
            I=field.I,
            G=field.G,
        )

    jax.lax.switch(
        field.source._value,
        [_printer(name) for name in FIELD_SOURCE_NAMES],
    )
