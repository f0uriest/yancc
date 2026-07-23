"""Main interface for solving drift kinetic equations in yancc."""

import copy
from typing import Any

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
from .preconditioner import DKEPreconditioner, MDKEPreconditioner
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
    f = jnp.array([f1, f1, f2]).T
    sol = MDKESolution(f, rhs, field, pitchgrid, nuhat, erhohat)
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
        M = DKEPreconditioner(**multigrid_options)
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
    )
    info = {
        "niter": j1,
        "nmv": nmv1,
        "res": res1 / jnp.linalg.norm(rhs),
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

    F0 = jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    F0 = F0[:, :, None, None, None]

    sol = DKESolution(
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

    if verbose:
        sol.print_summary()

    return (
        sol,
        info,
    )


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
