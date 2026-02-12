"""Main interface for solving drift kinetic equations in yancc."""

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from scipy.constants import elementary_charge, proton_mass

from .collisions import RosenbluthPotentials
from .field import Field
from .krylov import gcrotmk
from .linalg import BorderedOperator, InverseBorderedOperator
from .misc import (
    DKEConstraint,
    DKESources,
    compute_fluxes,
    compute_monoenergetic_coefficients,
    dke_rhs,
    mdke_rhs,
)
from .preconditioner import DKEPreconditioner, MDKEPreconditioner
from .species import Estar, LocalMaxwellian, nustar
from .trajectories import DKE, MDKE
from .velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def solve_mdke(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    erhohat: Union[float, Float[Any, ""]],
    nuhat: Union[float, Float[Any, ""]],
    verbose: Union[bool, int] = False,
    multigrid_options: Optional[dict] = None,
    **options,
):
    """Solve the mono-energetic drift kinetic equation, giving 3x3 transport matrix.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v = -∂Φ /∂ρ /v in units of V*s/m.
    nuhat : float
        Monoenergetic collisionality, ν/v in units of 1/m.
    verbose: bool, int
        Level of verbosity:
          - 0: no into printed.
          - 1: print initialization info.
          - 2: print info from krylov solver at each iteration. Frequency can be
          controlled by also passing `print_every=<int>`
          - 3: also print residuals at each multigrid level before and after smoothing.
          - 4: also print residuals within smoothing iterations.
        Note that verbose > 2 may slow things down as additional diagnostic info is
        calculated at each step.
    multigrid_options : dict, optional
        Optional parameters to control behavior of multigrid preconditioner.

    Returns
    -------
    f : jax.Array, shape(N,3)
        Solution of DKE for each right hand side.
    rhs : jax.Array, shape(N,3)
        Source terms for DKE.
    Dij : jax.Array, shape(3,3)
        Monoenergetic transport coefficients
    info : dict
        Info about the solve, such as number of iterations, number of matrix-vector
        products, final residual etc.

    """
    multigrid_options = {} if multigrid_options is None else multigrid_options

    p1 = options.pop("p1", "4d")
    p2 = options.pop("p2", 4)
    rtol = jnp.asarray(options.pop("rtol", 1e-5))
    atol = jnp.asarray(options.pop("atol", 0.0))
    m = options.pop("m", 300)
    k = options.pop("k", 5)
    maxiter = options.pop("maxiter", 5)
    print_every = options.pop("print_every", 10)
    U1 = options.pop("U1", None)
    U2 = options.pop("U2", None)
    f1 = options.pop("f1", None)
    f2 = options.pop("f2", None)

    assert len(options) == 0, "solve_mdke got unknown option " + str(options)

    if verbose:
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
    f1, j1, nmv1, res1, C1, U1 = gcrotmk(
        A,
        rhs[:, 0],
        x0=f1,
        MR=M,
        m=m,
        k=k,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        print_every=print_every if verbose > 1 else 0,
        U=U1,
    )
    if f2 is None:
        f2 = jnp.zeros_like(rhs[:, 0])
    f2, j2, nmv2, res2, C2, U2 = gcrotmk(
        A,
        rhs[:, 2],
        x0=f2,
        MR=M,
        m=m,
        k=k,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        print_every=print_every if verbose > 1 else 0,
        U=U2,
    )
    info = {
        "j1": j1,
        "nmv1": nmv1,
        "res1": res1 / jnp.linalg.norm(rhs[:, 0]),
        "j2": j2,
        "nmv2": nmv2,
        "res2": res2 / jnp.linalg.norm(rhs[:, 2]),
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
    Dij = compute_monoenergetic_coefficients(f, field, pitchgrid)
    return (
        f,
        rhs,
        Dij,
        info,
    )


def solve_dke(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: MaxwellSpeedGrid,
    species: list[LocalMaxwellian],
    Erho: Union[float, Float[Any, ""]],
    EparB: Union[float, Float[Any, ""]] = 0.0,
    verbose: Union[bool, int] = False,
    multigrid_options: Optional[dict] = None,
    **options,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array], dict[str, jax.Array]]:
    """Solve the drift kinetic equation, giving fluxes.

    Parameters
    ----------
    field : yancc.Field
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
    verbose: bool, int
        Level of verbosity:
          - 0: no into printed.
          - 1: print initialization info.
          - 2: print info from krylov solver at each iteration. Frequency can be
          controlled by also passing `print_every=<int>`
          - 3: also print residuals at each multigrid level before and after smoothing.
          - 4: also print residuals within smoothing iterations.
        Note that verbose > 2 may slow things down as additional diagnostic info is
        calculated at each step.
    multigrid_options : dict, optional
        Optional parameters to control behavior of multigrid preconditioner.

    Returns
    -------
    f : jax.Array, shape(ns,nx,na,nt,nz)
        Distribution function f = F0 + f1 where F0 is the leading order Maxwellian and
        f1 is the perturbation.
    rhs : jax.Array, shape(ns,nx,na,nt,nz)
        Drive term from leading order Maxwellian. Right hand side of DKE.
    fluxes: dict of jax.Array
        Contains:
        <particle_flux> : jax.Array, shape(ns)
            Γₐ = FSA particle flux for each species, in particles/(meter² second).
        <heat_flux> : jax.Array, shape(ns)
            Qₐ = FSA heat flux for each species, in Joules/(meter² second)
        V|| : jax.Array, shape(ns, nt, nz)
            V|| = Parallel velocity for each species, in meters/second
        <BV||>: jax.Array, shape(ns)
            <BV||> = Flux surface average field*parallel velocity for each species,
            in Tesla*meter/second
        <J||B>: float
            <J||B> = Bootstrap current, in Tesla*Amps/meter².
        J_rho : float
            J_rho = Radial current, in Amps/meter².
        J|| : jax.Array, shape(nt, nz)
            J|| = Parallel current density in Amps/meter².
    stats : dict
        Info about the solve, such as number of iterations, number of matrix-vector
        products, final residual etc.

    """
    multigrid_options = {} if multigrid_options is None else multigrid_options

    p1 = options.pop("p1", "4d")
    p2 = options.pop("p2", 4)
    rtol = jnp.asarray(options.pop("rtol", 1e-5))
    atol = jnp.asarray(options.pop("atol", 0.0))
    m = options.pop("m", 150)
    k = options.pop("k", 10)
    maxiter = options.pop("maxiter", 10)
    print_every = options.pop("print_every", 10)
    operator_weights = options.pop("operator_weights", jnp.ones(8).at[-1].set(0))
    nL = options.pop("nL", 4)
    quad = options.pop("quad", False)
    skip_init_print = options.pop("skip_init_print", False)
    potentials = options.pop("potentials", None)
    M = options.pop("M", None)
    B = options.pop("B", None)
    C = options.pop("C", None)
    U = options.pop("U", None)
    f1 = options.pop("f1", None)

    assert len(options) == 0, "solve_dke got unknown option " + str(options)

    if verbose and not skip_init_print:
        _print_species_summary(species, field, speedgrid)
        _print_er_summary(species, field, Erho)

    if potentials is None:
        potentials = RosenbluthPotentials(speedgrid, species, nL=nL, quad=quad)

    if M is None:
        if len(species) > 1:
            default_operator_weights = operator_weights.at[-2].set(0)
        else:
            default_operator_weights = operator_weights
        multigrid_options.setdefault("operator_weights", default_operator_weights)
        multigrid_options.setdefault("field", field)
        multigrid_options.setdefault("pitchgrid", pitchgrid)
        multigrid_options.setdefault("speedgrid", speedgrid)
        multigrid_options.setdefault("species", species)
        multigrid_options.setdefault("Erho", Erho)
        multigrid_options.setdefault("potentials", potentials)
        multigrid_options.setdefault("gauge", True)
        multigrid_options.setdefault("verbose", verbose)
        M = DKEPreconditioner(**multigrid_options)

    if verbose and not skip_init_print:
        _print_dke_resolutions(M)

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
        potentials=potentials,
        p1=p1,
        p2=p2,
        gauge=False,
        operator_weights=operator_weights,
    )

    operator = BorderedOperator(A, B, C)
    preconditioner = InverseBorderedOperator(M, B, C)

    rhs = dke_rhs(field, pitchgrid, speedgrid, species, Erho, EparB, True, True)
    shape = (len(species), speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta)
    size = np.prod(shape)
    if U is not None:
        assert U.shape[0] == size
        U = U.reshape((size, -1))
        U = jnp.pad(U, [(0, 2 * len(species)), (0, 0)])
    if f1 is not None:
        f1 = f1.flatten()
        assert f1.size == size
        f1 = jnp.pad(f1, [(0, 2 * len(species))])

    f1, j1, nmv1, res1, C1, U1 = gcrotmk(
        operator,
        rhs,
        x0=f1,
        MR=preconditioner,
        m=m,
        k=k,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        print_every=print_every if verbose > 1 else 0,
        U=U,
    )
    info = {
        "niter": j1,
        "nmv": nmv1,
        "res": res1 / jnp.linalg.norm(rhs),
        "C": C1[:size],
        "U": U1[:size],
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
    F0 = jnp.tile(
        F0[:, :, None, None, None], (1, 1, pitchgrid.nxi, field.ntheta, field.nzeta)
    )
    F0 = jnp.concatenate([F0.flatten(), jnp.zeros(2 * len(species))])

    f = F0 + f1

    fluxes = compute_fluxes(
        f,
        field,
        pitchgrid,
        speedgrid,
        species,
    )

    return (
        f[: np.prod(shape)].reshape(shape),
        rhs[: np.prod(shape)].reshape(shape),
        fluxes,
        info,
    )


def _print_species_summary(species, field, speedgrid):
    for si, spec in enumerate(species):
        jax.debug.print(
            "Species {si}:  "
            "m={mass: .2e} (mₚ)  "
            "q={charge: .2e} (qₚ)  "
            "n={dens: .2e} (m⁻³)  "
            "T={temp: .2e} (eV)  ",
            si=si,
            mass=spec.species.mass / proton_mass,
            charge=spec.species.charge / elementary_charge,
            dens=spec.density,
            temp=spec.temperature,
            ordered=True,
        )
        tempx = jnp.array([speedgrid.x[0], 1.0, speedgrid.x[-1]])
        nustars = nustar(spec, field, tempx)
        for nu, x in zip(nustars, tempx):
            jax.debug.print("ν* (x={x:.2e}): {nu: .3e}", x=x, nu=nu, ordered=True)


def _print_er_summary(species, field, Erho):
    erstars = jnp.array([Estar(spec, field, Erho, 1.0) for spec in species])
    s = "E* (x=1.0): [" + "{: .3e} " * len(species) + "] (per species)"
    jax.debug.print(s, *erstars, ordered=True)


def _print_dke_resolutions(preconditioner):
    for i, op in enumerate(preconditioner.operators):
        ns = len(op.species)
        nx = op.speedgrid.nx
        na = op.pitchgrid.nxi
        nt = op.field.ntheta
        nz = op.field.nzeta
        # these values aren't traced so we can use regular print
        print(
            f"Grid {i}: nx={nx:4d}, "
            f"na={na:4d}, "
            f"nt={nt:4d}, "
            f"nz={nz:4d}, "
            f"N={ns*nx*na*nt*nz}"
        )
