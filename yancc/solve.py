"""Main interface for solving drift kinetic equations in yancc."""

import jax.numpy as jnp
import lineax as lx

from .collisions import RosenbluthPotentials
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
from .species import Estar, nustar
from .trajectories import DKE, MDKE


def solve_mdke(field, pitchgrid, erhohat, nuhat, **options):
    """Solve the mono-energetic drift kinetic equation, giving 3x3 transport matrix.

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


    Returns
    -------
    Dij : jax.Array, shape(3,3)
        Monoenergetic transport coefficients
    f : jax.Array, shape(N,3)
        Solution of DKE for each right hand side.
    rhs : jax.Array, shape(N,3)
        Source terms for DKE.
    info : dict
        Info about the solve, such as number of iterations, number of matrix-vector
        products, final residual etc.

    """
    p1a = options.pop("p1a", "4d")
    p2a = options.pop("p2a", 4)
    rtol = jnp.asarray(options.pop("rtol", 1e-5))
    atol = jnp.asarray(options.pop("atol", 0.0))
    m = options.pop("m", 300)
    k = options.pop("k", 5)
    maxiter = options.pop("maxiter", 5)
    print_every = options.pop("print_every", 0)

    M = MDKEPreconditioner(
        field=field, pitchgrid=pitchgrid, nuhat=nuhat, erhohat=erhohat, **options
    )
    A = MDKE(
        field,
        pitchgrid,
        erhohat,
        nuhat,
        p1=p1a,
        p2=p2a,
        gauge=True,
    )
    rhs = mdke_rhs(field, pitchgrid)

    x0 = jnp.zeros_like(rhs[:, 0])
    f1, j1, nmv1, res1, _, _ = gcrotmk(
        A,
        rhs[:, 0],
        x0=x0,
        MR=M,
        m=m,
        k=k,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        print_every=print_every,
    )
    f3, j3, nmv3, res3, _, _ = gcrotmk(
        A,
        rhs[:, 3],
        x0=x0,
        MR=M,
        m=m,
        k=k,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        print_every=print_every,
    )
    f2 = f1.copy()
    f = jnp.array([f1, f2, f3]).T
    Dij = compute_monoenergetic_coefficients(f, field, pitchgrid)
    return (
        Dij,
        f,
        rhs,
        {
            "j1": j1,
            "nmv1": nmv1,
            "res1": res1 / jnp.linalg.norm(rhs[:, 0]),
            "j2": j3,
            "nmv2": nmv3,
            "res2": res3 / jnp.linalg.norm(rhs[:, 3]),
        },
    )


def solve_dke(field, pitchgrid, speedgrid, species, Erho, **options):
    """Solve the mono-energetic drift kinetic equation, giving 3x3 transport matrix.

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

    Returns
    -------
    f` : jax.Array, shape(N,)
        Perturbed distribution function, solution of DKE
    rhs : jax.Array, shape(N,)
        Drive term from leading order Maxwellian. Right hand side of DKE.
    fluxes: dict of ndarray
        Contains:
        particle_flux : jax.Array, shape(ns)
            Γₐ = Particle flux for each species.
        heat_flux : jax.Array, shape(ns)
            Qₐ = Heat flux for each species.
        Vpar : jax.Array, shape(ns, nt, nz)
            V|| = Parallel velocity for each species
        BVpar: jax.Array, shape(ns)
            〈BV||〉 = Flux surface average field*parallel velocity for each species
        bootstrap_current: float
            〈J||B〉 = Bootstrap current.
        radial_current : float
            Jr = Radial current.
        Jpar : jax.Array, shape(nt, nz)
            J|| = Parallel current density.
    stats : dict
        Info about the solve, such as number of iterations, number of matrix-vector
        products, final residual etc.

    """
    p1a = options.pop("p1a", "4d")
    p2a = options.pop("p2a", 4)
    p1b = options.pop("p1b", "2d")
    p2b = options.pop("p2b", 2)
    rtol = jnp.asarray(options.pop("rtol", 1e-5))
    atol = jnp.asarray(options.pop("atol", 0.0))
    m = options.pop("m", 150)
    k = options.pop("k", 10)
    maxiter = options.pop("maxiter", 10)
    print_every = options.pop("print_every", 0)
    nL = options.pop("nL", 4)
    quad = options.pop("quad", False)
    verbose = options.get("verbose", 0)

    potentials = RosenbluthPotentials(speedgrid, species, nL=nL, quad=quad)

    M = DKEPreconditioner(
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        Erho=Erho,
        potentials=potentials,
        gauge=True,
        p1=p1b,
        p2=p2b,
        **options,
    )
    A = DKE(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho,
        potentials=potentials,
        p1=p1a,
        p2=p2a,
        gauge=True,
    )
    if verbose or print_every:
        for iop, op in enumerate(M.operators):
            assert isinstance(op, DKE)
            print(
                f"Grid {iop}: nx={op.speedgrid.nx:4d}, "
                f"na={op.pitchgrid.nxi:4d}, "
                f"nt={op.field.ntheta:4d}, "
                f"nz={op.field.nzeta:4d}, "
                f"N={op.in_structure().size:4d}"
            )

        for spec in species:
            print(spec)
            x = speedgrid.x[0]
            print(f"ν* (x={x:.2e}): {nustar(spec, field, x):.3e}")
            print(f"E* (x={x:.2e}): {Estar(spec, field, Erho, x):.3e}")
            x = 1.0
            print(f"ν* (x={x:.2e}): {nustar(spec, field, x):.3e}")
            print(f"E* (x={x:.2e}): {Estar(spec, field, Erho, x):.3e}")
            x = speedgrid.x[-1]
            print(f"ν* (x={x:.2e}): {nustar(spec, field, x):.3e}")
            print(f"E* (x={x:.2e}): {Estar(spec, field, Erho, x):.3e}")

    B = DKESources(field, pitchgrid, speedgrid, species)
    C = DKEConstraint(field, pitchgrid, speedgrid, species, True)
    D = lx.MatrixLinearOperator(jnp.zeros((len(species) * 2, len(species) * 2)))

    operator = BorderedOperator(A, B, C, D)
    preconditioner = InverseBorderedOperator(M, B, C, D)

    rhs = dke_rhs(field, pitchgrid, speedgrid, species, Erho, True, False)

    x0 = jnp.zeros_like(rhs)
    f1, j1, nmv1, res1, _, _ = gcrotmk(
        operator,
        rhs,
        x0=x0,
        MR=preconditioner,
        m=m,
        k=k,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        print_every=print_every,
    )
    stats = {"niter": j1, "nmv": nmv1, "res": res1 / jnp.linalg.norm(rhs)}

    fluxes = compute_fluxes(
        f1,
        field,
        pitchgrid,
        speedgrid,
        species,
    )
    shape = (len(species), speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta)

    return f1.reshape(shape), rhs.reshape(shape), fluxes, stats
