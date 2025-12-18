"""Main interface for solving drift kinetic equations in yancc."""

import jax.numpy as jnp

from .krylov import gcrotmk
from .misc import compute_monoenergetic_coefficients, mdke_rhs
from .preconditioner import MDKEPreconditioner
from .trajectories import MDKE


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
