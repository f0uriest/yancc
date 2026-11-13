"""Tests for the preconditioner for axisymmetric fields."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import yancc
from yancc._linalg import BorderedOperator
from yancc._misc import DKEConstraint, DKESources, dke_rhs
from yancc._tokamak import TokamakPreconditioner, _PitchThetaSolver
from yancc._trajectories import DKE
from yancc.field import Field
from yancc.solve import solve_dke, solve_dke_ambipolar

from .test_solve import _jvp_1_arg


@pytest.fixture(scope="module")
def tokamak():
    """Axisymmetric field, with more theta nodes than the widest stencil."""
    theta = np.linspace(0, 2 * np.pi, 11, endpoint=False)[:, None]
    return Field.from_boozer(
        rho=0.5,
        Bmag=2.5 - 0.25 * np.cos(theta) + 0.05 * np.sin(2 * theta),
        I=0.0,
        G=14.4,
        iota=-0.88356,
        Psi=-2.0004,
        R_major=5.4832,
        a_minor=0.5211,
        NFP=1,
    )


def _fluxes(sol):
    return jnp.concatenate(
        [sol.get("<particle_flux>"), sol.get("<heat_flux>"), sol.get("<V||B>")]
    )


@pytest.mark.parametrize("p1", ["4d", "4z", "2d"])
def test_pitch_theta_solver(tokamak, pitchgrid, speedgrid, species2, p1):
    """The plane solve inverts the terms of the DKE that it keeps."""
    w = jnp.array([0.9, 1.1, 1.2, 1.0, 0.8, 1.3, 0.7, 0.1])
    A = DKE(tokamak, pitchgrid, speedgrid, species2, 1e3, p1=p1, operator_weights=w)
    # 9 pitch nodes are not a multiple of the group size, so the last group is padded
    planes = _PitchThetaSolver(A)
    assert pitchgrid.nalpha % len(planes.groups[0]) != 0

    shift = (
        w[0] * A._opx.diagonal()
        + w[5] * A._C.CE.diagonal()
        + w[6] * A._C.CF.diagonal()
        + w[7]
    )
    f = jax.random.normal(jax.random.PRNGKey(0), (A.in_size(),))
    Pf = w[1] * A._opa.mv(f) + w[2] * A._opt.mv(f) + w[4] * A._C.CL.mv(f) + shift * f
    np.testing.assert_allclose(planes.solve(Pf), f, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("p1", ["4d", "4z"])
def test_tokamak_preconditioner(tokamak, pitchgrid, speedgrid, species2, p1):
    """Coarse level is the Galerkin inverse, transposes are consistent."""
    A = DKE(tokamak, pitchgrid, speedgrid, species2, 1e3, p1=p1, gauge=False)
    B = DKESources(tokamak, pitchgrid, speedgrid, species2)
    C = DKEConstraint(tokamak, pitchgrid, speedgrid, species2, True)
    op = BorderedOperator(A, B, C)
    ns, nx, moments = len(species2), speedgrid.nx, 4
    M = TokamakPreconditioner(op, moments)

    # moments of the shapes are the identity
    fs = M.coarse.fluid
    np.testing.assert_allclose(
        jnp.einsum("sixa,sjxa->sij", fs.Psi, fs.V),
        jnp.broadcast_to(jnp.eye(nx * moments), (ns, nx * moments, nx * moments)),
        atol=1e-12,
    )

    # the coarse solve inverts R A P. The bordered system is poorly conditioned (the
    # sources are only weakly determined), so check its residual and not its error
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    c0 = jax.random.normal(keys[0], (ns, nx * moments, tokamak.ntheta, 1))
    h0 = jax.random.normal(keys[1], (2 * ns,))
    v = op.mv((fs.prolong(c0), h0))
    c1, h1 = fs.solve(fs.restrict(v[0]), v[1])
    v1 = op.mv((fs.prolong(c1), h1))
    r0, r1 = fs.restrict(v[0]), fs.restrict(v1[0])
    np.testing.assert_allclose(r1, r0, rtol=0, atol=1e-10 * float(jnp.max(jnp.abs(r0))))
    np.testing.assert_allclose(
        v1[1], v[1], rtol=0, atol=1e-10 * float(jnp.max(jnp.abs(v[1])))
    )
    np.testing.assert_allclose(c1, c0, rtol=0, atol=1e-5 * float(jnp.max(jnp.abs(c0))))

    # so the residual of the preconditioned operator vanishes on the coarse moments
    x = M.mv(v)
    rx = fs.restrict(op.mv(x)[0])
    np.testing.assert_allclose(rx, r0, rtol=0, atol=1e-8 * float(jnp.max(jnp.abs(r0))))

    r = (jax.random.normal(keys[2], (A.in_size(),)), h0)
    s = (jax.random.normal(keys[3], (A.in_size(),)), h1)

    def dot(x, y):
        return jnp.vdot(jnp.concatenate(x), jnp.concatenate(y))

    np.testing.assert_allclose(dot(M.mv(r), s), dot(r, M.T.mv(s)), rtol=1e-8)


def test_tokamak_preconditioner_errors(field, tokamak, pitchgrid, speedgrid, species2):
    """Fields with nzeta > 1 and bad options are rejected."""

    def operator(field):
        A = DKE(field, pitchgrid, speedgrid, species2, 1e3, gauge=False)
        B = DKESources(field, pitchgrid, speedgrid, species2)
        C = DKEConstraint(field, pitchgrid, speedgrid, species2, True)
        return BorderedOperator(A, B, C)

    with pytest.raises(ValueError, match="nzeta=1"):
        TokamakPreconditioner(operator(field))
    with pytest.raises(ValueError, match="moments"):
        TokamakPreconditioner(operator(tokamak), moments=2)
    with pytest.raises(ValueError, match="preconditioner"):
        solve_dke(
            tokamak,
            pitchgrid,
            speedgrid,
            species2,
            1e3,
            preconditioner="lu",  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValueError, match="nzeta=1"):
        solve_dke(field, pitchgrid, speedgrid, species2, 1e3, preconditioner="tokamak")
    with pytest.raises(ValueError, match="moments"):
        solve_dke(
            tokamak,
            pitchgrid,
            speedgrid,
            species2,
            1e3,
            preconditioner="tokamak",
            tokamak_options={"moments": 9},
        )
    with pytest.raises(ValueError, match="preconditioner"):
        solve_dke_ambipolar(
            tokamak, pitchgrid, speedgrid, species2, 1, preconditioner="tokamak"
        )


@pytest.mark.parametrize("p1", ["4d", "4z"])
def test_solve_dke_tokamak(tokamak, pitchgrid, speedgrid, species2, p1):
    """Solution with the tokamak preconditioner matches a dense solve."""
    if os.environ.get("CI"):
        jax.clear_caches()
    sol, info = solve_dke(
        tokamak,
        pitchgrid,
        speedgrid,
        species2,
        1e3,
        p1=p1,
        rtol=1e-10,
        verbose=2,
        preconditioner="tokamak",
    )
    assert info["success"]
    assert info["nmv"] < 30

    A = DKE(tokamak, pitchgrid, speedgrid, species2, 1e3, p1=p1, gauge=False)
    B = DKESources(tokamak, pitchgrid, speedgrid, species2)
    C = DKEConstraint(tokamak, pitchgrid, speedgrid, species2, True)
    rhs = dke_rhs(tokamak, pitchgrid, speedgrid, species2, 1e3, 0.0)
    rhs = jnp.concatenate([rhs, jnp.zeros(B.in_size())])
    f = jnp.linalg.solve(BorderedOperator(A, B, C).as_matrix(), rhs)[: rhs.size - 4]
    np.testing.assert_allclose(
        sol.f1.reshape(-1), f, rtol=0, atol=1e-7 * float(jnp.max(jnp.abs(f)))
    )

    if p1 == "4d":
        # same fluxes as with the default preconditioner
        sol_mg, _ = solve_dke(
            tokamak, pitchgrid, speedgrid, species2, 1e3, p1=p1, rtol=1e-10
        )
        np.testing.assert_allclose(_fluxes(sol), _fluxes(sol_mg), rtol=1e-6)


def test_solve_dke_tokamak_derivatives(tokamak, pitchgrid, speedgrid):
    """Forward and reverse mode derivatives through the tokamak preconditioner."""
    # these are super low res, just to test jax logic, not physical correctness
    if os.environ.get("CI"):
        jax.clear_caches()

    def foo(inputs):
        n, T, dn, dT, Er = inputs
        species = [yancc.species.LocalMaxwellian(yancc.species.Hydrogen, T, n, dT, dn)]
        sol, info = solve_dke(
            tokamak,
            pitchgrid,
            speedgrid,
            species,
            Er,
            p1="4z",
            rtol=1e-12,
            preconditioner="tokamak",
        )
        return _fluxes(sol)

    inputs = jnp.array([1e19, 1e3, -1e19, -1e3, 1e3])
    Jfd = jnp.array([_jvp_1_arg(foo, inputs, i, 1e-4, 0.0) for i in range(len(inputs))])
    Jf = jax.jacfwd(foo)(inputs)
    Jr = jax.jacrev(foo)(inputs)
    np.testing.assert_allclose(Jr, Jf, rtol=1e-10)
    np.testing.assert_allclose(Jr, Jfd.T, rtol=1e-6)
    np.testing.assert_allclose(Jf, Jfd.T, rtol=1e-6)
