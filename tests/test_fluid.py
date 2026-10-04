"""Tests for the Galerkin fluid correction of the DKE preconditioner."""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import yancc._fluid as _fluid
from yancc._linalg import BorderedOperator
from yancc._misc import DKEConstraint, DKESources
from yancc._trajectories import DKE
from yancc.field import Field


@pytest.fixture(scope="module")
def fluid_field():
    """Field with theta lines shorter and zeta lines longer than the DKE stencil."""
    nt, nz, NFP = 5, 13, 5
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)[:, None]
    zeta = np.linspace(0, 2 * np.pi / NFP, nz, endpoint=False)[None, :]
    Bmag = (
        2.50302
        + 0.25660 * np.cos(NFP * zeta)
        - 0.11118 * np.cos(theta - NFP * zeta)
        - 0.05205 * np.cos(theta)
    )
    return Field.from_boozer(
        rho=0.5,
        Bmag=Bmag,
        I=0.0,
        G=14.4,
        iota=-0.88356,
        Psi=-2.0004,
        R_major=5.4832,
        a_minor=0.5211,
        NFP=NFP,
    )


def test_fluid_correction(fluid_field, pitchgrid, speedgrid, species2, monkeypatch):
    """Fluid solve inverts R A P, CR matches dense LU, transposes are consistent."""
    # theta lines shorter than the stencil use dense line blocks
    field = fluid_field
    A = DKE(field, pitchgrid, speedgrid, species2, 1e3, gauge=False)
    B = DKESources(field, pitchgrid, speedgrid, species2)
    C = DKEConstraint(field, pitchgrid, speedgrid, species2, True)
    op = BorderedOperator(A, B, C)
    ns, n = len(species2), A.in_size()
    fs = _fluid._FluidSolver(A, B, C, species2, speedgrid, pitchgrid, field)
    assert not fs.dense  # 13 zeta planes give 3 periodic block tridiagonal groups

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    c0 = jax.random.normal(keys[0], (ns, 3, field.ntheta, field.nzeta))
    h0 = jax.random.normal(keys[1], (2 * ns,))
    v = op.mv(jnp.concatenate([fs.prolong(c0), h0]))
    c1, h1 = fs.solve(fs.restrict(v[:n]), v[n:])
    np.testing.assert_allclose(c1, c0, rtol=0, atol=1e-8 * float(jnp.max(jnp.abs(c0))))
    # the sources are only weakly determined by the bordered fluid system
    np.testing.assert_allclose(h1, h0, rtol=0, atol=1e-4 * float(jnp.max(jnp.abs(h0))))

    a = (jax.random.normal(keys[2], c0.shape), jax.random.normal(keys[3], h0.shape))
    xa = fs.solve(*a)

    monkeypatch.setattr(_fluid, "_plane_groups", lambda nz, _: np.arange(nz)[None])
    fsd = _fluid._FluidSolver(A, B, C, species2, speedgrid, pitchgrid, field)
    assert fsd.dense
    cd, hd = fsd.solve(*a)
    np.testing.assert_allclose(
        cd, xa[0], rtol=0, atol=1e-8 * float(jnp.max(jnp.abs(cd)))
    )
    np.testing.assert_allclose(
        hd, xa[1], rtol=0, atol=1e-4 * float(jnp.max(jnp.abs(hd)))
    )

    # corrected preconditioner around a stand in for the multigrid preconditioner
    M = lx.DiagonalLinearOperator(1 / jnp.abs(jnp.concatenate([A.diagonal(), h0])))
    eye = lx.IdentityLinearOperator(op.in_structure())

    def corrected(F):
        return F + M @ (eye - op @ F)

    r = jax.random.normal(keys[0], (n + 2 * ns,))
    s = jax.random.normal(keys[1], (n + 2 * ns,))
    w = jnp.array([0.3, 1.0])
    P = corrected(_fluid.FluidOperator(op, species2, speedgrid, pitchgrid, field, w))
    np.testing.assert_allclose(jnp.vdot(P.mv(r), s), jnp.vdot(r, P.T.mv(s)), rtol=1e-8)

    # weights only known at run time, all zero turns the correction off
    @jax.jit
    def apply(w, r):
        F = _fluid.FluidOperator(op, species2, speedgrid, pitchgrid, field, w)
        return corrected(F).mv(r)

    np.testing.assert_allclose(apply(jnp.zeros(2), r), M.mv(r), rtol=1e-12)
    np.testing.assert_allclose(apply(jnp.array([0.3, 1.0]), r), P.mv(r), rtol=1e-8)

    w = _fluid.fluid_weights(species2, field)
    assert jnp.all(((w >= _fluid._MIN_WEIGHT) & (w < 1)) | (w == 0))
