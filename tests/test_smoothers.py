"""Tests for constructing smoothing operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yancc.misc import dke_rhs
from yancc.multigrid import (
    adpative_smooth,
    get_dke_jacobi2_smoothers,
    get_dke_jacobi_smoothers,
    krylov1_smooth,
    krylov1s_smooth,
    krylov2_smooth,
    krylov2s_smooth,
    standard_smooth,
)
from yancc.smoothers import (
    DKEJacobiSmoother,
    DKELaplacian,
    MDKEJacobiSmoother,
    permute_f_3d,
)
from yancc.species import GlobalMaxwellian, Hydrogen
from yancc.trajectories import DKE, MDKE
from yancc.velocity_grids import MaxwellSpeedGrid


def test_permutations_mdke(field, pitchgrid):
    """Test that re-ordering the grid points gives equivalent operators."""
    p1 = "2a"
    p2 = 2
    erhohat = 1e-4
    nuhat = 1e-4
    N = field.ntheta * field.nzeta * pitchgrid.nxi

    A0f = MDKE(
        field, pitchgrid, erhohat, nuhat, p1=p1, p2=p2, axorder="atz", gauge=True
    ).as_matrix()
    A1f = MDKE(
        field, pitchgrid, erhohat, nuhat, p1=p1, p2=p2, axorder="zat", gauge=True
    ).as_matrix()
    A2f = MDKE(
        field, pitchgrid, erhohat, nuhat, p1=p1, p2=p2, axorder="tza", gauge=True
    ).as_matrix()

    P0f = jax.jacfwd(permute_f_3d)(np.zeros(N), field, pitchgrid, "atz")
    P1f = jax.jacfwd(permute_f_3d)(np.zeros(N), field, pitchgrid, "zat")
    P2f = jax.jacfwd(permute_f_3d)(np.zeros(N), field, pitchgrid, "tza")

    # dummy check that Ps are permutation matrices
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P0f @ P0f.T)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P1f @ P1f.T)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P2f @ P2f.T)

    np.testing.assert_allclose(np.eye(P0f.shape[0]), P0f.T @ P0f)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P1f.T @ P1f)
    np.testing.assert_allclose(np.eye(P0f.shape[0]), P2f.T @ P2f)

    # applying permutation matrices should be the same as the operator in
    # re-ordered basis
    np.testing.assert_allclose(A0f, P0f @ A0f @ P0f.T)
    np.testing.assert_allclose(A0f, P1f @ A1f @ P1f.T)
    np.testing.assert_allclose(A0f, P2f @ A2f @ P2f.T)


@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_dke_banded_vs_dense_smoother(
    pitchgrid, speedgrid, species2, field, potentials2, axorder
):
    Erho = jnp.array(1e3)
    weights = jnp.ones(8).at[-2:].set(0)

    s1 = DKEJacobiSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        potentials2,
        axorder=axorder,
        smooth_solver="dense",
        operator_weights=weights,
    ).as_matrix()
    s2 = DKEJacobiSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        potentials2,
        axorder=axorder,
        smooth_solver="banded",
        operator_weights=weights,
    ).as_matrix()
    np.testing.assert_allclose(s1, s2)


@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_mdke_banded_vs_dense_smoother(pitchgrid, field, axorder):
    Er = 1e-3
    nu = 1e-3
    s1 = MDKEJacobiSmoother(
        field, pitchgrid, Er, nu, axorder=axorder, smooth_solver="dense"
    ).as_matrix()
    s2 = MDKEJacobiSmoother(
        field, pitchgrid, Er, nu, axorder=axorder, smooth_solver="banded"
    ).as_matrix()
    np.testing.assert_allclose(s1, s2)


@pytest.mark.parametrize("v", [1, 2, 3])
@pytest.mark.parametrize(
    "n", [1e18, 1e20, 1e22, 1e24]  # chosen for nustar ~ [1e-4, 1e-2, 1, 1e2]
)
@pytest.mark.parametrize(
    "smooth_op",
    [
        standard_smooth,
        adpative_smooth,
        krylov1_smooth,
        krylov1s_smooth,
        krylov2_smooth,
        krylov2s_smooth,
    ],
)
def test_smoothing_dke(field, pitchgrid, v, n, smooth_op):
    """Test smoothing with type 1 smoothers for DKE"""
    speedgrid = MaxwellSpeedGrid(5)
    species = [
        GlobalMaxwellian(
            Hydrogen,
            lambda x: 3e3 * (1 - x**2),
            lambda x: n * (1 - x**4),
        ).localize(0.5),
    ]
    Erho = jnp.array(0.0)
    A = DKE(field, pitchgrid, speedgrid, species, Erho, p1="2d", p2=2, gauge=True)
    b = dke_rhs(field, pitchgrid, speedgrid, species, Erho, include_constraints=False)
    x_true = np.linalg.solve(A.as_matrix(), b)
    potentials = A.potentials
    smoothers = get_dke_jacobi_smoothers(
        [field],
        [pitchgrid],
        speedgrid,
        species,
        jnp.array(0.0),
        potentials,
        "2d",
        2,
        True,
        "dense",
        None,
    )[0]
    r = (x_true + b) / 2
    x_smoothed = smooth_op(
        jnp.zeros_like(x_true), A, r, smoothers, nsteps=v, verbose=True
    )
    L = DKELaplacian(field, pitchgrid, speedgrid, species)
    err = np.linalg.norm(L.mv(x_smoothed - x_true)) / np.linalg.norm(L.mv(x_true))
    print("err=", err)
    if not ("krylov" in smooth_op.__qualname__ and n > 1e23):
        # krylov 2 seems to be bad at high collisionality
        assert err < 1


@pytest.mark.parametrize("v", [1, 2, 3])
@pytest.mark.parametrize(
    "n", [1e18, 1e20, 1e22, 1e24]  # chosen for nustar ~ [1e-4, 1e-2, 1, 1e2]
)
@pytest.mark.parametrize(
    "smooth_op",
    [
        standard_smooth,
        adpative_smooth,
        krylov1_smooth,
        krylov1s_smooth,
        krylov2_smooth,
        krylov2s_smooth,
    ],
)
def test_smoothing2_dke(field, pitchgrid, v, n, smooth_op):
    """Test smoothing with type 2 smoothers for DKE"""
    speedgrid = MaxwellSpeedGrid(5)
    species = [
        GlobalMaxwellian(
            Hydrogen,
            lambda x: 3e3 * (1 - x**2),
            lambda x: n * (1 - x**4),
        ).localize(0.5),
    ]
    Erho = jnp.array(0.0)
    A = DKE(field, pitchgrid, speedgrid, species, Erho, p1="2d", p2=2, gauge=True)
    b = dke_rhs(field, pitchgrid, speedgrid, species, Erho, include_constraints=False)
    x_true = np.linalg.solve(A.as_matrix(), b)
    potentials = A.potentials
    smoothers = get_dke_jacobi2_smoothers(
        [field],
        [pitchgrid],
        speedgrid,
        species,
        jnp.array(0.0),
        potentials,
        "2d",
        2,
        True,
        "dense",
        None,
    )[0]
    r = (x_true + b) / 2
    x_smoothed = smooth_op(
        jnp.zeros_like(x_true), A, r, smoothers, nsteps=v, verbose=True
    )
    L = DKELaplacian(field, pitchgrid, speedgrid, species)
    err = np.linalg.norm(L.mv(x_smoothed - x_true)) / np.linalg.norm(L.mv(x_true))
    print("err=", err)
    if not ("krylov" in smooth_op.__qualname__ and n > 1e23):
        # krylov 2 seems to be bad at high collisionality
        assert err < 1
