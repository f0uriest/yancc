"""Tests for constructing smoothing operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yancc.smoothers import DKEJacobiSmoother, MDKEJacobiSmoother, permute_f_3d
from yancc.trajectories import MDKE


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
        potentials=potentials2,
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
        potentials=potentials2,
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
