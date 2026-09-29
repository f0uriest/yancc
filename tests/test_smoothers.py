"""Tests for constructing smoothing operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yancc._misc import dke_rhs
from yancc._multigrid import (
    adpative_smooth,
    get_dke_jacobi_smoothers,
    krylov1_smooth,
    krylov1s_smooth,
    krylov2_smooth,
    krylov2s_smooth,
    standard_smooth,
)
from yancc._smoothers import (
    DKEJacobiSmoother,
    DKELaplacian,
    MDKEJacobiSmoother,
    optimal_smoothing_parameter_3d,
    optimal_smoothing_parameter_4d,
    permute_f_3d,
)
from yancc._trajectories import DKE, MDKE
from yancc.field import Field
from yancc.species import Electron, GlobalMaxwellian, Hydrogen, LocalMaxwellian
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def test_permutations_mdke(field, pitchgrid):
    """Test that re-ordering the grid points gives equivalent operators."""
    p1 = "2a"
    p2 = 2
    erhohat = 1e-4
    nuhat = 1e-4
    N = field.ntheta * field.nzeta * pitchgrid.nalpha

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
    s3 = DKEJacobiSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        potentials=potentials2,
        axorder=axorder,
        smooth_solver="cr",
        operator_weights=weights,
    ).as_matrix()
    np.testing.assert_allclose(s1, s2)
    np.testing.assert_allclose(s1, s3)


@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_mdke_banded_vs_dense_smoother(pitchgrid, field, axorder):
    erhohat = 1e-3
    nuhat = 1e-5
    s1 = MDKEJacobiSmoother(
        field, pitchgrid, erhohat, nuhat, axorder=axorder, smooth_solver="dense"
    ).as_matrix()
    s2 = MDKEJacobiSmoother(
        field, pitchgrid, erhohat, nuhat, axorder=axorder, smooth_solver="banded"
    ).as_matrix()
    s3 = MDKEJacobiSmoother(
        field, pitchgrid, erhohat, nuhat, axorder=axorder, smooth_solver="cr"
    ).as_matrix()
    np.testing.assert_allclose(s1, s2)
    np.testing.assert_allclose(s1, s3)


@pytest.mark.parametrize("v", [1, 2, 3])
@pytest.mark.parametrize("n", [1e18, 1e20, 1e22])  # chosen for nustar ~ [1e-4, 1e-2, 1]
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
    operator_weights = jnp.ones(8).at[-2:].set(0)
    A = DKE(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho,
        p1="2d",
        p2=2,
        gauge=True,
        operator_weights=operator_weights,
    )
    b = dke_rhs(field, pitchgrid, speedgrid, species, Erho, include_constraints=False)
    x_true = np.linalg.solve(A.as_matrix(), b)
    potentials = A.potentials
    smoothers = get_dke_jacobi_smoothers(
        [field],
        [pitchgrid],
        speedgrid,
        species,
        jnp.array(0.0),
        [],
        potentials,
        "2d",
        2,
        True,
        "dense",
        None,
        operator_weights=operator_weights,
    )[0]
    r = (x_true + b) / 2
    x_smoothed, _ = smooth_op(
        jnp.zeros_like(x_true), A, r, smoothers, nsteps=v, verbose=True
    )
    L = DKELaplacian(field, pitchgrid, speedgrid, species)
    err = np.linalg.norm(L.mv(x_smoothed - x_true)) / np.linalg.norm(L.mv(x_true))
    print("err=", err)
    assert err < 1


# ---------------------------------------------------------------------------
# operator protocol sweep: out_structure / in_structure / transpose / as_matrix
# ---------------------------------------------------------------------------


def _check_protocol(op):
    """in/out structures agree (square) and transpose matches matrix transpose."""
    assert op.out_structure() == op.in_structure()
    opT = op.transpose()
    assert opT.in_structure() == op.out_structure()
    assert opT.out_structure() == op.in_structure()
    M = op.as_matrix()
    # TransposedLinearOperator.as_matrix is defined as operator.as_matrix().T
    np.testing.assert_allclose(opT.as_matrix(), M.T)
    # and the transpose action (via jax.linear_transpose) matches M.T @ v
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal(M.shape[0]))
    np.testing.assert_allclose(opT.mv(v), M.T @ v, atol=1e-8, rtol=1e-6)


def test_smoother_protocol_mdke(field, pitchgrid):
    op = MDKEJacobiSmoother(field, pitchgrid, 1e-3, 1e-3, smooth_solver="dense")
    _check_protocol(op)


def test_smoother_protocol_dke_jacobi(
    field, pitchgrid, speedgrid, species2, potentials2
):
    op = DKEJacobiSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        jnp.array(1e3),
        potentials=potentials2,
        axorder="atzsx",
        smooth_solver="dense",
        operator_weights=jnp.ones(8).at[-2:].set(0),
    )
    _check_protocol(op)


def test_dke_jacobi_banded_default_operator_weights(
    field, pitchgrid, speedgrid, species2, potentials2
):
    """A banded smoother with default (None) operator_weights also zeros slot -2."""
    op = DKEJacobiSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        jnp.array(1e3),
        potentials=potentials2,
        axorder="atzsx",
        smooth_solver="banded",
        operator_weights=None,
    )
    assert op.smooth_solver == "banded"
    _check_protocol(op)


@pytest.mark.parametrize("normalize", [True, False])
def test_dke_laplacian_protocol(field, pitchgrid, speedgrid, species2, normalize):
    op = DKELaplacian(field, pitchgrid, speedgrid, species2, normalize=normalize)
    _check_protocol(op)


# ---------------------------------------------------------------------------
# optimal_smoothing_parameter fallbacks (unknown stencil / axis -> warn + default)
# ---------------------------------------------------------------------------


def test_optimal_smoothing_parameter_3d_unknown_stencil():
    with pytest.warns(UserWarning, match="stencil"):
        w = optimal_smoothing_parameter_3d("not_a_stencil", 2, 1e-3, "a")
    np.testing.assert_allclose(float(w), 0.1)


def test_optimal_smoothing_parameter_3d_unknown_axis():
    with pytest.warns(UserWarning, match="ax="):
        w = optimal_smoothing_parameter_3d("1a", 2, 1e-3, "q")
    np.testing.assert_allclose(float(w), 0.1)


def test_optimal_smoothing_parameter_4d_unknown_stencil():
    with pytest.warns(UserWarning, match="stencil"):
        w = optimal_smoothing_parameter_4d("not_a_stencil", 2, 1e-3, "a")
    np.testing.assert_allclose(float(w), 0.01)


def test_optimal_smoothing_parameter_4d_unknown_axis():
    with pytest.warns(UserWarning, match="ax="):
        w = optimal_smoothing_parameter_4d("2d", 2, 1e-3, "q")
    np.testing.assert_allclose(float(w), 0.01)


# ---------------------------------------------------------------------------
# smoother constructor default-argument branches
# ---------------------------------------------------------------------------


def test_dke_jacobi_smoother_default_operator_weights_explicit_weight(
    pitchgrid, speedgrid, species2, field, potentials2
):
    """operator_weights=None default branch + explicit (scalar) weight branch."""
    Erho = jnp.array(1e3)
    s = DKEJacobiSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        potentials=potentials2,
        axorder="atzsx",
        smooth_solver="dense",
        weight=jnp.array(0.5),  # exercises the `else: _weight = weight` branch
        # operator_weights omitted -> None -> default-weights branch
    )
    mat = s.as_matrix()
    assert mat.shape[0] == mat.shape[1]
    assert np.all(np.isfinite(mat))


# convolved axis last: "a" (pitch, non-periodic), "t"/"z" (periodic lines).
# The cyclic-reduction solver must reproduce the banded solver exactly (same factor,
# different log-depth elimination).
@pytest.mark.parametrize("axorder", ["tzsxa", "azsxt", "atsxz"])
def test_dke_cr_matches_banded(axorder):
    field = Field.from_vmec("tests/data/wout_NCSX.nc", 0.5, 11, 11)
    am = float(field.a_minor)
    pg = UniformPitchAngleGrid(25)
    sg = MaxwellSpeedGrid(4)
    n = 4.09e21
    species = [
        LocalMaxwellian(Electron, 3.0e3, n, -2e3 * am, -0.4e20 * am),
        LocalMaxwellian(Hydrogen, 3.0e3, n, -2e3 * am, -0.4e20 * am),
    ]
    Erho = 4.0 * am * 1000.0
    banded = DKEJacobiSmoother(
        field,
        pg,
        sg,
        species,
        Erho,
        axorder=axorder,
        smooth_solver="banded",
        coulomb_log=17.0,
    )
    cr = DKEJacobiSmoother(
        field,
        pg,
        sg,
        species,
        Erho,
        axorder=axorder,
        smooth_solver="cr",
        coulomb_log=17.0,
    )

    n_state = pg.nalpha * field.ntheta * field.nzeta * len(species) * sg.nx
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = jnp.asarray(rng.standard_normal(n_state))
        np.testing.assert_allclose(
            np.asarray(cr.mv(x)), np.asarray(banded.mv(x)), rtol=1e-7, atol=1e-9
        )


@pytest.mark.parametrize("axorder", ["atz", "tza", "zat"])
def test_mdke_cr_matches_banded(axorder):
    field = Field.from_vmec("tests/data/wout_NCSX.nc", 0.5, 11, 11)
    pg = UniformPitchAngleGrid(25)
    banded = MDKEJacobiSmoother(
        field, pg, 1e-3, 1e-3, axorder=axorder, smooth_solver="banded"
    )
    cr = MDKEJacobiSmoother(field, pg, 1e-3, 1e-3, axorder=axorder, smooth_solver="cr")
    n_state = pg.nalpha * field.ntheta * field.nzeta
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = jnp.asarray(rng.standard_normal(n_state))
        np.testing.assert_allclose(
            np.asarray(cr.mv(x)), np.asarray(banded.mv(x)), rtol=1e-7, atol=1e-9
        )
