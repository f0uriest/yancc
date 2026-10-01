"""Tests for constructing smoothing operators."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yancc._misc import dke_rhs
from yancc._multigrid import (
    _DKE_SMOOTHER_TOKENS,
    _MDKE_SMOOTHER_TOKENS,
    _parse_smooth_type,
    adpative_smooth,
    get_dke_smoothers,
    get_mdke_smoothers,
    krylov1_smooth,
    krylov1s_smooth,
    krylov2_smooth,
    krylov2s_smooth,
    standard_smooth,
)
from yancc._smoothers import (
    DKEFrozenPlaneSmoother,
    DKEJacobiSmoother,
    DKEL01LineSmoother,
    DKELaplacian,
    MDKEFrozenPlaneSmoother,
    MDKEJacobiSmoother,
    MDKEL01LineSmoother,
    optimal_smoothing_parameter_3d,
    optimal_smoothing_parameter_4d,
    permute_f_3d,
)
from yancc._trajectories import DKE, MDKE
from yancc.field import Field
from yancc.species import Electron, GlobalMaxwellian, Hydrogen, LocalMaxwellian
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


def test_permutations_mdke(field, pitchgrid):
    """The smoothers' reordering of f matches the layout of the operator blocks."""
    op = MDKE(field, pitchgrid, 1e-4, 1e-4, p1="2a", p2=2, gauge=True)
    A = np.asarray(op.as_matrix())
    N = A.shape[0]
    sizes = {"a": pitchgrid.nalpha, "t": field.ntheta, "z": field.nzeta}
    for axorder in ["atz", "zat", "tza"]:
        # maps f in the axorder layout to the canonical (a, t, z) layout
        P = np.asarray(jax.jacfwd(permute_f_3d)(np.zeros(N), field, pitchgrid, axorder))
        np.testing.assert_allclose(P @ P.T, np.eye(N))
        m = sizes[axorder[-1]]
        Ap = P.T @ A @ P
        blocks = [Ap[k : k + m, k : k + m] for k in range(0, N, m)]
        np.testing.assert_allclose(op.block_diagonal(axorder=axorder), blocks)


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
    b = dke_rhs(field, pitchgrid, speedgrid, species, Erho)
    x_true = np.linalg.solve(A.as_matrix(), b)
    potentials = A.potentials
    smoothers = get_dke_smoothers(
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
        "z,t,a,x,s",
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


def _assert_arrays_close(a, b):
    """Assert two pytrees have the same structure and matching floating point leaves."""
    jax.tree_util.tree_map(
        np.testing.assert_allclose,
        eqx.filter(a, eqx.is_inexact_array),
        eqx.filter(b, eqx.is_inexact_array),
    )


def test_get_smoothers_order_and_shared_operator(
    field, pitchgrid, speedgrid, species2, potentials2
):
    """Smoothers come in smooth_type order, and sharing work between them is exact."""
    ow = jnp.ones(8).at[-2:].set(0)
    dke_args = (
        [field],
        [pitchgrid],
        speedgrid,
        species2,
        jnp.array(1e3),
        [],
        potentials2,
        "2d",
        2,
        True,
        "l01t,plane,x,l01z",
        None,
        {"l01t": 0.5, "x": 0.3},
    )
    (group,) = get_dke_smoothers(*dke_args, operator_weights=ow)
    assert [type(op) for op in group] == [
        DKEL01LineSmoother,
        DKEFrozenPlaneSmoother,
        DKEJacobiSmoother,
        DKEL01LineSmoother,
    ]
    assert group[2].axorder == "atzsx"
    assert [group[0].line, group[3].line] == ["t", "z"]
    # each smoother built on its own, with no shared operator
    kw: dict[str, Any] = dict(
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species2,
        Erho=jnp.array(1e3),
        background=[],
        potentials=potentials2,
        p1="2d",
        p2=2,
        gauge=True,
        operator_weights=ow,
    )
    direct = [
        DKEL01LineSmoother(**kw, line="t", weight=0.5),
        DKEFrozenPlaneSmoother(**kw),
        DKEJacobiSmoother(**kw, axorder="atzsx", weight=0.3),
        DKEL01LineSmoother(**kw, line="z"),
    ]
    _assert_arrays_close(direct, group)

    # a shared level operator gives the same smoothers
    op = DKE(
        field,
        pitchgrid,
        speedgrid,
        species2,
        jnp.array(1e3),
        potentials=potentials2,
        p1="2d",
        p2=2,
        gauge=True,
        operator_weights=ow,
    )
    shared = get_dke_smoothers(*dke_args, operator_weights=ow, operators=[op])
    _assert_arrays_close([group], shared)

    mdke_args = (
        [field],
        [pitchgrid],
        1e-3,
        1e-2,
        "2d",
        2,
        True,
        "a,t,z,plane,l01t,l01z",
    )
    own = get_mdke_smoothers(*mdke_args, None, None)
    with pytest.raises(ValueError, match="not in smooth_type"):
        get_mdke_smoothers(*mdke_args, None, {"plane": 0.5, "s": 0.5})
    # a single weight applies to every smoother
    (group,) = get_mdke_smoothers(*mdke_args, None, 0.5)
    for sm in group:
        np.testing.assert_allclose(sm.weight, 0.5)
    op = MDKE(field, pitchgrid, 1e-3, 1e-2, "2d", 2, True)
    shared = get_mdke_smoothers(*mdke_args, None, None, operators=[op])
    assert [type(sm) for sm in shared[0]] == [
        MDKEJacobiSmoother,
        MDKEJacobiSmoother,
        MDKEJacobiSmoother,
        MDKEFrozenPlaneSmoother,
        MDKEL01LineSmoother,
        MDKEL01LineSmoother,
    ]
    _assert_arrays_close(own, shared)


def test_l01_line_smoother_block(field, pitchgrid, speedgrid, species2, potentials2):
    """L01 line blocks match the projection of the dense DKE/MDKE block diagonals."""
    ns, nx = len(species2), speedgrid.nx
    na, nt, nz = pitchgrid.nalpha, field.ntheta, field.nzeta
    Erho = jnp.array(1e3)
    for line in "tz":
        M = DKEL01LineSmoother(
            field, pitchgrid, speedgrid, species2, Erho, None, potentials2, line=line
        )
        n, nother = (nt, nz) if line == "t" else (nz, nt)

        op = DKE(
            field,
            pitchgrid,
            speedgrid,
            species2,
            Erho,
            potentials=potentials2,
            p1="2d",
            p2=2,
            gauge=True,
            operator_weights=jnp.ones(8).at[-1].set(0),
        )

        W, Q = M._W, M._Q
        D = op.block_diagonal("dense", axorder="sxzat" if line == "t" else "sxtaz")
        D = D.reshape(ns, nx, nother, na, n, n)
        t1 = jnp.einsum("la,sxoaij,am->sxolmij", W, D, Q)
        Aa = op.block_diagonal("dense", axorder="sxtza")
        Aa = Aa.reshape(ns, nx, nt, nz, na, na)
        Aa = Aa - Aa * jnp.eye(na)
        k2 = jnp.einsum("la,sxtzab,bm->sxtzlm", W, Aa, Q)
        if line == "t":
            k2 = jnp.moveaxis(k2, 2, 3)
        B = jnp.transpose(t1, (0, 1, 2, 5, 3, 6, 4)) + jnp.einsum(
            "sxoilm,ij->sxoiljm", k2, jnp.eye(n)
        )
        B = B.reshape(ns * nx * nother, 2 * n, 2 * n)
        np.testing.assert_allclose(
            jnp.linalg.inv(M._inv), B, atol=1e-10 * float(jnp.abs(B).max())
        )

        # monoenergetic analog, same construction without species and speed
        Mm = MDKEL01LineSmoother(field, pitchgrid, 1e-3, 1e-2, line=line)

        mop = MDKE(field, pitchgrid, 1e-3, 1e-2, "2d", 2, True)
        D = mop.block_diagonal("dense", axorder="zat" if line == "t" else "taz")
        t1 = jnp.einsum("la,oaij,am->olmij", W, D.reshape(nother, na, n, n), Q)
        Aa = mop.block_diagonal("dense", axorder="tza").reshape(nt, nz, na, na)
        Aa = Aa - Aa * jnp.eye(na)
        k2 = jnp.einsum("la,tzab,bm->tzlm", W, Aa, Q)
        if line == "t":
            k2 = jnp.moveaxis(k2, 0, 1)
        B = jnp.transpose(t1, (0, 3, 1, 4, 2)) + jnp.einsum(
            "oilm,ij->oiljm", k2, jnp.eye(n)
        )
        B = B.reshape(nother, 2 * n, 2 * n)
        np.testing.assert_allclose(
            jnp.linalg.inv(Mm._inv), B, atol=1e-10 * float(jnp.abs(B).max())
        )


def test_parse_smooth_type():
    assert _parse_smooth_type(" plane, x,l01t ", _DKE_SMOOTHER_TOKENS) == [
        "plane",
        "x",
        "l01t",
    ]
    assert _parse_smooth_type("plane,a,l01z", _MDKE_SMOOTHER_TOKENS) == [
        "plane",
        "a",
        "l01z",
    ]
    with pytest.raises(ValueError, match="unknown"):
        _parse_smooth_type("plane,x", _MDKE_SMOOTHER_TOKENS)
    with pytest.raises(ValueError, match="empty"):
        _parse_smooth_type("plane,,x", _DKE_SMOOTHER_TOKENS)


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


# ---------------------------------------------------------------------------
# frozen (theta, zeta)-plane FFT smoothers (DKEFrozenPlaneSmoother /
# MDKEFrozenPlaneSmoother)
# ---------------------------------------------------------------------------


def _constant_boozer_field(nt, nz):
    """A Boozer-coordinate field with constant |B| over the surface.

    Not physically real, just used for testing the frozen plane approximation.
    """
    Bmag = jnp.ones((nt, nz))
    return Field.from_boozer(rho=0.5, Bmag=Bmag, I=0.1, G=1.0, iota=0.9, Psi=1.0, NFP=1)


def test_frozen_plane_protocol_mdke(field, pitchgrid):
    """out/in structure, transpose, and as_matrix agree (FFT-based mv transposes)."""
    op = MDKEFrozenPlaneSmoother(field, pitchgrid, 1e-3, 1e-3)
    _check_protocol(op)


def test_frozen_plane_protocol_dke(field, pitchgrid, speedgrid, species2, potentials2):
    op = DKEFrozenPlaneSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        jnp.array(1e3),
        background=[],  # explicit (non-None) background -> skips the default branch
        potentials=potentials2,
        operator_weights=jnp.ones(8).at[-2:].set(0),
    )
    _check_protocol(op)


def test_frozen_plane_weight_linear_mdke(field, pitchgrid):
    """Matvec is linear in the (scalar) under-relaxation weight; default is 0.7."""
    s1 = MDKEFrozenPlaneSmoother(field, pitchgrid, 1e-3, 1e-3, weight=jnp.array(1.0))
    sw = MDKEFrozenPlaneSmoother(field, pitchgrid, 1e-3, 1e-3, weight=jnp.array(0.3))
    x = jnp.asarray(np.random.default_rng(0).standard_normal(s1.in_size()))
    np.testing.assert_allclose(
        np.asarray(sw.mv(x)), 0.3 * np.asarray(s1.mv(x)), rtol=1e-10, atol=1e-12
    )
    # default-weight branch (weight=None -> 0.7)
    assert float(MDKEFrozenPlaneSmoother(field, pitchgrid, 1e-3, 1e-3).weight) == 0.7


def test_frozen_plane_dke_default_args(
    field, pitchgrid, speedgrid, species2, potentials2
):
    """background=None, operator_weights=None, weight=None default branches."""
    s = DKEFrozenPlaneSmoother(
        field,
        pitchgrid,
        speedgrid,
        species2,
        jnp.array(1e3),
        potentials=potentials2,
        # background / operator_weights / weight all omitted -> default branches
    )
    assert float(s.weight) == 0.7
    mat = s.as_matrix()
    assert mat.shape[0] == mat.shape[1] == s.in_size()
    assert np.all(np.isfinite(mat))


def test_frozen_plane_mdke_matches_dense_block(pitchgrid):
    """On a constant-|B| field the winds are plane-constant so the frozen
    approximation is exact, and each pitch block of the smoother must be
    weight * inv of the true (theta, zeta) sub-block of the MDKE. Also confirms the
    smoother is block-diagonal in pitch and that the plane block has genuine
    off-diagonal (streaming) coupling for the FFT solve to invert.
    """
    nt, nz = 5, 7
    cfield = _constant_boozer_field(nt, nz)
    erhohat, nuhat, weight = 1e-3, 1e-2, 0.6
    sm = MDKEFrozenPlaneSmoother(
        cfield, pitchgrid, erhohat, nuhat, gauge=False, weight=jnp.array(weight)
    )
    # use the public MDKE.as_matrix (catches a change to the private op attributes
    # the smoother reads).
    A = np.asarray(MDKE(cfield, pitchgrid, erhohat, nuhat, "2d", 2, False).as_matrix())
    M = np.asarray(sm.as_matrix())

    na, npl = pitchgrid.nalpha, nt * nz
    eye = np.eye(npl)
    for a in range(na):
        idx = slice(a * npl, (a + 1) * npl)
        plane = A[idx, idx]
        # streaming actually couples the plane (not a trivially-diagonal block)
        assert np.abs(plane - np.diag(np.diag(plane))).max() > 0
        # off-pitch-block coupling is exactly zero (block diagonal in pitch)
        off = M[idx].copy()
        off[:, idx] = 0.0
        np.testing.assert_allclose(off, 0.0, atol=1e-12)
        # the block inverts its (exact) frozen operator, up to the weight
        np.testing.assert_allclose(plane @ M[idx, idx], weight * eye, atol=1e-8)


def test_frozen_plane_dke_matches_dense_block(speedgrid, species2, potentials2):
    """DKE analog of the block-inverse check, on a constant-|B| Boozer field with 2
    species (exercises the sxatz layout / per-(s, x, a) plane blocks).
    """
    nt, nz = 5, 5
    cfield = _constant_boozer_field(nt, nz)
    pg = UniformPitchAngleGrid(7)
    Erho = jnp.array(1e3)
    ow = jnp.ones(8).at[-1].set(0)
    weight = 0.6
    sm = DKEFrozenPlaneSmoother(
        cfield,
        pg,
        speedgrid,
        species2,
        Erho,
        potentials=potentials2,
        operator_weights=ow,
        gauge=False,
        weight=jnp.array(weight),
    )
    # Dense reference built from the public ``DKE.as_matrix``. (catches a change to
    # the private op attributes the smoother reads).
    A = np.asarray(
        DKE(
            cfield,
            pg,
            speedgrid,
            species2,
            Erho,
            background=[],
            potentials=potentials2,
            p1="2d",
            p2=2,
            gauge=False,
            operator_weights=ow,
        ).as_matrix()
    )
    M = np.asarray(sm.as_matrix())

    ns, nx, na = len(species2), speedgrid.nx, pg.nalpha
    npl = nt * nz
    eye = np.eye(npl)
    nblock = ns * nx * na
    saw_offdiag = False
    for b in range(nblock):
        idx = slice(b * npl, (b + 1) * npl)
        plane = A[idx, idx]
        saw_offdiag |= np.abs(plane - np.diag(np.diag(plane))).max() > 0
        off = M[idx].copy()
        off[:, idx] = 0.0
        np.testing.assert_allclose(off, 0.0, atol=1e-12)
        np.testing.assert_allclose(plane @ M[idx, idx], weight * eye, atol=1e-8)
    assert saw_offdiag  # at least some blocks have real plane coupling to invert
