"""Tests for FGMRES/GCROT(m,k) solver."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy

from yancc.krylov import _fgmres, gcrotmk, lgmres
from yancc.linalg import InverseLinearOperator


def crop2(a, b):
    """Crop a,b to the same shape."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.ndim == b.ndim
    for i in range(a.ndim):
        if a.shape[i] < b.shape[i]:
            b = np.take(b, np.arange(a.shape[i]), axis=i)
        if b.shape[i] < a.shape[i]:
            a = np.take(a, np.arange(b.shape[i]), axis=i)
    return a, b


def _scipy_fgmres(A_mv, b, m, k, atol, cs=(), **kw):
    """Run scipy's internal _fgmres and return (H, B, V, Z, y)."""
    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A_mv, b, m=m + k - len(cs), atol=atol, cs=cs, **kw
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T
    return H1, B1, V1, Z1, y1


@pytest.mark.parametrize("gs_method", ["cgs", "cgs2", "mgs"])
def test_fgmres_gs_methods(gs_method):
    """All three Gram-Schmidt variants should produce same Krylov basis."""
    n = 20
    rng = np.random.default_rng(0)
    A = rng.random((n, n)) + n * np.eye(n)  # diagonally-dominant for stability
    A = lx.MatrixLinearOperator(jnp.array(A))
    b = rng.random(n)
    b /= np.linalg.norm(b)

    m, k = 10, 5
    _, _, _, Z, y, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=0.0, C=None, lc=None, gs_method=gs_method
    )
    # Z = V when no right preconditioner; reconstruct x and verify residual.
    x = Z @ y
    np.testing.assert_allclose(A.mv(x), b, atol=1e-8)


def test_fgmres_base():
    """FGMRES without C or preconditioner matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 7, 5, 0

    H1, B1, V1, Z1, y1 = _scipy_fgmres(A.mv, b, m=m, k=k, atol=atol)
    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=None, lc=None
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)


def test_fgmres_with_c():
    """FGMRES with a full C augmentation matrix matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 7, 5, 0
    C = np.linalg.qr(rng.random((3, n)).T)[0].T
    lc = 3

    H1, B1, V1, Z1, y1 = _scipy_fgmres(A.mv, b, m=m, k=k, atol=atol, cs=C)
    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C.T, lc=lc
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)


def test_fgmres_right_preconditioner():
    """FGMRES with right preconditioner and C augmentation matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 7, 5, 0
    C = np.linalg.qr(rng.random((3, n)).T)[0].T
    lc = 3

    H1, B1, V1, Z1, y1 = _scipy_fgmres(A.mv, b, m=m, k=k, atol=atol, cs=C, rpsolve=M.mv)
    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C.T, lc=lc, rpsolve=M.mv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)


def test_fgmres_left_preconditioner():
    """FGMRES with left preconditioner and C augmentation matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 7, 5, 0
    C = np.linalg.qr(rng.random((3, n)).T)[0].T
    lc = 3

    H1, B1, V1, Z1, y1 = _scipy_fgmres(A.mv, b, m=m, k=k, atol=atol, cs=C, lpsolve=M.mv)
    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C.T, lc=lc, lpsolve=M.mv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)


def test_fgmres_partial_c():
    """FGMRES with a partially-filled C (lc < shape) matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 7, 5, 0
    C = np.linalg.qr(rng.random((3, n)).T)[0].T
    C[-1] = 0
    lc = 2

    H1, B1, V1, Z1, y1 = _scipy_fgmres(
        A.mv, b, m=m, k=k, atol=atol, cs=C[:-1], rpsolve=M.mv
    )
    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C.T, lc=lc, rpsolve=M.mv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)


def test_fgmres_early_stop():
    """FGMRES stops early when atol is met and the reconstructed x satisfies it."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 15, 5, 0.2

    H1, B1, V1, Z1, y1 = _scipy_fgmres(A.mv, b, m=m, k=k, atol=atol, rpsolve=M.mv)
    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=None, lc=None, rpsolve=M.mv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)

    u = Z2 @ y2
    c = V2 @ H2.T @ y2
    a = np.linalg.norm(c)
    u /= a
    c /= a
    x = np.dot(c, b) * u
    assert np.linalg.norm(A.mv(x) - b) < atol


def test_fgmres_full_space_convergence():
    """FGMRES with m >= n converges to machine precision within atol."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 70, 0, 1e-10

    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=None, lc=None, rpsolve=M.mv
    )

    u = Z2 @ y2
    c = V2 @ H2.T @ y2
    a = np.linalg.norm(c)
    u /= a
    c /= a
    x = np.dot(c, b) * u
    assert np.linalg.norm(A.mv(x) - b) < atol


def test_fgmres_outer_v():
    """FGMRES with LGMRES-style outer_v augmentation matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 7, 5, 0
    outer_v = rng.random((3, n))
    outer_Av = jax.vmap(A.mv, in_axes=0, out_axes=0)(outer_v)
    lv = 3

    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A.mv,
        b,
        m=m + k,
        atol=atol,
        outer_v=list((v, Av) for v, Av in zip(outer_v, outer_Av)),
        prepend_outer_v=True,
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T

    H2, B2, V2, Z2, y2, _, _, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, outer_v=outer_v.T, outer_Av=outer_Av.T, lv=lv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)


def test_fgmres_nonflexible_base():
    """Non-flexible _fgmres matches flexible for a linear preconditioner."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k, atol = 7, 5, 0

    H1, B1, V1, Z1, y1, j1, *_ = _fgmres(A.mv, b, m=m, k=k, atol=atol, rpsolve=M.mv)
    H2, B2, V2, Z2, y2, j2, *_ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, rpsolve=M.mv, flexible=False
    )

    np.testing.assert_allclose(H1, H2, rtol=1e-10)
    np.testing.assert_allclose(B1, B2, rtol=1e-10)
    np.testing.assert_allclose(V1, V2, rtol=1e-10)
    np.testing.assert_allclose(y1, y2, rtol=1e-10)
    assert int(j1) == int(j2)
    # Z is just a placeholder when flexible=False
    assert Z2.shape == ()

    # Reconstructed dx (M_R(V @ y_masked)) should equal Z1 @ y1. Mask y past
    # j_final because V has one more filled column than Z.
    y_masked = np.where(np.arange(y2.shape[0]) < int(j2), np.asarray(y2), 0.0)
    dx_flex = np.asarray(Z1) @ np.asarray(y1)
    dx_nonflex = np.asarray(M.mv(np.asarray(V2[..., :-1]) @ y_masked))
    np.testing.assert_allclose(dx_flex, dx_nonflex, rtol=1e-10)


def test_fgmres_nonflexible_augmented():
    """Non-flexible _fgmres matches flexible with LGMRES-style outer_v augmentation."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, atol = 7, 0
    outer_v = rng.random((3, n))
    outer_Av = jax.vmap(A.mv, in_axes=0, out_axes=0)(outer_v)
    lv = 3

    H1, B1, V1, Z1, y1, *_ = _fgmres(
        A.mv,
        b,
        m=m,
        k=0,
        atol=atol,
        rpsolve=M.mv,
        outer_v=outer_v.T,
        outer_Av=outer_Av.T,
        lv=lv,
    )
    H2, B2, V2, Z2, y2, *_ = _fgmres(
        A.mv,
        b,
        m=m,
        k=0,
        atol=atol,
        rpsolve=M.mv,
        outer_v=outer_v.T,
        outer_Av=outer_Av.T,
        lv=lv,
        flexible=False,
    )

    np.testing.assert_allclose(H1, H2, rtol=1e-10)
    np.testing.assert_allclose(B1, B2, rtol=1e-10)
    np.testing.assert_allclose(V1, V2, rtol=1e-10)
    np.testing.assert_allclose(y1, y2, rtol=1e-10)


@pytest.mark.parametrize("flexible", [True, False])
def test_gcrotmk_base(flexible):
    """GCROT(m,k) from a cold start matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    m, k, tol, maxiter = 7, 5, 0, 10
    CU = []

    x1, _ = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, rtol=tol, maxiter=maxiter, m=m, k=k, CU=CU
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]

    x2, _, _, _, _, C2, U2 = gcrotmk(
        A, b, rtol=tol, maxiter=maxiter, m=m, k=k, refine=False, flexible=flexible
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)


@pytest.mark.parametrize("flexible", [True, False])
def test_gcrotmk_warm_start(flexible):
    """GCROT(m,k) warm-started with 1, k, and x0+k CU pairs matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    m, k, tol = 7, 5, 0

    # Round 1: cold start → produces k CU pairs.
    CU = []
    x1, _ = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, rtol=tol, maxiter=10, m=m, k=k, CU=CU
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]
    x2, _, _, _, _, C2, U2 = gcrotmk(
        A, b, rtol=tol, maxiter=10, m=m, k=k, refine=False, flexible=flexible
    )
    # sanity check: both implementations agree on the newest CU pair
    np.testing.assert_allclose(C1[:, 0], C2[:, 0])
    np.testing.assert_allclose(U1[:, 0], U2[:, 0])

    # Round 2: warm-start with 1 CU pair.
    CU = [(C1.T[0].copy(), U1.T[0].copy())]
    x1, _ = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, rtol=tol, maxiter=6, m=m, k=k, CU=CU
    )
    x2, _, _, _, _, C2, U2 = gcrotmk(
        A,
        b,
        rtol=tol,
        maxiter=6,
        m=m,
        k=k,
        C=C2[:, 0],
        U=U2[:, 0],
        refine=False,
        flexible=flexible,
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)

    # Round 3: warm-start with k CU pairs.
    CU = [(c, u) for c, u in zip(C1[:, :k].T, U1[:, :k].T)]
    x1, _ = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, rtol=tol, maxiter=1, m=m, k=k, CU=CU
    )
    x2, _, _, _, _, C2, U2 = gcrotmk(
        A,
        b,
        rtol=tol,
        maxiter=1,
        m=m,
        k=k,
        C=C2,
        U=U2,
        refine=False,
        flexible=flexible,
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)

    # Round 4: warm-start with x0 and k CU pairs.
    CU = [(c, u) for c, u in zip(C1[:, :k].T, U1[:, :k].T)]
    x1, _ = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, x0=x1, rtol=tol, maxiter=1, m=m, k=k, CU=CU
    )
    x2, _, _, _, _, C2, U2 = gcrotmk(
        A,
        b,
        x0=x2,
        rtol=tol,
        maxiter=1,
        m=m,
        k=k,
        C=C2,
        U=U2,
        refine=False,
        flexible=flexible,
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)


@pytest.mark.parametrize("flexible", [True, False])
def test_gcrotmk_preconditioner(flexible):
    """GCROT(m,k) with a right preconditioner matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    m, k, tol, maxiter = 7, 5, 0, 6
    CU = []

    x1, _ = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix),
        b,
        M=np.array(M.matrix),
        rtol=tol,
        maxiter=maxiter,
        m=m,
        k=k,
        CU=CU,
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]

    x2, _, _, _, _, C2, U2 = gcrotmk(
        A,
        b,
        MR=M,
        rtol=tol,
        maxiter=maxiter,
        m=m,
        k=k,
        refine=False,
        flexible=flexible,
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)


@pytest.mark.parametrize("flexible", [True, False])
def test_lgmres_base(flexible):
    """LGMRES from a cold start matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    m, k, tol, maxiter = 7, 5, 0, 10
    outer_v = []

    x1, _ = scipy.sparse.linalg.lgmres(
        np.array(A.matrix),
        b,
        rtol=tol,
        maxiter=maxiter,
        inner_m=m,
        outer_k=k,
        outer_v=outer_v,
        prepend_outer_v=True,
    )
    V1 = np.array([v for (v, Av) in outer_v])[::-1].T
    A1 = np.array([Av for (v, Av) in outer_v])[::-1].T

    x2, _, _, _, _, V2, A2 = lgmres(
        A, b, rtol=tol, maxiter=maxiter, m=m, k=k, refine=False, flexible=flexible
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(V1, V2)
    np.testing.assert_allclose(A1, A2)


@pytest.mark.parametrize("flexible", [True, False])
def test_lgmres_with_outer_v(flexible):
    """LGMRES warm-started with outer_v matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    m, k, tol, maxiter = 7, 5, 0, 10

    outer_v_arr = rng.random((3, n))
    outer_v1 = [(v, None) for v in outer_v_arr[::-1]]

    x1, _ = scipy.sparse.linalg.lgmres(
        np.array(A.matrix),
        b,
        rtol=tol,
        maxiter=maxiter,
        inner_m=m,
        outer_k=k,
        outer_v=outer_v1,
        prepend_outer_v=True,
    )
    V1 = np.array([v for (v, Av) in outer_v1])[::-1].T
    A1 = np.array([Av for (v, Av) in outer_v1])[::-1].T

    x2, _, _, _, _, V2, A2 = lgmres(
        A,
        b,
        rtol=tol,
        maxiter=maxiter,
        m=m,
        k=k,
        outer_v=outer_v_arr.T,
        refine=False,
        flexible=flexible,
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(V1, V2)
    np.testing.assert_allclose(A1, A2)


@pytest.mark.parametrize("flexible", [True, False])
def test_lgmres_preconditioner(flexible):
    """LGMRES with a left preconditioner matches scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    M = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    m, k, tol, maxiter = 5, 3, 0, 10
    outer_v1 = []

    x1, _ = scipy.sparse.linalg.lgmres(
        np.array(A.matrix),
        b,
        rtol=tol,
        M=np.array(M.matrix),
        maxiter=maxiter,
        inner_m=m,
        outer_k=k,
        outer_v=outer_v1,
        prepend_outer_v=True,
    )
    V1 = np.array([v for (v, Av) in outer_v1])[::-1].T
    A1 = np.array([Av for (v, Av) in outer_v1])[::-1].T

    x2, _, _, _, _, V2, A2 = lgmres(
        A,
        b,
        rtol=tol,
        ML=M,
        maxiter=maxiter,
        m=m,
        k=k,
        outer_v=None,
        refine=False,
        flexible=flexible,
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(V1, V2)
    np.testing.assert_allclose(A1, A2)


@pytest.mark.parametrize("flexible", [True, False])
@pytest.mark.parametrize("solver", [gcrotmk, lgmres], ids=["gcrotmk", "lgmres"])
def test_left_right_preconditioner(solver, flexible):
    """Left and right preconditioning both converge to the same solution."""
    rng = np.random.default_rng(123)
    n = 1000

    I = np.eye(n)
    b = rng.random(n)
    A = -scipy.sparse.random_array((n, n), rng=rng, format="csc")
    A += 5 * scipy.sparse.eye_array(n)
    Ai = scipy.sparse.linalg.spilu(A).solve(I)

    A = lx.MatrixLinearOperator(jnp.array(A.toarray()))
    M = lx.MatrixLinearOperator(jnp.array(Ai))

    m, k, maxiter, tol = 5, 3, 10, 1e-5

    x1, _, _, beta1, *_ = solver(
        A, b, rtol=tol, maxiter=maxiter, m=m, k=k, ML=M, flexible=flexible
    )
    x2, _, _, beta2, *_ = solver(
        A, b, rtol=tol, maxiter=maxiter, m=m, k=k, MR=M, flexible=flexible
    )

    assert beta1 / np.linalg.norm(b) < tol
    assert beta2 / np.linalg.norm(b) < tol
    np.testing.assert_allclose(x1, x2, rtol=tol, atol=tol)


@pytest.mark.parametrize("solver", [gcrotmk, lgmres], ids=["gcrotmk", "lgmres"])
def test_complex_system(solver):
    """Solve a complex-valued ``A x = b`` end to end."""
    n = 20
    rng = np.random.default_rng(0)
    A = rng.random((n, n)) + 1j * rng.random((n, n)) + n * np.eye(n)
    Aop = lx.MatrixLinearOperator(jnp.array(A))
    b = rng.random(n) + 1j * rng.random(n)
    x_true = np.linalg.solve(A, b)

    x, _, _, _, _, _, _ = solver(
        Aop, jnp.array(b), rtol=1e-10, maxiter=50, m=20, k=5, refine=False
    )
    np.testing.assert_allclose(np.asarray(x), x_true, rtol=1e-6, atol=1e-7)


def test_fgmres_optional_defaults():
    """FGMRES derives lv, outer_Av from outer_v, lc from C when they are None."""
    n = 20
    rng = np.random.default_rng(0)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n))))
    b = rng.random(n)
    b /= np.linalg.norm(b)
    m, k = 7, 5

    # outer_v given, lv and outer_Av defaulted (computed internally).
    outer_v = rng.random((n, 3))
    outer_Av = jax.vmap(A.mv, in_axes=1, out_axes=1)(outer_v)
    out_def = _fgmres(A.mv, b, m=m, k=k, atol=0.0, outer_v=outer_v)
    out_exp = _fgmres(
        A.mv, b, m=m, k=k, atol=0.0, outer_v=outer_v, outer_Av=outer_Av, lv=3
    )
    for a, c in zip(out_def[:5], out_exp[:5]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-10)

    # C given, lc defaulted (derived from C column count).
    C = np.linalg.qr(rng.random((n, 3)))[0]  # orthonormal columns
    out_def = _fgmres(A.mv, b, m=m, k=k, atol=0.0, C=C)
    out_exp = _fgmres(A.mv, b, m=m, k=k, atol=0.0, C=C, lc=3)
    for a, c in zip(out_def[:5], out_exp[:5]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-10)


def test_gcrotmk_optional_defaults():
    """GCROT with U supplied but C=None recomputes C = AU, k=None defaults to m."""
    n = 20
    rng = np.random.default_rng(1)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n)) + n * np.eye(n)))
    b = rng.random(n)
    U = rng.random((n, 2))

    # k=None -> k=m
    x, _, _, _, _, _, _ = gcrotmk(A, b, rtol=1e-10, maxiter=30, m=5, U=U)
    np.testing.assert_allclose(np.asarray(A.mv(x)), b, rtol=1e-6, atol=1e-7)


def test_lgmres_optional_defaults():
    """LGMRES with explicit x0 and verbose=True exercises the print branches."""
    n = 20
    rng = np.random.default_rng(2)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n)) + n * np.eye(n)))
    b = rng.random(n)

    x, _, _, _, _, _, _ = lgmres(
        A, b, x0=jnp.zeros(n), rtol=1e-10, maxiter=30, m=5, k=3, verbose=True
    )
    np.testing.assert_allclose(np.asarray(A.mv(x)), b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("flexible", [True, False])
def test_krylov_autodiff(flexible):
    """Test that gcrot is differentiable."""
    n = 20
    Af = jnp.array(np.random.normal(size=(n, n)))
    Ad = jnp.diag(jnp.arange(1, n + 1) ** 2)
    I = jnp.eye(n)
    b = jnp.array(np.random.normal(size=n))
    c = jnp.array(np.random.normal(size=n))

    def get_A(x):
        return lx.MatrixLinearOperator(Af @ Ad @ (Af + I * x))

    def solve_gcrot(x):
        A = get_A(x)
        bx = b * x
        M = InverseLinearOperator(A)
        y, _, _, res, _, _, _ = gcrotmk(
            A, bx, m=1, k=1, maxiter=1, MR=M, flexible=flexible
        )
        y = eqx.error_if(y, res > 1e-5, "didn't converge")
        return jnp.dot(c, y)

    def solve_lgmres(x):
        A = get_A(x)
        bx = b * x
        M = InverseLinearOperator(A)
        y, _, _, res, _, _, _ = lgmres(
            A, bx, m=1, k=1, maxiter=1, MR=M, flexible=flexible
        )
        y = eqx.error_if(y, res > 1e-5, "didn't converge")
        return jnp.dot(c, y)

    x0 = 1.0
    dA = jax.jacfwd(get_A)(x0)
    A = get_A(x0)
    db = b
    y0 = lx.linear_solve(A, b).value
    dy = lx.linear_solve(A, db - dA.mv(y0)).value
    dz_exact = jnp.dot(c, dy)

    dz_jvp_gcrot = jax.jacfwd(solve_gcrot)(x0)
    dz_vjp_gcrot = jax.jacrev(solve_gcrot)(x0)
    dz_jvp_lgmres = jax.jacfwd(solve_lgmres)(x0)
    dz_vjp_lgmres = jax.jacrev(solve_lgmres)(x0)

    np.testing.assert_allclose(dz_exact, dz_jvp_gcrot)
    np.testing.assert_allclose(dz_exact, dz_vjp_gcrot)
    np.testing.assert_allclose(dz_exact, dz_jvp_lgmres)
    np.testing.assert_allclose(dz_exact, dz_vjp_lgmres)


@pytest.mark.parametrize("solver_fn", [gcrotmk, lgmres], ids=["gcrotmk", "lgmres"])
def test_krylov_throw_forward(solver_fn):
    """throw=True raises on forward non-convergence; throw=False does not."""
    n = 20
    rng = np.random.default_rng(42)
    A = lx.MatrixLinearOperator(jnp.array(rng.random((n, n)) + n * np.eye(n)))
    b = jnp.array(rng.random(n))

    # maxiter=0: no iterations, residual = norm(b) >> rtol*norm(b), success=False
    solver_fn(A, b, m=1, k=1, maxiter=0, throw=False)

    with pytest.raises(Exception, match="forward"):
        solver_fn(A, b, m=1, k=1, maxiter=0, throw=True)


@pytest.mark.parametrize("solver_fn", [gcrotmk, lgmres], ids=["gcrotmk", "lgmres"])
def test_krylov_throw_tangent(solver_fn):
    """throw=True raises during vjp when the tangent solve fails to converge."""
    n = 30
    rng = np.random.default_rng(42)
    A_mat = rng.random((n, n)) + n * np.eye(n)
    b = jnp.array(rng.random(n))
    # x0 = exact solution so forward residual ≈ machine-epsilon << atol=1e-12
    x_exact = jnp.array(np.linalg.solve(A_mat, np.array(b)))
    A = lx.MatrixLinearOperator(jnp.array(A_mat))

    # m=1, k=1, maxiter=1 → only 2 inner Arnoldi steps per outer iteration.
    # Forward: starts at x_exact, residual ~1e-14 < atol=1e-12 → success immediately.
    # Tangent: starts at x_exact (wrong for A^T y = g), residual O(sqrt(n)) >> 1e-12,
    #          2 Krylov steps cannot converge a 30×30 system → success=False.
    def f(b):
        x, *_ = solver_fn(
            A, b, x0=x_exact, m=1, k=1, maxiter=1, rtol=0.0, atol=1e-12, throw=True
        )
        return x.sum()

    with pytest.raises(Exception, match="tangent"):
        jax.grad(f)(b)
