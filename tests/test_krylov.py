"""Tests for FGMRES/GCROT(m,k) solver."""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy

from yancc.krylov import _fgmres, gcrotmk, lgmres


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


def test_fgmres():
    """Test that FGMRES is equivalent to scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    A = lx.MatrixLinearOperator(jnp.array(A))
    M = rng.random((n, n))
    M = lx.MatrixLinearOperator(jnp.array(M))
    b = rng.random(n)
    b /= np.linalg.norm(b)  # scipy version assumes b is unit norm

    atol = 0
    m = 7
    k = 5
    C = None
    lc = None

    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A.mv, b, m=m + k, atol=atol, cs=()
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T

    H2, B2, V2, Z2, y2, _, _, _ = _fgmres(A.mv, b, m=m, k=k, atol=atol, C=C, lc=lc)

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)

    # test passing C
    atol = 0
    m = 7
    k = 5
    C = rng.random((3, n))
    lc = 3

    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A.mv, b, m=m + k - len(C), atol=atol, cs=C
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T

    H2, B2, V2, Z2, y2, _, _, _ = _fgmres(A.mv, b, m=m, k=k, atol=atol, C=C.T, lc=lc)

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)

    # test using bogus preconditioner
    atol = 0
    m = 7
    k = 5
    C = rng.random((3, n))
    lc = 3

    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A.mv, b, m=m + k - len(C), atol=atol, cs=C, rpsolve=M.mv
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T

    H2, B2, V2, Z2, y2, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C.T, lc=lc, rpsolve=M.mv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)

    # check using a partial C matrix
    atol = 0
    m = 7
    k = 5
    C = rng.random((3, n))
    C[-1] = 0
    lc = 2

    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A.mv, b, m=m + k - lc, atol=atol, cs=C[:-1], rpsolve=M.mv
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T

    H2, B2, V2, Z2, y2, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C.T, lc=lc, rpsolve=M.mv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)

    # test for stopping on tolerance
    atol = 0.2
    m = 15
    k = 5
    C = None
    lc = 0

    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A.mv, b, m=m + k, atol=atol, cs=(), rpsolve=M.mv
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T

    H2, B2, V2, Z2, y2, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C, lc=lc, rpsolve=M.mv
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

    # test for stopping on tolerance
    atol = 1e-10
    m = 70
    k = 0
    C = None
    lc = 0

    Q1, R1, B1, vs1, zs1, y1, _ = scipy.sparse.linalg._isolve._gcrotmk._fgmres(
        A.mv, b, m=m + k, atol=atol, cs=(), rpsolve=M.mv
    )
    H1 = (Q1 @ R1).T
    V1 = np.array(vs1).T
    Z1 = np.array(zs1).T

    H2, B2, V2, Z2, y2, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, C=C, lc=lc, rpsolve=M.mv
    )

    u = Z2 @ y2
    c = V2 @ H2.T @ y2
    a = np.linalg.norm(c)
    u /= a
    c /= a
    x = np.dot(c, b) * u

    assert np.linalg.norm(A.mv(x) - b) < atol

    # test passing v
    atol = 0
    m = 7
    k = 5
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

    H2, B2, V2, Z2, y2, _, _, _ = _fgmres(
        A.mv, b, m=m, k=k, atol=atol, outer_v=outer_v.T, outer_Av=outer_Av.T, lv=lv
    )

    np.testing.assert_allclose(*crop2(H1, H2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(B1, B2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(V1, V2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(Z1, Z2), rtol=1e-6)
    np.testing.assert_allclose(*crop2(y1, y2), rtol=1e-6)


def test_gcrotmk():
    """Test that GCROT(m,k) agrees with scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    A = lx.MatrixLinearOperator(jnp.array(A))
    M = rng.random((n, n))
    M = lx.MatrixLinearOperator(jnp.array(M))
    b = rng.random(n)

    tol = 0
    m = 7
    k = 5
    maxiter = 10
    CU = []

    x1, info1 = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, rtol=tol, maxiter=maxiter, m=m, k=k, CU=CU
    )
    # scipy outputs None for the last c, ours outputs Ax
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]

    x2, _, _, _, C2, U2 = gcrotmk(A, b, rtol=tol, maxiter=maxiter, m=m, k=k)

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)

    # test passing in C, U
    tol = 0
    m = 7
    k = 5
    maxiter = 6

    CU = [(C1.T[0].copy(), U1.T[0].copy())]
    # idiot check to make sure we're putting in the same thing to both
    np.testing.assert_allclose(CU[0][0], C2[:, 0])
    np.testing.assert_allclose(CU[0][1], U2[:, 0])

    x1, info1 = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, rtol=tol, maxiter=maxiter, m=m, k=k, CU=CU
    )

    x2, _, _, _, C2, U2 = gcrotmk(
        A, b, rtol=tol, maxiter=maxiter, m=m, k=k, C=C2[:, 0], U=U2[:, 0]
    )

    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)

    # test passing in C, U
    tol = 0
    m = 7
    k = 5
    maxiter = 1

    CU = [(c, u) for c, u in zip(C1[:, :k].T, U1[:, :k].T)]

    x1, info1 = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, rtol=tol, maxiter=maxiter, m=m, k=k, CU=CU
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]

    x2, _, _, _, C2, U2 = gcrotmk(A, b, rtol=tol, maxiter=maxiter, m=m, k=k, C=C2, U=U2)

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)

    # test passing in x
    tol = 0
    m = 7
    k = 5
    maxiter = 1

    CU = [(c, u) for c, u in zip(C1[:, :k].T, U1[:, :k].T)]

    x1, info1 = scipy.sparse.linalg.gcrotmk(
        np.array(A.matrix), b, x0=x1, rtol=tol, maxiter=maxiter, m=m, k=k, CU=CU
    )
    CU[-1] = (A.mv(x1), CU[-1][1])
    C1 = np.array([c for c, u in CU]).T[:, ::-1][:, :k]
    U1 = np.array([u for c, u in CU]).T[:, ::-1][:, :k]

    x2, _, _, _, C2, U2 = gcrotmk(
        A, b, x0=x2, rtol=tol, maxiter=maxiter, m=m, k=k, C=C2, U=U2
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(C1, C2)
    np.testing.assert_allclose(U1, U2)


def test_lgmres():
    """Test that LCGMRES agrees with scipy."""
    n = 20
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    A = lx.MatrixLinearOperator(jnp.array(A))
    M = rng.random((n, n))
    M = lx.MatrixLinearOperator(jnp.array(M))
    b = rng.random(n)

    tol = 0
    m = 7
    k = 5
    maxiter = 10
    outer_v = []

    x1, info1 = scipy.sparse.linalg.lgmres(
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

    x2, j_outer, nmv, beta, V2, A2 = lgmres(A, b, rtol=tol, maxiter=maxiter, m=m, k=k)

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(V1, V2)
    np.testing.assert_allclose(A1, A2)

    # test passing in outer_v
    outer_v2 = rng.random((3, n))
    outer_v1 = [(v, None) for v in outer_v2[::-1]]

    x1, info1 = scipy.sparse.linalg.lgmres(
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

    x2, j_outer, nmv, beta, V2, A2 = lgmres(
        A, b, rtol=tol, maxiter=maxiter, m=m, k=k, outer_v=outer_v2.T
    )

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(V1, V2)
    np.testing.assert_allclose(A1, A2)
