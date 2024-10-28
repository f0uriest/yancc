"""Linear algebra helpers."""

import cola
import jax
import jax.numpy as jnp
import numpy as np
import optimistix
from cola.linalg import pinv
from cola.fns import dot, add


class LSTSQ(cola.linalg.Algorithm):
    """
    Least-squares algorithm for computing the solve of a linear equation
    """
    def __call__(self, A: cola.ops.LinearOperator):
        return LSTSQSolve(A)


class LSTSQSolve(cola.ops.LinearOperator):
    def __init__(self, A: cola.ops.LinearOperator):
        super().__init__(A.dtype, (A.shape[-1], A.shape[-2]))
        self.A = A.to_dense()

    def _matmat(self, X):
        return self.xnp.lstsq(self.A, X)



@cola.dispatch(precedence=0)
def pinv(A: cola.ops.LinearOperator, alg: cola.linalg.Auto):
    """ Auto:
        - if A is small, use dense algorithms
        - if A is large, use iterative algorithms
    """
    alg = LSTSQ()
    return pinv(A, alg)


@cola.dispatch
def pinv(A: cola.ops.LinearOperator, alg: LSTSQ):
    return LSTSQSolve(A)


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Identity, alg: cola.linalg.Algorithm):
    return A


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Diagonal, alg: cola.linalg.Algorithm):  # noqa: F811
    """Pseudoinverse of diagonal matrix."""
    eps = jnp.finfo(A.dtype).eps * max(A.shape)
    small = jnp.abs(A.diag) < eps
    d = jnp.where(small, 0, 1 / jnp.where(small, 1, A.diag))
    return cola.ops.Diagonal(d)


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Kronecker, alg: cola.linalg.Algorithm):  # noqa: F811
    """Pseudoinverse of Kronecker matrix."""
    return cola.ops.Kronecker(*[pinv(M, alg) for M in A.Ms])







@cola.dispatch
def dot(A: cola.ops.Diagonal, B: cola.ops.Diagonal):  # noqa: F811
    """Product of 2 diagonals is diagonal."""
    return cola.ops.Diagonal(A.diag * B.diag)


@cola.dispatch
def dot(A: cola.ops.Dense, B: cola.ops.Dense):  # noqa: F811
    """Product of 2 dense is dense."""
    return cola.ops.Dense(A.to_dense() @ B.to_dense())


@cola.dispatch
def dot(A: cola.ops.Diagonal, B: cola.ops.Dense):  # noqa: F811
    """Product of diagonal * dense is dense."""
    return cola.ops.Dense(A.diag[:,None] * B.to_dense())


@cola.dispatch
def dot(A: cola.ops.Dense, B: cola.ops.Diagonal):  # noqa: F811
    """Product of dense * diagonal is dense."""
    return cola.ops.Dense(A.to_dense() * B.diag[None,:])


@cola.dispatch(precedence=2)
def dot(A: cola.ops.LinearOperator, B: cola.ops.Identity):  # noqa: F811
    return A


@cola.dispatch(precedence=2)
def dot(A: cola.ops.Identity, B: cola.ops.LinearOperator):  # noqa: F811
    return B


@cola.dispatch
def add(A: cola.ops.Diagonal, B: cola.ops.Diagonal):  # noqa: F811
    """Sum of 2 diagonals is diagonal."""
    return cola.ops.Diagonal(A.diag + B.diag)


@cola.dispatch
def add(A: cola.ops.Diagonal, B: cola.ops.Identity):  # noqa: F811
    """Sum of 2 diagonals is diagonal."""
    return cola.ops.Diagonal(A.diag + 1)


@cola.dispatch
def add(A: cola.ops.Identity, B: cola.ops.Diagonal):  # noqa: F811
    """Sum of 2 diagonals is diagonal."""
    return cola.ops.Diagonal(B.diag + 1)


@cola.dispatch
def add(A: cola.ops.Dense, B: cola.ops.Dense):  # noqa: F811
    """Sum of 2 dense is dense."""
    return cola.ops.Dense(A.to_dense() + B.to_dense())


@cola.dispatch
def add(A: cola.ops.Dense, B: cola.ops.Diagonal):  # noqa: F811
    """Sum of dense + diagonal is dense."""
    return cola.ops.Dense(A.to_dense() + B.to_dense())


@cola.dispatch
def add(A: cola.ops.Diagonal, B: cola.ops.Dense):  # noqa: F811
    """Sum of diagonal + dense is dense."""
    return cola.ops.Dense(A.to_dense() + B.to_dense())


@cola.dispatch
def add(A: cola.ops.Dense, B: cola.ops.Identity):  # noqa: F811
    """Sum of dense + diagonal is dense."""
    return cola.ops.Dense(A.to_dense() + B.to_dense())


@cola.dispatch
def add(A: cola.ops.Identity, B: cola.ops.Dense):  # noqa: F811
    """Sum of diagonal + dense is dense."""
    return cola.ops.Dense(A.to_dense() + B.to_dense())


def inv_sum_kron(A, B):
    """Inverse of A+B where A, B are Kronecker matrices of conformal size."""
    assert isinstance(A, cola.ops.Kronecker)
    assert isinstance(A, cola.ops.Kronecker)
    assert len(A.Ms) == len(B.Ms)
    As = A.Ms
    Bs = B.Ms
    assert all(b.shape == c.shape for b, c in zip(As, Bs))
    Vs = []
    es = 1.0
    ViBis = []

    for Ak, Bk in zip(As, Bs):
        Bi = cola.linalg.inv(Bk, alg=cola.linalg.LU())
        e, v = jnp.linalg.eig((Bi @ Ak).to_dense())
        V = cola.ops.Dense(v)
        Vi = cola.linalg.inv(V, alg=cola.linalg.LU())
        ViBi = Vi @ Bi
        Vs.append(V)
        es = jnp.multiply.outer(es, e)
        ViBis.append(ViBi)
    term1 = cola.ops.Kronecker(*Vs)
    term2 = cola.linalg.inv(cola.ops.Diagonal((es.flatten() + 1)))
    term3 = cola.ops.Kronecker(*ViBis)
    return term1 @ term2 @ term3


def _safeinv(x):
    small = jnp.abs(x) < 10 * jnp.finfo(x.dtype).eps
    return jnp.where(small, 0.0, 1 / jnp.where(small, 1.0, x))


def approx_kron_plus_eye(A):
    """For A = kron(diag(...)) Find B = kron(diag(...)) st B ~ A + I"""
    assert isinstance(A, cola.ops.Kronecker)
    sizes = [M.shape[0] for M in A.Ms]
    splits = np.cumsum(sizes)[:-1]
    ones = jnp.ones(A.shape[0])
    # A is diagonal so can get diag by dot with ones
    Adiag = A @ ones
    Ap1 = cola.ops.Diagonal(Adiag + 1)
    Ai = cola.linalg.pinv(Ap1)

    @jax.jit
    def loss(z, *args):
        zr = z[: len(z) // 2]
        zi = z[len(z) // 2 :]
        z = zr + 1j * zi
        zs = jnp.split(z, splits)
        Zis = (cola.ops.Diagonal(_safeinv(zk)) for zk in zs)
        Zi = cola.ops.Kronecker(*Zis)
        Y = Ai - Zi
        # Y is diagonal, so we can get the diagonal by dot with ones
        x = Y @ ones
        return jnp.sum(jnp.abs(x) ** 2)

    z0 = jnp.concatenate([M.diag for M in A.Ms])
    z0 = jnp.concatenate([z0.real, z0.imag])
    z = optimistix.minimise(
        loss, y0=z0, solver=optimistix.BFGS(rtol=1e-6, atol=1e-6), throw=False
    ).value
    zr = z[: len(z) // 2]
    zi = z[len(z) // 2 :]
    zc = zr + 1j * zi
    zs = jnp.split(zc, splits)
    Zs = (cola.ops.Diagonal(zk) for zk in zs)
    Z = cola.ops.Kronecker(*Zs)
    print(jnp.sqrt(loss(z) / loss(z * 0)))
    print(jnp.sqrt(loss(z0) / loss(z * 0)))
    return Z


def approx_kron_plus_eye1(A):
    """For A = kron(diag(...)) Find B = kron(diag(...)) st B ~ A + I"""
    assert isinstance(A, cola.ops.Kronecker)
    sizes = [M.shape[0] for M in A.Ms]
    ones = jnp.ones(A.shape[0])
    # A is diagonal so can get diag by dot with ones
    Adiag = A @ ones
    Ap1 = cola.ops.Diagonal(Adiag + 1)
    Ai = cola.linalg.pinv(Ap1)

    @jax.jit
    def loss(z, *args):
        zr = z[: len(z) // 2]
        zi = z[len(z) // 2 :]
        z = zr + 1j * zi
        Bis = (cola.ops.Diagonal(_safeinv(zk + M.diag)) for zk, M in zip(z, A.Ms))
        Bi = cola.ops.Kronecker(*Bis)
        Y = Ai - Bi
        # Y is diagonal, so we can get the diagonal by dot with ones
        x = Y @ ones
        return jnp.sum(jnp.abs(x) ** 2)

    z0 = jnp.zeros(len(sizes))
    z0 = jnp.concatenate([z0.real, z0.imag])
    z = optimistix.minimise(
        loss, y0=z0, solver=optimistix.BFGS(rtol=1e-6, atol=1e-6), throw=False
    ).value
    zr = z[: len(z) // 2]
    zi = z[len(z) // 2 :]
    zc = zr + 1j * zi
    Bs = (cola.ops.Diagonal(zk + M.diag) for zk, M in zip(zc, A.Ms))
    B = cola.ops.Kronecker(*Bs)
    print(jnp.sqrt(loss(z) / loss(z * 0)))
    print(jnp.sqrt(loss(z0) / loss(z * 0)))
    return B


def prodkron2kronprod(A):
    """Convert a product of Kronecker matrices into a Kronecker matrix of products."""
    assert isinstance(A, cola.ops.Product)
    Ks = A.Ms
    for K in Ks:
        assert isinstance(K, cola.ops.Kronecker)
        assert all(G.shape == M.shape for G, M in zip(K.Ms, Ks[0].Ms))

    Hs = []
    num = len(Ks[0].Ms)
    for j in range(num):
        H = cola.ops.Product(*(Ki.Ms[j] for Ki in Ks))
        Hs.append(H)
    return cola.ops.Kronecker(*Hs)


def kron_densify(A):
    """Make a Kronecker matrix of dense operators from nested operators."""
    assert isinstance(A, cola.ops.Kronecker)
    Hs = [cola.ops.Dense(M.to_dense()) for M in A.Ms]
    return cola.ops.Kronecker(*Hs)


def approx_sum_kron(A, B):
    """Find a Kronecker matrix C such that C ~ A + B where A, B are also Kronecker."""
    assert isinstance(A, cola.ops.Kronecker)
    assert isinstance(A, cola.ops.Kronecker)
    assert len(A.Ms) == len(A.Ms)
    As = A.Ms
    Bs = B.Ms
    assert all(b.shape == c.shape for b, c in zip(As, Bs))
    Vs = []
    es = []
    Vis = []

    for Ak, Bk in zip(As, Bs):
        Bi = cola.linalg.inv(Bk, alg=cola.linalg.LU())
        e, V = cola.linalg.eig((Bi @ Ak), k=Ak.shape[0], alg=cola.linalg.Eig())
        Vi = cola.linalg.inv(V, alg=cola.linalg.LU())
        Vs.append(V)
        Vis.append(Vi)
        es.append(cola.ops.Diagonal(e))
    V = cola.ops.Kronecker(*Vs)
    D = cola.ops.Kronecker(*es)
    D = approx_kron_plus_eye(D)
    Vi = cola.ops.Kronecker(*Vis)
    out = B @ V @ D @ Vi
    return prodkron2kronprod(out)
