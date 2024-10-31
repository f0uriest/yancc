"""Linear algebra helpers."""

import cola
import jax.numpy as jnp
import numpy as np
import tensorly
from cola.fns import add, dot
from cola.linalg import pinv


class FullRank(cola.linalg.Algorithm):
    """Algorithm for computing full rank pinv using SVD"""

    def __call__(self, A: cola.ops.LinearOperator):
        """Do the solve."""
        return FullRankPinv(A)


class FullRankPinv(cola.ops.LinearOperator):
    """SVD based pseudo-inverse."""

    def __init__(self, A: cola.ops.LinearOperator):
        super().__init__(A.dtype, (A.shape[-1], A.shape[-2]))
        self.U, self.s, self.VT = jnp.linalg.svd(A.to_dense())

    def _matmat(self, X):
        si = _safepinv(self.s)
        y = si[:, None] * self.U.T @ X
        return self.VT.T @ y


def _min_nonzero(x):
    large = jnp.abs(x) > 1e-14
    return jnp.min(jnp.abs(x), where=large, initial=1.0)


def _safeinv(x):
    small = jnp.abs(x) < x.size * jnp.finfo(x.dtype).eps
    return jnp.where(small, 0.0, 1 / jnp.where(small, 1.0, x))


def _safepinv(x):
    small = jnp.abs(x) < x.size * jnp.finfo(x.dtype).eps
    return jnp.where(small, _min_nonzero(x), 1 / jnp.where(small, 1.0, x))


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.LinearOperator, alg: FullRank):  # noqa: F811
    """Dense SVD based pseudoinverse."""
    return FullRankPinv(A)


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Identity, alg: cola.linalg.Algorithm):  # noqa: F811
    """Pseudoinverse of identity matrix."""
    return A


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Diagonal, alg: FullRank):  # noqa: F811
    """Pseudoinverse of diagonal matrix."""
    di = _safepinv(A.diag)
    return cola.ops.Diagonal(di)


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Kronecker, alg: cola.linalg.Algorithm):  # noqa: F811
    """Pseudoinverse of Kronecker matrix."""
    return cola.ops.Kronecker(*[pinv(M, alg) for M in A.Ms])


@cola.dispatch
def dot(A: cola.ops.Diagonal, B: cola.ops.Diagonal):  # noqa: F811
    """Product of 2 diagonals is diagonal."""
    return cola.ops.Diagonal(A.diag * B.diag)


@cola.dispatch
def dot(A: cola.ops.Diagonal, B: cola.ops.Dense):  # noqa: F811
    """Product of diagonal * dense is dense."""
    return cola.ops.Dense(A.diag[:, None] * B.to_dense())


@cola.dispatch
def dot(A: cola.ops.Dense, B: cola.ops.Diagonal):  # noqa: F811
    """Product of dense * diagonal is dense."""
    return cola.ops.Dense(A.to_dense() * B.diag[None, :])


@cola.dispatch(precedence=2)
def dot(A: cola.ops.LinearOperator, B: cola.ops.Identity):  # noqa: F811
    """A @ I = A."""
    return A


@cola.dispatch(precedence=2)
def dot(A: cola.ops.Identity, B: cola.ops.LinearOperator):  # noqa: F811
    """I @ A = A."""
    return B


@cola.dispatch(precedence=3)
def dot(A: cola.ops.Identity, B: cola.ops.Identity):  # noqa: F811
    """I @ I = I."""
    return A


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
        Bi = cola.linalg.pinv(Bk, FullRank())
        assert not np.any(np.isnan(Bi.to_dense()))
        e, v = jnp.linalg.eig((Bi @ Ak).to_dense())
        V = cola.ops.Dense(v)
        Vi = cola.linalg.inv(V, alg=cola.linalg.LU())
        ViBi = Vi @ Bi
        Vs.append(V)
        es = jnp.multiply.outer(es, e)
        ViBis.append(ViBi)
    term1 = cola.ops.Kronecker(*Vs)
    term2 = cola.linalg.pinv(cola.ops.Diagonal(es.flatten() + 1), FullRank())
    term3 = cola.ops.Kronecker(*ViBis)
    return term1 @ term2 @ term3


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


def approx_sum_kron(A, B, inv=True):
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
        Bi = cola.linalg.pinv(Bk, FullRank())
        e, V = cola.linalg.eig((Bi @ Ak), k=Ak.shape[0], alg=cola.linalg.Eig())
        Vi = cola.linalg.inv(V, alg=cola.linalg.LU())
        Vs.append(V)
        Vis.append(Vi)
        es.append(e)
    V = cola.ops.Kronecker(*Vs)
    A = jnp.array(1.0)
    for e in es:
        A = jnp.multiply.outer(A, e)
    A += 1
    D = approx_kron(A, inv)
    Vi = cola.ops.Kronecker(*Vis)
    out = B @ V @ D @ Vi
    return prodkron2kronprod(out)


def approx_kron(A, inv=True):
    """For ndarray A, find B = kron(diag(...)) st B ~ diag(A.flatten())."""
    if inv:
        A = _safeinv(A)
    w, fs = tensorly.decomposition.parafac(A, 1)
    if inv:
        fs = [_safeinv(fi) for fi in fs]
    fs = [cola.ops.Diagonal(fi.squeeze()) for fi in fs]
    return cola.ops.Kronecker(*fs)


class BorderedOperator(cola.ops.LinearOperator):
    """Operator of the form [[A, B], [C, D]].

    Assumes A is invertible, though D need not be.
    """

    def __init__(self, A, B, C, D):
        assert A.shape[0] == B.shape[0]
        assert A.shape[1] == C.shape[1]
        assert B.shape[1] == D.shape[1]
        assert C.shape[0] == D.shape[0]
        self.A = cola.fns.lazify(A)
        self.B = cola.fns.lazify(B)
        self.C = cola.fns.lazify(C)
        self.D = cola.fns.lazify(D)
        dtypeAB = self.A.xnp.promote_types(self.A.dtype, self.B.dtype)
        dtypeCD = self.A.xnp.promote_types(self.C.dtype, self.D.dtype)
        dtype = self.A.xnp.promote_types(dtypeAB, dtypeCD)
        shape = (self.A.shape[0] + self.C.shape[0], self.A.shape[1] + self.B.shape[1])
        super().__init__(dtype, shape)

    def _matmat(self, X):
        # [A B] [X1] = [AX1 + BX2]
        # [C D] [X2] = [CX1 + DX2]
        X1 = X[: self.A.shape[0]]
        X2 = X[self.A.shape[0] :]
        Y1 = self.A @ X1 + self.B @ X2
        Y2 = self.C @ X1 + self.D @ X2
        return self.A.xnp.concat([Y1, Y2])


@cola.dispatch
def inv(A: BorderedOperator, alg: cola.linalg.Algorithm):
    """Block inverse assuming A is invertible."""
    A, B, C, D = A.A, A.B, A.C, A.D
    Ai = cola.linalg.inv(A, alg)
    schur = D - C @ Ai @ B
    schuri = cola.linalg.inv(schur, alg)
    AA = Ai + Ai @ B @ schuri @ C @ Ai
    BB = -Ai @ B @ schuri
    CC = -schuri @ C @ Ai
    DD = schuri
    return BorderedOperator(AA, BB, CC, DD)
