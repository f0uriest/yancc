"""Linear algebra helpers."""

import functools

import cola
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix
import scipy
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


def _safepinv(x):
    small = jnp.abs(x) < x.size * jnp.finfo(x.dtype).eps * jnp.abs(x).max()
    return jnp.where(small, _min_nonzero(x), 1 / jnp.where(small, 1.0, x))


class LSTSQ(cola.linalg.Algorithm):
    """Least-squares algorithm for computing the solve of a linear equation."""

    def __call__(self, A: cola.ops.LinearOperator):
        """Do the solve."""
        return LSTSQSolve(A)


class LSTSQSolve(cola.ops.LinearOperator):
    """Least-squares algorithm for computing the solve of a linear equation."""

    def __init__(self, A: cola.ops.LinearOperator):
        super().__init__(A.dtype, (A.shape[-1], A.shape[-2]))
        self.U, self.s, self.VT = jnp.linalg.svd(A.to_dense())

    def _matmat(self, X):
        small = (
            jnp.abs(self.s)
            < self.s.size * jnp.finfo(self.s.dtype).eps * jnp.abs(self.s).max()
        )
        si = jnp.where(small, self.s, 1 / self.s)
        y = si[:, None] * self.U.T @ X
        return self.VT.T @ y


def full_rank(A):
    """Minimally modify A to make it full rank."""
    if isinstance(A, cola.ops.Kronecker):
        return cola.ops.Kronecker(*[full_rank(M) for M in A.Ms])
    A = cola.fns.lazify(A)
    U, s, VT = jnp.linalg.svd(A.to_dense())
    large = s > 1e-15 * s.max() * s.size
    min_s = jnp.min(s, where=large, initial=1.0)
    min_s = min(min_s, 1)
    s = jnp.where(large, s, min_s)
    return cola.fns.lazify(U @ jnp.diag(s) @ VT)


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.LinearOperator, alg: FullRank):  # noqa: F811
    """Dense SVD based pseudoinverse."""
    return FullRankPinv(A)


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.LinearOperator, alg: LSTSQ):  # noqa: F811
    """Dense SVD based pseudoinverse."""
    return LSTSQSolve(A)


@cola.dispatch(precedence=2)
def pinv(A: cola.ops.Identity, alg: cola.linalg.Algorithm):  # noqa: F811
    """Pseudoinverse of identity matrix."""
    return A


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Diagonal, alg: FullRank):  # noqa: F811
    """Pseudoinverse of diagonal matrix."""
    di = _safepinv(A.diag)
    return cola.ops.Diagonal(di)


@cola.dispatch(precedence=1)
def pinv(A: cola.ops.Diagonal, alg: LSTSQ):  # noqa: F811
    """Pseudoinverse of diagonal matrix."""
    d = A.diag
    small = jnp.abs(d) < d.size * jnp.finfo(d.dtype).eps * jnp.abs(d).max()
    di = jnp.where(small, 0.0, 1 / jnp.where(small, 1.0, d))
    return cola.ops.Diagonal(di)


@cola.dispatch(precedence=2)
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


def inv_sum_kron(A, B, alg=FullRank()):
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
        Bi = cola.linalg.pinv(Bk, alg)
        e, v = jnp.linalg.eig((Bi @ Ak).to_dense())
        V = cola.ops.Dense(v)
        # eigenvectors should always be full rank so LU is fine
        Vi = cola.linalg.inv(V, alg=cola.linalg.LU())
        ViBi = Vi @ Bi
        Vs.append(V)
        es = jnp.multiply.outer(es, e)
        ViBis.append(ViBi)
    term1 = cola.ops.Kronecker(*Vs)
    term2 = cola.linalg.pinv(cola.ops.Diagonal(es.flatten() + 1), alg)
    term3 = cola.ops.Kronecker(*ViBis)
    if jnp.iscomplexobj(A) or jnp.iscomplexobj(B):
        return term1 @ term2 @ term3
    return RealOperator(term1 @ term2 @ term3)


def prodkron2kronprod(A, real=False):
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
        if real:
            Hs.append(RealOperator(H))
        else:
            Hs.append(H)
    return cola.ops.Kronecker(*Hs)


def approx_sum_kron_weights(A, B=None, **kwargs):
    """Find weights for linear combination of matrices for Kronecker approximation.

    Given A = sum(Kronecker(...)), and B = sum(Kronecker(...)) all of the same structure
    finds weights to minimize ||A - Kronecker(sum(wij Bij))||_F

    Parameters
    ----------
    A : iterable of cola.ops.Kronecker
        Matrix to approximate
    B : iterable of cola.ops.Kronecker
        Candidate matrices to use in approximation. Defaults to A

    Returns
    -------
    w : Array
        weights for linear combination of B to approximate A
    f : float
        Frobenius norm of approximation error
    """
    for a in A:
        assert isinstance(a, cola.ops.Kronecker)
        assert len(a.Ms) == len(A[0].Ms)
        assert all(b.shape == c.shape for b, c in zip(a.Ms, A[0].Ms))
    if B is not None:
        for a in B:
            assert isinstance(a, cola.ops.Kronecker)
            assert len(a.Ms) == len(A[0].Ms)
            assert all(b.shape == c.shape for b, c in zip(a.Ms, A[0].Ms))

    taa, tbb, tab = _compute_traces(A, B)

    atol = kwargs.get("atol", 1e-8)
    rtol = kwargs.get("rtol", 1e-8)
    max_steps = kwargs.get("max_steps", 256)

    def _minfun(x0):
        sol = optimistix.minimise(
            _objfun,
            optimistix.BFGS(rtol, atol),
            y0=x0,
            args=(taa, tbb, tab),
            max_steps=max_steps,
            throw=False,
        )
        return sol.value, _objfun(sol.value, (taa, tbb, tab))

    N = tbb.shape[0] * tbb.shape[1]
    if "x0" in kwargs:
        x0 = jnp.atleast_2d(kwargs["x0"])
    else:
        rng = scipy.stats.qmc.LatinHypercube(N, scramble=False, seed=0)
        x0 = -1 + 2 * rng.random(kwargs.get("nstart", 2 * N))
    xx, ff = jax.vmap(_minfun)(x0)
    i = jnp.argmin(ff)
    f = ff[i]
    x = xx[i]

    return x.reshape((tbb.shape[0], tbb.shape[1])), f


def approx_sum_kron_weights2(A, B=None, C=None, **kwargs):
    """Find weights for linear combination of matrices for Kronecker approximation.

    Given A,B,C = sum(Kronecker(...)) all of the same structure
    finds weights to minimize

    ||A - Kronecker(sum(alpha_ij B_ij)) - Kronecker(sum(beta_ij C_ij))||_F

    Parameters
    ----------
    A : iterable of cola.ops.Kronecker
        Matrix to approximate
    B : iterable of cola.ops.Kronecker
        Candidate matrices to use in approximation. Defaults to A
    C : iterable of cola.ops.Kronecker
        Candidate matrices to use in approximation. Defaults to B

    Returns
    -------
    alpha : Array
        weights for linear combination of B
    beta : Array
        weights for linear combination of C
    f : float
        Frobenius norm of approximation error
    """
    for a in A:
        assert isinstance(a, cola.ops.Kronecker)
        assert len(a.Ms) == len(A[0].Ms)
        assert all(b.shape == c.shape for b, c in zip(a.Ms, A[0].Ms))
    if B is not None:
        for a in B:
            assert isinstance(a, cola.ops.Kronecker)
            assert len(a.Ms) == len(A[0].Ms)
            assert all(b.shape == c.shape for b, c in zip(a.Ms, A[0].Ms))
    if C is not None:
        for a in C:
            assert isinstance(a, cola.ops.Kronecker)
            assert len(a.Ms) == len(A[0].Ms)
            assert all(b.shape == c.shape for b, c in zip(a.Ms, A[0].Ms))

    taa, tbb, tcc, tab, tac, tbc = _compute_traces2(A, B, C)

    atol = kwargs.get("atol", 1e-8)
    rtol = kwargs.get("rtol", 1e-8)
    max_steps = kwargs.get("max_steps", 256)

    def _minfun(x0):
        sol = optimistix.minimise(
            _objfun2,
            optimistix.BFGS(rtol, atol),
            y0=x0,
            args=(taa, tbb, tcc, tab, tac, tbc),
            max_steps=max_steps,
            throw=False,
        )
        return sol.value, _objfun2(sol.value, (taa, tbb, tcc, tab, tac, tbc))

    Na = tbb.shape[0] * tbb.shape[1]
    Nb = tcc.shape[0] * tcc.shape[1]
    N = Na + Nb
    if "x0" in kwargs:
        x0 = jnp.atleast_2d(kwargs["x0"])
    else:
        rng = scipy.stats.qmc.LatinHypercube(N, scramble=False, seed=0)
        x0 = -1 + 2 * rng.random(kwargs.get("nstart", 2 * N))
    xx, ff = jax.vmap(_minfun)(x0)
    i = jnp.argmin(ff)
    f = ff[i]
    x = xx[i]
    alpha = x[:Na]
    beta = x[Na:]

    return (
        alpha.reshape((tbb.shape[0], tbb.shape[1])),
        beta.reshape((tcc.shape[0], tcc.shape[1])),
        f,
    )


def _compute_traces(A, B=None):
    """Compute inner products needed for Kronecker product approximation."""
    I = len(A)
    J = len(A[0].Ms)
    taa = jnp.zeros((J, I, I))
    for j in range(J):
        for i in range(I):
            for l in range(I):
                t = cola.linalg.trace(A[i].Ms[j].T @ A[l].Ms[j])
                taa = taa.at[j, i, l].set(t)
    if B is None:  # B = A
        tab = tbb = taa
    else:
        K = len(B)
        tab = jnp.zeros((J, I, K))
        tbb = jnp.zeros((J, K, K))
        for j in range(J):
            for i in range(I):
                for k in range(K):
                    t = cola.linalg.trace(A[i].Ms[j].T @ B[k].Ms[j])
                    tab = tab.at[j, i, k].set(t)
            for k in range(K):
                for l in range(K):
                    t = cola.linalg.trace(B[k].Ms[j].T @ B[l].Ms[j])
                    tbb = tbb.at[j, k, l].set(t)

    return taa, tbb, tab


def _compute_traces2(A, B=None, C=None):
    """Compute inner products needed for Kronecker product approximation."""
    taa, tbb, tab = _compute_traces(A, B)

    if C is None:  # C = B
        tac = tab
        tcc = tbb
    else:
        J = len(A[0].Ms)
        I = len(A)
        L = len(C)
        tac = jnp.zeros((J, I, L))
        tcc = jnp.zeros((J, L, L))
        for j in range(J):
            for i in range(I):
                for l in range(L):
                    t = cola.linalg.trace(A[i].Ms[j].T @ C[l].Ms[j])
                    tac = tac.at[j, i, l].set(t)
            for l in range(L):
                for m in range(L):
                    t = cola.linalg.trace(C[l].Ms[j].T @ C[m].Ms[j])
                    tcc = tcc.at[j, l, m].set(t)
    if B is None and C is None:  # B = C = A
        tbc = taa
    elif C is None:  # C = B
        tbc = tbb
    elif B is None:  # B = A
        tbc = tac
    else:
        K = len(B)
        tbc = jnp.zeros((J, K, L))
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    t = cola.linalg.trace(B[k].Ms[j].T @ C[l].Ms[j])
                    tbc = tbc.at[j, k, l].set(t)

    return taa, tbb, tcc, tab, tac, tbc


def _objfun(x, t):
    """Objective function for Kronecker product approximation."""
    taa, tbb, tab = t
    x = x.reshape((tbb.shape[0], tbb.shape[1]))
    term_aa = taa.prod(axis=0).sum()
    term_ab = (x[:, None, :] * tab).sum(axis=2).prod(axis=0).sum()
    term_bb = (x[:, :, None] * tbb * x[:, None, :]).sum(axis=(1, 2)).prod()
    res = term_aa + term_bb - 2 * term_ab
    return jnp.sqrt(jnp.abs(res / term_aa))


def _objfun2(x, t):
    """Objective function for Kronecker product approximation."""
    taa, tbb, tcc, tab, tac, tbc = t
    alpha = x[: tbb.shape[0] * tbb.shape[1]]
    beta = x[tbb.shape[0] * tbb.shape[1] :]
    alpha = alpha.reshape((tbb.shape[0], tbb.shape[1]))
    beta = beta.reshape((tcc.shape[0], tcc.shape[1]))
    term_aa = taa.prod(axis=0).sum()
    term_bb = (alpha[:, :, None] * tbb * alpha[:, None, :]).sum(axis=(1, 2)).prod()
    term_cc = (beta[:, :, None] * tcc * beta[:, None, :]).sum(axis=(1, 2)).prod()
    term_ab = (alpha[:, None, :] * tab).sum(axis=2).prod(axis=0).sum()
    term_ac = (beta[:, None, :] * tac).sum(axis=2).prod(axis=0).sum()
    term_bc = (beta[:, None, :] * tbc * alpha[:, :, None]).sum(axis=(1, 2)).prod()
    res = term_aa + term_bb + term_cc - 2 * term_ab - 2 * term_ac + 2 * term_bc
    return jnp.sqrt(jnp.abs(res / term_aa))


def approx_sum_kron(A, B=None, **kwargs):
    """Find a Kronecker product to approximate a sum of Kronecker products.

    Given Kronecker matrices A1, A2, ... of conformal sizes, finds a Kronecker matrix B
    that minimizes ||sum(A)-B||_F.

    Parameters
    ----------
    A : iterable of cola.ops.Kronecker
        Sum of these is the matrix to approximate
    B : iterable of cola.ops.Kronecker
        Candidate matrices to use in approximation. Defaults to components of A

    Returns
    -------
    B : cola.ops.Kronecker
        Matrix that approximates sum(A)
    """
    w, f = approx_sum_kron_weights(A, B, **kwargs)
    # w shape(N, T) for N products, T terms in sum
    if B is None:
        B = A
    out = []
    for i in range(w.shape[0]):
        outi = []
        for j in range(w.shape[1]):
            wij = w[i, j]
            Bij = B[j].Ms[i]
            outi.append(wij * Bij)
        outi = cola.ops.Sum(*outi)
        out.append(outi)
    return cola.ops.Kronecker(*out)


def approx_sum_kron2(A, B=None, C=None, **kwargs):
    """Find a Kronecker product to approximate a sum of Kronecker products.

    Given Kronecker matrices A1, A2, ... of conformal sizes, finds a Kronecker matrices
    B and C that minimizes ||sum(A) - (B + C)||_F.

    Parameters
    ----------
    A : iterable of cola.ops.Kronecker
        Sum of these is the matrix to approximate
    B : iterable of cola.ops.Kronecker
        Candidate matrices to use in approximation. Defaults to components of A
    C : iterable of cola.ops.Kronecker
        Candidate matrices to use in approximation. Defaults to components of B

    Returns
    -------
    B : cola.ops.Kronecker
        Matrix that approximates sum(A)
    C : cola.ops.Kronecker
        Matrix that approximates sum(A)
    """
    alpha, beta, f = approx_sum_kron_weights2(A, B, C, **kwargs)
    # w shape(N, T) for N products, T terms in sum
    if B is None:
        B = A
    if C is None:
        C = B
    out1 = []
    for i in range(alpha.shape[0]):
        outi = []
        for j in range(alpha.shape[1]):
            wij = alpha[i, j]
            Bij = B[j].Ms[i]
            outi.append(wij * Bij)
        outi = cola.ops.Sum(*outi)
        out1.append(outi)
    out1 = cola.ops.Kronecker(*out1)
    out2 = []
    for i in range(beta.shape[0]):
        outi = []
        for j in range(beta.shape[1]):
            wij = beta[i, j]
            Cij = C[j].Ms[i]
            outi.append(wij * Cij)
        outi = cola.ops.Sum(*outi)
        out2.append(outi)
    out2 = cola.ops.Kronecker(*out2)
    return out1, out2


def approx_kron_diag2d(D, n1, n2):
    """Approximate D as a kronecker matrix of size n1, n2."""
    if len(D.shape) == 2:
        D = cola.fns.lazify(D)
        d = cola.fns.diag(D)
    else:
        d = D
    d = d.reshape((n1, n2))
    u, s, v = jnp.linalg.svd(d)
    A = cola.ops.Diagonal(u[:, 0] * s[0])
    B = cola.ops.Diagonal(v[0])
    return cola.ops.Kronecker(A, B)


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


class BlockOperator(cola.ops.LinearOperator):
    """Block matrix.

    Assumes all blocks have the same size and dtype.
    """

    def __init__(self, blocks: list[list[cola.ops.LinearOperator]]):

        # first make sure all rows have the same # of blocks
        nrow = len(blocks)
        ncol = len(blocks[0])
        for blockrow in blocks[1:]:
            assert len(blockrow) == ncol

        blockshape = blocks[0][0].shape
        blockdtype = blocks[0][0].dtype
        for i in range(len(blocks)):
            for j in range(len(blocks[i])):
                blocks[i][j] = cola.fns.lazify(blocks[i][j])
                assert blocks[i][j].shape == blockshape
                assert blocks[i][j].dtype == blockdtype

        shape = (blockshape[0] * nrow, blockshape[1] * ncol)
        self.blocks = blocks
        self.blockshape = blockshape
        self.nrow = nrow
        self.ncol = ncol
        super().__init__(blockdtype, shape)

    def _matmat(self, X):
        Xblocks = jnp.split(X, self.ncol)
        out = []
        for i in range(self.nrow):
            outrow = 0
            for j in range(self.ncol):
                Aij = self.blocks[i][j]
                outrow += Aij @ Xblocks[j]
            out.append(outrow)

        return jnp.concatenate(out)


class RealOperator(cola.ops.LinearOperator):
    """Real part of a linear operator."""

    def __init__(self, op):
        self._op = op
        super().__init__(jnp.empty(0, dtype=op.dtype).real.dtype, op.shape)

    def _matmat(self, X):
        # assume op = R + jC, X = r + jc
        # (R+jC)(r+jc) = Rr + jRc + jCr - Cc
        # (R+jC)(r) = Rr + jCr
        # (R+jC)(jc) = jRc - Cc
        # (R)(r+jc) = Rr + jRc = real(op @ r) + j*real(op @ c)
        # (C)(r+jc) = Cr - jCc = imag(op @ -j*r) - j*imag(op @ c)
        r, c = X.real, X.imag
        out = jnp.real(self._op @ r) + 1j * jnp.real(self._op @ c)
        if jnp.iscomplexobj(X):
            return out
        return jnp.real(out)


class ImagOperator(cola.ops.LinearOperator):
    """Imaginary part of a linear operator."""

    def __init__(self, op):
        self._op = op
        super().__init__(jnp.empty(0, dtype=op.dtype).real.dtype, op.shape)

    def _matmat(self, X):
        r, c = X.real, X.imag
        out = jnp.real(self._op @ (-1j * r)) + 1j * jnp.imag(self._op @ c)
        if jnp.iscomplexobj(X):
            return out
        return jnp.real(out)


def build_precond(Ai, B, C, D):
    """Inverse of bordered operator, assuming A is already inverted."""
    Ai, B, C, D = map(cola.fns.lazify, (Ai, B, C, D))
    schur = D - C @ Ai @ B
    schuri = cola.linalg.pinv(cola.fns.lazify(schur), FullRank())
    AA = Ai + Ai @ B @ schuri @ C @ Ai
    BB = -Ai @ B @ schuri
    CC = -schuri @ C @ Ai
    DD = schuri
    return BorderedOperator(AA, BB, CC, DD)


def make_dense_tridiag(l, d, u, l0=jnp.array(0.0), un=jnp.array(0.0)):
    """Make a dense matrix from tridiagonals.

    Parameters
    ----------
    l, d, u : jax.Array, shape[N-1], shape[N], shape[N-1]
        lower, main, upper diagonals
    l0, un : float
        cyclic edge values
    """
    out = jnp.diag(d) + jnp.diag(l, k=-1) + jnp.diag(u, k=1)
    out = out.at[0, -1].set(l0).at[-1, 0].set(un)
    return out


@jax.jit
@functools.partial(jnp.vectorize, signature="(m,m),(m)->(m)")
def tridiag_solve_dense(A, b):
    """Solve a (possibly cyclical) tridiagonal system in dense form."""
    d = jnp.diag(A, k=0)
    l = jnp.diag(A, k=-1)
    u = jnp.diag(A, k=1)
    l0 = A[0, -1]
    un = A[-1, 0]
    return tridiag_solve(l, d, u, b, l0, un)


@jax.jit
def tridiag_solve(l, d, u, b, l0=jnp.array(0.0), un=jnp.array(0.0)):
    """Solve a (possibly cyclical) tridiagonal system in banded form.

    Parameters
    ----------
    l, d, u : jax.Array, shape[N-1], shape[N], shape[N-1]
        lower, main, upper diagonals.
    b : jax.Array, shape[N]
        rhs vector.
    l0, un : float
        cyclic edge values.
    """
    return _tridiag_solve_cyclic(l, d, u, b, l0, un)


@functools.partial(jnp.vectorize, signature="(m),(n),(m),(n),(),()->(n)")
def _tridiag_solve_cyclic(l, d, u, b, l0, un):
    gamma = -d[0]
    t = jnp.zeros_like(d).at[0].set(gamma).at[-1].set(un)
    v = jnp.zeros_like(d).at[0].set(1.0).at[-1].set(l0 / gamma)
    d = d.at[0].add(-gamma).at[-1].add(-un * l0 / gamma)
    y = _tridiag_solve(l, d, u, b)
    q = _tridiag_solve(l, d, u, t)
    out = y - q * jnp.dot(v, y) / (1 + jnp.dot(v, q))
    return out


def _tridiag_solve(l, d, u, b, *args):
    l = jnp.pad(l, (1, 0))
    u = jnp.pad(u, (0, 1))
    return jax.lax.linalg.tridiagonal_solve(l, d, u, b[:, None])[:, 0]


@jax.jit
def lstsq(a, b):
    """Least squares via normal equations."""
    A = a.T @ a
    B = a.T @ b
    return jnp.linalg.solve(A, B)


class InverseLinearOperator(lx.AbstractLinearOperator):
    """Inverse of another linear operator."""

    operator: lx.AbstractLinearOperator
    solver: lx.AbstractLinearSolver
    static_state: object = eqx.field(static=True)
    dynamic_state: object
    options: dict
    throw: bool = eqx.field(static=True)

    def __init__(
        self,
        operator,
        solver=lx.AutoLinearSolver(well_posed=True),
        options=None,
        throw=True,
    ):
        self.operator = operator
        self.solver = solver
        state = solver.init(operator, options)
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        dynamic_state = jax.lax.stop_gradient(dynamic_state)
        self.static_state = static_state
        self.dynamic_state = dynamic_state
        self.options = options
        self.throw = throw

    def mv(self, x):
        """Matrix vector product."""
        return lx.linear_solve(
            self.operator,
            x,
            solver=self.solver,
            state=eqx.combine(self.dynamic_state, self.static_state),
            options=self.options,
            throw=self.throw,
        ).value

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return self.operator.out_structure()

    def out_structure(self):
        """Pytree structure of expected output."""
        return self.operator.in_structure()

    def transpose(self):
        """Transpose of the operator."""
        return InverseLinearOperator(self.operator.T, self.solver)


@lx.is_symmetric.register(InverseLinearOperator)
def _(operator):
    return lx.is_symmetric(operator.operator)


@lx.is_diagonal.register(InverseLinearOperator)
def _(operator):
    return lx.is_diagonal(operator.operator)


@functools.partial(jax.jit, static_argnames=["p", "q", "periodic"])
@functools.partial(jnp.vectorize, signature="(m,n)->(n,n)", excluded=(0, 1, 3))
def banded_to_dense(p, q, A, periodic=False):
    """Convert from banded representation to dense.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    A : jax.Array, shape(...,p+q+1, N)
        Matrix in banded storage format.
    periodic : bool
        Whether to include periodic parts of the matrix (ie upper right and lower left
        corners).

    Returns
    -------
    A : jax.Array, shape(...,N,N)
        Matrix in dense format.
    """
    n = A.shape[1]
    assert p <= n
    assert q <= n
    if periodic:
        assert n > 2 * max(p, q)
    B = jnp.zeros((n, n))
    for k, a in zip(range(-p, q + 1), A[::-1]):
        if k == 0:
            B += jnp.diag(a, k=k)
        elif k > 0:
            B += jnp.diag(a[k:], k=k)
            if periodic:
                B += jnp.diag(a[:k], k=-n + k)
        else:
            B += jnp.diag(a[:k], k=k)
            if periodic:
                B += jnp.diag(a[k:], k=n + k)

    return B


@functools.partial(jax.jit, static_argnames=["p", "q", "periodic"])
@functools.partial(jnp.vectorize, signature="(n,n)->(m,n)", excluded=(0, 1, 3))
def dense_to_banded(p, q, A, periodic=False):
    """Convert from dense representation to banded.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    A : jax.Array, shape(...,N,N)
        Matrix in dense format.
    periodic : bool
        Whether to include periodic parts of the matrix (ie upper right and lower left
        corners).

    Returns
    -------
    A : jax.Array, shape(...,p+q+1, N)
        Matrix in banded storage format.
    """
    n = A.shape[1]
    assert p <= n
    assert q <= n
    if periodic:
        assert n > 2 * max(p, q)
    B = jnp.zeros((p + q + 1, n))
    for i, k in enumerate(range(q, -p - 1, -1)):
        if k == 0:
            B = B.at[i].set(jnp.diag(A, k=k))
        elif k < 0:
            if periodic:
                B = B.at[i].set(
                    jnp.concatenate([jnp.diag(A, k=k), jnp.diag(A, k=n - abs(k))])
                )
            else:
                B = B.at[i].set(jnp.pad(jnp.diag(A, k=k), (0, abs(k))))
        else:
            if periodic:
                B = B.at[i].set(
                    jnp.concatenate([jnp.diag(A, k=-n + k), jnp.diag(A, k=k)])
                )
            else:
                B = B.at[i].set(jnp.pad(jnp.diag(A, k=k), (abs(k), 0)))
    return B


def _safediv(a, b):
    mask = jnp.abs(b) < jnp.finfo(b.dtype).eps
    b = jnp.where(mask, 1, b)
    return jnp.where(mask, 0, a / b)


@functools.partial(jax.jit, static_argnames=("p", "q"))
@functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=(0, 1))
def lu_factor_banded(p, q, A):
    """LU factorization of banded matrix in banded storage format.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    A : jax.Array, shape(...,p+q+1,N)
        Matrix in banded format.

    Returns
    -------
    lu : jax.Array, shape(...,p+q+1,N)
        LU factorized matrix. Upper triangle is U, lower triangle is L (unit diagonal
        is assumed.)
    """
    n = A.shape[1]
    assert p <= n
    assert q <= n
    assert A.shape[0] == (p + q + 1)

    def kloop(k, A):

        def iloop(i, A):
            return A.at[q + i - k, k].set(_safediv(A[q + i - k, k], A[q, k]))

        A = jax.lax.fori_loop(k + 1, jnp.minimum(k + p + 1, n), iloop, A)

        def jloop(j, A):

            def mloop(m, A):
                return A.at[q + m - j, j].add(-A[q + m - k, k] * A[q + k - j, j])

            A = jax.lax.fori_loop(k + 1, jnp.minimum(k + p + 1, n), mloop, A)
            return A

        A = jax.lax.fori_loop(k + 1, jnp.minimum(k + q + 1, n), jloop, A)
        return A

    A = jax.lax.fori_loop(0, n - 1, kloop, A)
    return A


@functools.partial(jax.jit, static_argnames=("p", "q"))
@functools.partial(jnp.vectorize, signature="(m,n),(n)->(n)", excluded=(0, 1))
def lu_solve_banded(p, q, lu, b):
    """Solve a linear system with a pre-factored banded matrix in banded storage format.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    lu : jax.Array, shape(...,p+q+1,N)
        LU factorization of matrix in banded format. Output from ``lu_factor_banded``.
    b : jax.Array, shape(...,N)
        RHS vector.

    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    n = lu.shape[1]
    assert p <= n
    assert q <= n
    assert lu.shape[0] == (p + q + 1)

    # first solve Ly = b, overwriting b with y
    def jloop(j, b):
        def iloop(i, b):
            return b.at[i].add(-lu[q + i - j, j] * b[j])

        b = jax.lax.fori_loop(j + 1, jnp.minimum(j + p + 1, n), iloop, b)
        return b

    b = jax.lax.fori_loop(0, n, jloop, b)

    # now solve Ux=y, overwriting y with x
    def jloop(k, b):
        j = n - 1 - k
        b = b.at[j].set(_safediv(b[j], lu[q, j]))

        def iloop(i, b):
            return b.at[i].add(-lu[q + i - j, j] * b[j])

        b = jax.lax.fori_loop(jnp.maximum(0, j - q), j, iloop, b)
        return b

    x = jax.lax.fori_loop(0, n, jloop, b)
    return x


@functools.partial(jax.jit, static_argnames=("p", "q"))
@functools.partial(jnp.vectorize, signature="(m,n),(n)->(n)", excluded=(0, 1))
def solve_banded(p, q, A, b):
    """Solve a linear system with a banded matrix in banded storage format.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    A : jax.Array, shape(...,p+q+1,N)
        Matrix in banded format.
    b : jax.Array, shape(...,N)
        RHS vector.

    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    lu = lu_factor_banded(p, q, A)
    return lu_solve_banded(p, q, lu, b)


@functools.partial(jax.jit, static_argnames=("r",))
@functools.partial(
    jnp.vectorize, signature="(n,n)->(k,n),(n),(n,m),(m,n)", excluded=(0,)
)
def lu_factor_banded_periodic(r, A):
    """LU factorization of periodic banded matrix in dense storage format.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    r: int
        Maximum of lower and upper bandwidth.
    A : jax.Array, shape(...,N,N)
        Matrix in dense format.

    Returns
    -------
    lu : jax.Array, shape(...,N,N)
        LU factorized matrix. Upper triangle is U, lower triangle is L (unit diagonal
        is assumed.)
    BUschur : jax.Array, shape(...,N, 2*r+1)
        Additional matrix for solving the periodic part
    V : jax.Array, shape(...,2*r+1, N)
        Additional matrix for solving the periodic part
    """
    nn = A.shape[0]
    if (nn <= 2 * r) or (r == 0):
        lu, piv = jax.scipy.linalg.lu_factor(A)
        return lu, piv, jnp.zeros((nn, r)), jnp.zeros((r, nn))
    F = A[:r, -r:]
    G = A[-r:, :r]
    A0 = A[:r, :r]
    U0 = jnp.eye(r)
    V0 = -A0
    Vn = F
    Un = -G @ jnp.linalg.inv(A0)
    C = jnp.eye(r)
    U = jnp.concatenate([U0, jnp.zeros((nn - 2 * r, r)), Un], axis=0)
    V = jnp.concatenate([V0, jnp.zeros((r, nn - 2 * r)), Vn], axis=1)
    B = dense_to_banded(r, r, A - U @ V)
    lu = lu_factor_banded(r, r, B)
    BinvU = lu_solve_banded(r, r, lu, U.T).T
    schur = jnp.linalg.inv(C + V @ BinvU)
    BUschur = BinvU @ schur
    piv = jnp.arange(nn)  # dummy pivots for now
    return lu, piv, BUschur, V


@functools.partial(jax.jit, static_argnames=("r",))
def lu_solve_banded_periodic(r, lu_schur_v, b):
    """Solve a periodic banded linear system with matrix pre-factored.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    r: int
        Maximum of lower and upper bandwidth.
    lu_schur_v : tuple of jax.Array
        Output from ``lu_factor_banded_periodic``
    b : jax.Array, shape(...,N)
        RHS vector.


    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    Blu, piv, BUschur, V = lu_schur_v
    return _lu_solve_banded_periodic(r, Blu, piv, BUschur, V, b)


@functools.partial(
    jnp.vectorize, signature="(k,n),(n),(n,m),(m,n),(n)->(n)", excluded=(0,)
)
def _lu_solve_banded_periodic(r, lu, piv, BUschur, V, b):
    nn = b.shape[-1]
    if (nn <= 2 * r) or (r == 0):
        return jax.scipy.linalg.lu_solve((lu, piv), b)
    Binvb = lu_solve_banded(r, r, lu, b)
    return Binvb - BUschur @ (V @ Binvb)


@functools.partial(jax.jit, static_argnames=("r",))
@functools.partial(jnp.vectorize, signature="(n,n),(n)->(n)", excluded=(0,))
def solve_banded_periodic(r, A, b):
    """Solve a periodic banded linear system.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    r: int
        Maximum of lower and upper bandwidth.
    A : jax.Array, shape(...,N,N)
        Matrix in dense format.
    b : jax.Array, shape(...,N)
        RHS vector.


    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    lu_schur_v = lu_factor_banded_periodic(r, A)
    return lu_solve_banded_periodic(r, lu_schur_v, b)
