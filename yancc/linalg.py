"""Linear algebra helpers."""

import functools
from typing import Any

import cola
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


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
    options: Any
    throw: bool = eqx.field(static=True)

    def __init__(
        self,
        operator: lx.AbstractLinearOperator,
        solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=True),
        options=None,
        throw=True,
    ):
        if options is None:
            options = {}
        self.operator = operator
        self.solver = solver
        state = solver.init(operator, options)
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        dynamic_state = jax.lax.stop_gradient(dynamic_state)
        self.static_state = static_state
        self.dynamic_state = dynamic_state
        self.options = options
        self.throw = throw

    def mv(self, vector):
        """Matrix vector product."""
        return lx.linear_solve(
            self.operator,
            vector,
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
    def kloop(k, b):
        j = n - 1 - k
        b = b.at[j].set(_safediv(b[j], lu[q, j]))

        def iloop(i, b):
            return b.at[i].add(-lu[q + i - j, j] * b[j])

        b = jax.lax.fori_loop(jnp.maximum(0, j - q), j, iloop, b)
        return b

    x = jax.lax.fori_loop(0, n, kloop, b)
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
