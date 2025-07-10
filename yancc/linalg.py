"""Linear algebra helpers."""

import functools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


class BorderedOperator(lx.AbstractLinearOperator):
    """Operator for a bordered matrix.

    [A B]
    [C D]
    """

    A: lx.AbstractLinearOperator
    B: lx.AbstractLinearOperator
    C: lx.AbstractLinearOperator
    D: lx.AbstractLinearOperator

    def __init__(self, A, B, C, D):
        assert A.out_size() == B.out_size()
        assert A.in_size() == C.in_size()
        assert B.in_size() == D.in_size()
        assert C.out_size() == D.out_size()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def mv(self, vector):
        """Matrix vector product."""
        # [A B] [X1] = [AX1 + BX2]
        # [C D] [X2] = [CX1 + DX2]
        X1 = vector[: self.A.in_size()]
        X2 = vector[self.A.in_size() :]
        Y1 = self.A.mv(X1) + self.B.mv(X2)
        Y2 = self.C.mv(X1) + self.D.mv(X2)
        return jnp.concatenate([Y1, Y2])

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv, out_axes=-1)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.A.in_size() + self.D.in_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.A.out_size() + self.D.out_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class InverseBorderedOperator(lx.AbstractLinearOperator):
    """Inverse of a bordered matrix, using already inverted A."""

    Ai: lx.AbstractLinearOperator
    B: lx.AbstractLinearOperator
    C: lx.AbstractLinearOperator
    D: lx.AbstractLinearOperator
    schuri: lx.AbstractLinearOperator
    AiB: lx.AbstractLinearOperator

    def __init__(self, Ai, B, C, D):
        assert Ai.in_size() == B.out_size()
        assert Ai.out_size() == C.in_size()
        assert B.in_size() == D.in_size()
        assert C.out_size() == D.out_size()

        AiB = jnp.array([Ai.mv(x) for x in B.as_matrix().T]).T
        CAiB = C.as_matrix() @ AiB
        schur = D.as_matrix() - CAiB
        self.schuri = lx.MatrixLinearOperator(jnp.linalg.pinv(schur))
        self.AiB = lx.MatrixLinearOperator(AiB)
        self.Ai = Ai
        self.B = B
        self.C = C
        self.D = D

    def mv(self, vector):
        """Matrix vector product."""
        # [AA DB] [X1] = [AAX1 + BBX2]
        # [CC DD] [X2] = [CCX1 + DDX2]
        # with
        # AA = Ai + Ai @ B @ schuri @ C @ Ai    # noqa: E800
        # BB = -Ai @ B @ schuri                 # noqa: E800
        # CC = -schuri @ C @ Ai                 # noqa: E800
        # DD = schuri                           # noqa: E800
        X1 = vector[: self.Ai.in_size()]
        X2 = vector[self.Ai.in_size() :]
        Aix1 = self.Ai.mv(X1)
        Amv = Aix1 + self.AiB.mv(self.schuri.mv(self.C.mv(Aix1)))
        Bmv = -self.AiB.mv(self.schuri.mv(X2))
        Cmv = -self.schuri.mv(self.C.mv(Aix1))
        Dmv = self.schuri.mv(X2)
        Y1 = Amv + Bmv
        Y2 = Cmv + Dmv
        return jnp.concatenate([Y1, Y2])

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv, out_axes=-1)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.Ai.in_size() + self.D.in_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.Ai.out_size() + self.D.out_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(InverseBorderedOperator)
@lx.is_diagonal.register(InverseBorderedOperator)
@lx.is_tridiagonal.register(InverseBorderedOperator)
@lx.is_symmetric.register(BorderedOperator)
@lx.is_diagonal.register(BorderedOperator)
@lx.is_tridiagonal.register(BorderedOperator)
def _(operator):
    return False


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
