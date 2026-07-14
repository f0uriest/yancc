"""Linear algebra helpers."""

import functools
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np


def _where(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    # need this bc type checkers are stuuuuuupid
    return jnp.where(a, b, c)


def _banded_row_scale(p, q, A, periodic):
    """Row-equilibration factors for a matrix in banded storage.

    Returns ``s`` of shape ``(n,)`` with ``s[i] = 1 / max_j |A[i, j]|`` (and
    ``s[i] = 1`` for an all-zero row). Scaling each dense row ``i`` by ``s[i]``
    pushes the matrix toward unit-magnitude rows, which improves the
    conditioning of the (non-pivoted) elimination without changing the
    solution. For ``periodic=True`` the wrap-around entries are folded into the
    row maxima via the cyclic row index.
    """
    H, n = A.shape
    r_idx = jnp.arange(H)[:, None]
    j_idx = jnp.arange(n)[None, :]
    # Dense row index of each banded entry; entries off the matrix are padding.
    i_linear = j_idx + r_idx - q
    in_band = (i_linear >= 0) & (i_linear < n)
    i_dense = i_linear % n if periodic else jnp.clip(i_linear, 0, n - 1)
    valid = in_band | periodic
    contrib = _where(valid, jnp.abs(A), jnp.zeros_like(A))
    row_max = jnp.zeros(n, dtype=A.dtype).at[i_dense.flatten()].max(contrib.flatten())
    return _where(row_max > 0, 1.0 / row_max, jnp.ones_like(row_max))


def _scale_banded_rows(p, q, A, s, periodic):
    """Scale dense row ``i`` of a banded matrix by ``s[i]`` in banded storage."""
    H, n = A.shape
    r_idx = jnp.arange(H)[:, None]
    j_idx = jnp.arange(n)[None, :]
    i_linear = j_idx + r_idx - q
    i_dense = i_linear % n if periodic else jnp.clip(i_linear, 0, n - 1)
    return A * s[i_dense]


class BorderedOperator(lx.AbstractLinearOperator):
    """Operator for a bordered matrix.

    [A B]
    [C 0]
    """

    A: lx.AbstractLinearOperator
    B: lx.AbstractLinearOperator
    C: lx.AbstractLinearOperator

    def __init__(self, A, B, C):
        assert A.out_size() == B.out_size()
        assert A.in_size() == C.in_size()
        self.A = A
        self.B = B
        self.C = C

    def mv(self, vector):
        """Matrix vector product."""
        # [A B] [X1] = [AX1 + BX2]
        # [C 0] [X2] = [CX1      ]
        X1 = vector[: self.A.in_size()]
        X2 = vector[self.A.in_size() :]
        Y1 = self.A.mv(X1) + self.B.mv(X2)
        Y2 = self.C.mv(X1)
        return jnp.concatenate([Y1, Y2])

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv, out_axes=-1)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.A.in_size() + self.B.in_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.A.out_size() + self.C.out_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return BorderedOperator(self.A.T, self.C.T, self.B.T)


class InverseBorderedOperator(lx.AbstractLinearOperator):
    """(Pseudo) Inverse of a bordered matrix, using already inverted A.

    Assumes CA = AB = 0
    """

    Ai: lx.AbstractLinearOperator
    B: lx.AbstractLinearOperator
    C: lx.AbstractLinearOperator
    CBi: lx.AbstractLinearOperator

    def __init__(self, Ai, B, C):
        assert Ai.in_size() == B.out_size()
        assert Ai.out_size() == C.in_size()

        self.CBi = lx.MatrixLinearOperator(
            jnp.linalg.pinv(C.as_matrix() @ B.as_matrix())
        )
        self.Ai = Ai
        self.B = B
        self.C = C

    def mv(self, vector):
        """Matrix vector product."""
        X1 = vector[: self.Ai.in_size()]
        X2 = vector[self.Ai.in_size() :]
        cbic_x1 = self.CBi.mv(self.C.mv(X1))
        z11 = X1 - self.B.mv(cbic_x1)
        Az11 = self.Ai.mv(z11)
        z11 = Az11 - self.B.mv(self.CBi.mv(self.C.mv(Az11)))
        z12 = self.B.mv(self.CBi.mv(X2))
        Y1 = z11 + z12
        Y2 = cbic_x1
        return jnp.concatenate([Y1, Y2])

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv, out_axes=-1)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.Ai.in_size() + self.B.in_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.Ai.out_size() + self.C.out_size(),),
            dtype=jnp.array(1.0).dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return InverseBorderedOperator(
            self.Ai.T,
            self.C.T,
            self.B.T,
        )


class TransposedLinearOperator(lx.AbstractLinearOperator):
    """Transpose of another linear operator using jax.transpose"""

    operator: lx.AbstractLinearOperator

    def __init__(self, operator):
        self.operator = operator

    def mv(self, vector):
        """Matrix vector product with transposed operator."""
        x = jax.tree.map(jnp.ones_like, self.operator.in_structure())
        return jax.linear_transpose(self.operator.mv, x)(vector)[0]

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        return self.operator.as_matrix().T

    def in_structure(self):
        """Pytree structure of expected input."""
        return self.operator.out_structure()

    def out_structure(self):
        """Pytree structure of expected output."""
        return self.operator.in_structure()

    def transpose(self):
        """Transpose of the operator."""
        return self.operator

    def __getattr__(self, attr):
        return getattr(self.operator, attr)


@lx.is_symmetric.register(InverseBorderedOperator)
@lx.is_diagonal.register(InverseBorderedOperator)
@lx.is_tridiagonal.register(InverseBorderedOperator)
@lx.is_symmetric.register(BorderedOperator)
@lx.is_diagonal.register(BorderedOperator)
@lx.is_tridiagonal.register(BorderedOperator)
@lx.is_symmetric.register(TransposedLinearOperator)
@lx.is_diagonal.register(TransposedLinearOperator)
@lx.is_tridiagonal.register(TransposedLinearOperator)
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


@functools.partial(jax.jit, static_argnames=["p", "q"])
@functools.partial(jnp.vectorize, signature="(m,n)->(n,n)", excluded=(0, 1))
def banded_to_dense(p, q, A):
    """Convert from banded representation to dense.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    A : jax.Array, shape(...,p+q+1, N)
        Matrix in banded storage format.

    Returns
    -------
    A : jax.Array, shape(...,N,N)
        Matrix in dense format.
    """
    H, n = A.shape
    assert n >= max(p, q), "invalid bandwidth"

    # Create coordinate grids for the input banded matrix
    r = jnp.arange(H)[:, None]
    j = jnp.arange(n)[None, :]

    # Calculate the target row indices in the dense matrix
    i = (j + r - q) % n

    # Initialize a dense matrix of zeros and scatter the banded elements.
    M = jnp.zeros((n, n), dtype=A.dtype)
    return M.at[i, j].set(A)


@functools.partial(jax.jit, static_argnames=["p", "q"])
@functools.partial(jnp.vectorize, signature="(n,n)->(m,n)", excluded=(0, 1))
def dense_to_banded(p, q, A):
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
    n = A.shape[0]
    assert n >= max(p, q), "invalid bandwidth"
    H = p + q + 1  # Total height of the banded format

    # Create coordinate grids for the output banded matrix
    r = jnp.arange(H)[:, None]
    j = jnp.arange(n)[None, :]

    # Calculate the corresponding row indices in the dense matrix.
    # The modulo % n seamlessly wraps the indices for the periodic corners.
    i = (j + r - q) % n

    # Gather elements natively
    return A[i, j]


@functools.partial(jax.jit, static_argnames=("p", "q", "unroll", "equilibrate"))
@functools.partial(
    jnp.vectorize,
    signature="(m,n)->(m,n),(n)",
    excluded=(0, 1, "unroll", "pivot_tol", "equilibrate"),
)
def lu_factor_banded(p, q, A, *, unroll=None, pivot_tol=0.0, equilibrate=False):
    """LU factorization of banded matrix in banded storage format.

    Note: does not use row pivoting, so may be unstable unless A is diagonally
    dominant. Pass ``pivot_tol > 0`` to clamp tiny pivots away from zero
    ("static pivoting"), which prevents blow-up from near-zero pivots at the
    cost of a slightly inexact factorization (recoverable via iterative
    refinement). For ill-conditioned matrices, row equilibration also helps.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    A : jax.Array, shape(...,p+q+1,N)
        Matrix in banded format.
    pivot_tol : float
        If positive, any pivot with magnitude below ``pivot_tol`` is replaced by
        ``+/- pivot_tol`` (matching its sign) before forming the multipliers.
        Default 0.0 reproduces the unmodified non-pivoted factorization.
    equilibrate : bool
        If True, scale each row to unit max-magnitude before factoring. The
        scaling is stored in the returned tuple and applied automatically by
        ``lu_solve_banded``.

    Returns
    -------
    lu : jax.Array, shape(...,p+q+1,N)
        LU factorized matrix. Upper triangle is U, lower triangle is L (unit diagonal
        is assumed.)
    s : jax.Array, shape(...,N)
        Row-equilibration factors (ones when ``equilibrate=False``).
    """
    n = A.shape[1]
    assert p <= n
    assert q <= n
    assert A.shape[0] == (p + q + 1)

    s = (
        _banded_row_scale(p, q, A, periodic=False)
        if equilibrate
        else jnp.ones(n, dtype=A.dtype)
    )
    if equilibrate:
        A = _scale_banded_rows(p, q, A, s, periodic=False)

    # Pad A along the columns by q.
    # This acts as a safe "run-off" area for fixed-size slices near the right edge.
    A_padded = jnp.pad(A, ((0, 0), (0, q)))

    def kloop(k, A_acc):
        # --- 1. Vectorized L-update ---
        pivot = A_acc[q, k]

        # Static pivoting: clamp |pivot| up to pivot_tol (no-op when pivot_tol=0).
        # The clamped value is written back to the diagonal so the U factor used
        # by lu_solve_banded stays consistent with the multipliers below.
        pivot = _where(
            jnp.abs(pivot) < pivot_tol,
            jnp.where(pivot < 0, -pivot_tol, pivot_tol),
            pivot,
        )
        A_acc = A_acc.at[q, k].set(pivot)

        # Extract L multipliers: shape (p,)
        l_vec = jax.lax.dynamic_slice_in_dim(A_acc[:, k], q + 1, p, axis=0) / pivot

        # Update the L-column
        A_acc = A_acc.at[q + 1 : q + 1 + p, k].set(l_vec)

        # --- 2. Vectorized Rank-1 U-update ---
        # Since q is static and typically small, we unroll the loop in Python.
        # This provides XLA with constant, predictable row slice bounds.
        for j_off in range(1, q + 1):
            target_col = k + j_off

            # U-multiplier for this specific column
            u_val = A_acc[q - j_off, target_col]

            # The static starting row for this column's update
            start_row = q + 1 - j_off

            # Vectorized subtraction for the p elements
            A_acc = A_acc.at[start_row : start_row + p, target_col].add(-l_vec * u_val)

        return A_acc

    # Execute the main loop over pivot columns
    A_padded = jax.lax.fori_loop(0, n - 1, kloop, A_padded, unroll=unroll)

    # Slice back to the original mathematical shape
    return A_padded[:, :n], s


@functools.partial(jax.jit, static_argnames=("p", "q", "unroll", "trans"))
@functools.partial(
    jnp.vectorize, signature="(m,n),(n)->(n)", excluded=(0, 1, "unroll", "trans")
)
def _lu_solve_banded(p, q, lu, b, *, unroll=None, trans=False):
    n = lu.shape[1]
    assert p <= n
    assert q <= n
    assert lu.shape[0] == (p + q + 1)

    if trans:
        # forward: U^T y = b  (pad left by q so row a-q stays in range)
        b_padded = jnp.pad(b, (q, 0))

        def utT_forward(a, acc):
            window = jax.lax.dynamic_slice_in_dim(acc, a, q, axis=0)  # rows a-q .. a-1
            ya = (acc[a + q] - jnp.sum(lu[0:q, a] * window)) / lu[q, a]
            return acc.at[a + q].set(ya)

        y = jax.lax.fori_loop(0, n, utT_forward, b_padded, unroll=unroll)[q:]

        # backward: L^T x = y  (pad right by p so row a+p stays in range)
        y_padded = jnp.pad(y, (0, p))

        def ltT_backward(k, acc):
            a = n - 1 - k
            window = jax.lax.dynamic_slice_in_dim(
                acc, a + 1, p, axis=0
            )  # rows a+1 .. a+p
            xa = acc[a] - jnp.sum(lu[q + 1 : q + 1 + p, a] * window)
            return acc.at[a].set(xa)

        return jax.lax.fori_loop(0, n, ltT_backward, y_padded, unroll=unroll)[:n]

    # ==========================================
    # 1. Forward Substitution: Ly = b
    # ==========================================
    # Pad RIGHT with 'p' zeros to prevent clamping.
    b_padded = jnp.pad(b, (0, p))

    def forward_step(j, b_acc):
        y_j = b_acc[j]
        l_vec = lu[q + 1 : q + 1 + p, j]

        # Grab window of size p starting at j + 1
        window = jax.lax.dynamic_slice_in_dim(b_acc, j + 1, p, axis=0)

        updated_window = window - l_vec * y_j

        return jax.lax.dynamic_update_slice_in_dim(b_acc, updated_window, j + 1, axis=0)

    y_padded = jax.lax.fori_loop(0, n, forward_step, b_padded, unroll=unroll)
    y = y_padded[:n]

    # ==========================================
    # 2. Backward Substitution: Ux = y
    # ==========================================
    # Pad LEFT with 'q' zeros to prevent clamping.
    y_padded_back = jnp.pad(y, (q, 0))

    def backward_step(k, x_acc):
        j = n - 1 - k
        diag_idx = j + q

        xj_val = x_acc[diag_idx] / lu[q, j]
        x_acc = x_acc.at[diag_idx].set(xj_val)

        u_vec = lu[0:q, j]

        # Grab window of size q ending exactly before the diagonal
        window = jax.lax.dynamic_slice_in_dim(x_acc, j, q, axis=0)

        updated_window = window - u_vec * xj_val

        return jax.lax.dynamic_update_slice_in_dim(x_acc, updated_window, j, axis=0)

    x_padded = jax.lax.fori_loop(0, n, backward_step, y_padded_back, unroll=unroll)

    return x_padded[q:]


def lu_solve_banded(p, q, lu_factors, b, *, unroll=None, trans=False):
    """Solve a linear system with a pre-factored banded matrix in banded storage format.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    lu_factors : tuple of jax.Array
        Output from ``lu_factor_banded``. The row-equilibration scaling stored
        in the tuple is applied to ``b`` automatically.
    b : jax.Array, shape(...,N)
        RHS vector.
    trans : bool
        If True, solve ``A^T x = b`` instead of ``A x = b`` using the same factors.

    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    lu, s = lu_factors
    if trans:
        # A = S^-1 A_scaled (S = diag(s) is the row equilibration), so
        # A^-T b = S (A_scaled^-T b): scale the output by s, not the input.
        return s * _lu_solve_banded(p, q, lu, b, unroll=unroll, trans=True)
    return _lu_solve_banded(p, q, lu, s * b, unroll=unroll)


@functools.partial(jax.jit, static_argnames=("p", "q", "unroll", "equilibrate"))
@functools.partial(
    jnp.vectorize,
    signature="(m,n),(n)->(n)",
    excluded=(0, 1, "unroll", "pivot_tol", "equilibrate"),
)
def solve_banded(p, q, A, b, *, unroll=None, pivot_tol=0.0, equilibrate=False):
    """Solve a linear system with a banded matrix in banded storage format.

    Note: does not use row pivoting, so may be unstable unless A is diagonally
    dominant. Pass ``equilibrate=True`` and/or ``pivot_tol > 0`` to improve
    robustness on poorly-scaled or nearly-singular matrices.

    Parameters
    ----------
    p, q: int
        Lower and Upper bandwidth.
    A : jax.Array, shape(...,p+q+1,N)
        Matrix in banded format.
    b : jax.Array, shape(...,N)
        RHS vector.
    pivot_tol : float
        If positive, clamp pivots smaller than this in magnitude (see
        ``lu_factor_banded``).
    equilibrate : bool
        If True, scale each row to unit max-magnitude before factoring (and
        scale ``b`` to match). Row scaling does not change the solution but
        improves conditioning of the non-pivoted elimination.

    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    lu = lu_factor_banded(
        p, q, A, unroll=unroll, pivot_tol=pivot_tol, equilibrate=equilibrate
    )
    return lu_solve_banded(p, q, lu, b, unroll=unroll)


@functools.partial(jax.jit, static_argnames=("p", "q", "unroll", "equilibrate"))
@functools.partial(
    jnp.vectorize,
    signature="(k,n)->(l,n),(n),(n,m),(m,m),(n)",
    excluded=(0, 1, "unroll", "pivot_tol", "equilibrate"),
)
@jax.named_call
def lu_factor_banded_periodic(
    p, q, A, *, unroll=None, pivot_tol=0.0, equilibrate=False
):
    """LU factorization of periodic banded matrix in dense storage format.

    Note: does not use row pivoting, so may be unstable unless A is diagonally
    dominant. Pass ``equilibrate=True`` and/or ``pivot_tol > 0`` to improve
    robustness on poorly-scaled or nearly-singular matrices.

    Parameters
    ----------
    p, q : int
        Lower and upper bandwidth of A
    A : jax.Array, shape(...,p+q+1,N)
        Matrix in banded format.
    pivot_tol : float
        If positive, clamp pivots smaller than this in magnitude (see
        ``lu_factor_banded``).
    equilibrate : bool
        If True, scale each row to unit max-magnitude before factoring. Row
        scaling does not change the solution but improves conditioning of the
        non-pivoted elimination. The scaling is stored in the returned factors
        and applied automatically by ``lu_solve_banded_periodic``.

    Returns
    -------
    lu : jax.Array, shape(...,N,N)
        LU factorized matrix. Upper triangle is U, lower triangle is L (unit diagonal
        is assumed.)
    piv : jax.Array, shape(...,N)
        Pivots (only used for the small dense fallback).
    BUschur : jax.Array, shape(...,N, 2*r+1)
        Additional matrix for solving the periodic part
    V : jax.Array, shape(...,2*r+1, 2*r+1)
        Additional matrix for solving the periodic part. Only the ``2*r+1``
        wrap-around columns of the capacitance solution are nonzero, so just
        those columns are stored (see ``lu_solve_banded_periodic``).
    s : jax.Array, shape(...,N)
        Row-equilibration factors (ones when ``equilibrate=False``).
    """
    r = p + q
    H, n = A.shape
    ones = jnp.ones(n, dtype=A.dtype)
    if r == 0:  # diagonal, trivial
        return (
            A,
            jnp.arange(n).astype(jnp.int32),
            jnp.zeros((n, r)),
            jnp.zeros((r, r)),
            ones,
        )
    if n <= r:
        # below is incorrect, so just use dense solution
        A = banded_to_dense(p, q, A)
        lu, piv = jax.scipy.linalg.lu_factor(A)
        return lu, piv, jnp.zeros((n, r)), jnp.zeros((r, r)), ones

    # ---------------------------------------------------------
    # 1. Isolate the strictly banded part & identify wrap-arounds
    # ---------------------------------------------------------
    r_idx = jnp.arange(H)[:, None]
    j_idx = jnp.arange(n)[None, :]

    # Calculate virtual row indices to find wrap-around elements
    i_linear = j_idx + r_idx - q
    is_wrap = (i_linear < 0) | (i_linear >= n)
    i_cyclic = i_linear % n

    # A_band is the strictly banded part (wrap-around elements zeroed out)
    A_band = _where(is_wrap, jnp.array(0.0), A)

    # Row equilibration (no-op when equilibrate=False). Scaling the full-system
    # rows by s scales A_band and the low-rank columns U identically, leaving
    # the capacitance matrix (and hence the solution) unchanged; see
    # lu_solve_banded_periodic, which applies s to the RHS.
    s = _banded_row_scale(p, q, A, periodic=True) if equilibrate else ones
    A_band = _scale_banded_rows(p, q, A_band, s, periodic=True)

    # ---------------------------------------------------------
    # 2. Construct U and V^T for the low-rank update
    # ---------------------------------------------------------
    k_dim = p + q  # Rank of the update

    # Construct U (n x k_dim): Indicator matrix for the cyclic rows
    U = jnp.zeros((n, k_dim), dtype=A.dtype)
    # First q columns map to the bottom-left corner (rows n-q to n-1)
    U = U.at[n - q + jnp.arange(q), jnp.arange(q)].set(1.0)
    # Next p columns map to the top-right corner (rows 0 to p-1)
    U = U.at[jnp.arange(p), q + jnp.arange(p)].set(1.0)
    # Apply the same row scaling to the low-rank columns.
    U = s[:, None] * U

    # Construct V^T (k_dim x n): Contains the actual wrap-around values
    # Map the cyclic rows to the k_dim coordinate space
    k_idx = _where(i_cyclic >= n - q, i_cyclic - (n - q), q + i_cyclic)

    # We use scatter-add to build V^T while keeping static shapes.
    # Non-wrap elements add 0.0 to the 0-th index (harmless).
    safe_k = _where(is_wrap, k_idx, jnp.array(0)).flatten()
    safe_j = jnp.broadcast_to(j_idx, (H, n)).flatten()
    safe_vals = _where(is_wrap, A, jnp.array(0.0)).flatten()

    V_T = jnp.zeros((k_dim, n), dtype=A.dtype)
    V_T = V_T.at[safe_k, safe_j].add(safe_vals)

    lu_factors = lu_factor_banded(p, q, A_band, unroll=unroll, pivot_tol=pivot_tol)
    lu, _ = lu_factors
    # Z = inv(A_band), Z_U = Z@U
    Z_U = lu_solve_banded(p, q, lu_factors, U.T, unroll=unroll).T

    # Compute the capacitance matrix C = I + V^T @ Z_U
    C = jnp.eye(k_dim, dtype=A.dtype) + jnp.matmul(V_T, Z_U)
    # V^T is nonzero only in the wrap-around columns (the first q and last p),
    # so C^{-1} V^T is too. Solve/store just those k_dim columns: the dropped
    # columns contribute nothing to V^T @ b in lu_solve_banded_periodic.
    wrap_cols = jnp.concatenate([jnp.arange(q), n - p + jnp.arange(p)])
    # Solve the small dense system: C @ Y = (V^T restricted to wrap columns)
    Y = jnp.linalg.solve(C, V_T[:, wrap_cols])
    piv = jnp.arange(n)  # dummy pivots for now
    return lu, piv, Z_U, Y, s


@functools.partial(jax.jit, static_argnames=("p", "q", "unroll"))
@jax.named_call
def lu_solve_banded_periodic(p, q, lu, b, *, unroll=None):
    """Solve a periodic banded linear system with matrix pre-factored.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    p, q : int
        Lower and upper bandwidth of A
    lu : tuple of jax.Array
        Output from ``lu_factor_banded_periodic``
    b : jax.Array, shape(...,N)
        RHS vector.


    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    lu, piv, Z_U, Y, s = lu
    return _lu_solve_banded_periodic(p, q, lu, piv, Z_U, Y, s, b, unroll)


@functools.partial(
    jnp.vectorize, signature="(k,n),(n),(n,m),(m,m),(n),(n)->(n)", excluded=(0, 1, 8)
)
def _lu_solve_banded_periodic(p, q, lu, piv, Z_U, Y, s, b, unroll):
    nn = b.shape[-1]
    r = p + q
    if r == 0:  # diagonal
        return b / lu[0]
    if nn <= r:  # use dense method
        return jax.scipy.linalg.lu_solve((lu, piv), b)
    # s applies the row equilibration chosen at factor time (ones if disabled).
    Binvb = _lu_solve_banded(p, q, lu, s * b, unroll=unroll)
    # Y holds only the wrap-around columns of C^{-1} V^T (see factor), so contract
    # against the matching entries of Binvb: the first q and last p.
    wrap_cols = jnp.concatenate([jnp.arange(q), nn - p + jnp.arange(p)])
    return Binvb - Z_U @ (Y @ Binvb[wrap_cols])


@functools.partial(jax.jit, static_argnames=("p", "q", "unroll"))
@functools.partial(jnp.vectorize, signature="(k,n),(n)->(n)", excluded=(0, 1, "unroll"))
def solve_banded_periodic(p, q, A, b, *, unroll=None):
    """Solve a periodic banded linear system.

    Note: does not use any pivoting so may be unstable unless A is diagonally dominant.

    Parameters
    ----------
    p, q : int
        Lower and upper bandwidth of A
    A : jax.Array, shape(...,p+q+1,N)
        Matrix in banded format.
    b : jax.Array, shape(...,N)
        RHS vector.


    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    lu_schur_v = lu_factor_banded_periodic(p, q, A, unroll=unroll)
    return lu_solve_banded_periodic(p, q, lu_schur_v, b, unroll=unroll)


# ---------------------------------------------------------------------------
# Block cyclic reduction (CR) for banded / block-tridiagonal systems
# ---------------------------------------------------------------------------
# An alternative to the banded LU solvers above. A banded matrix (lower bandwidth
# ``p``, upper bandwidth ``q``) is reinterpreted as block-tridiagonal with block
# size ``b = max(p, q)`` then factored by cyclic reduction.


def _cr_mm(x, y):
    return jnp.einsum("Bmij,Bmjk->Bmik", x, y)


def _cr_mv(x, y):
    return jnp.einsum("Bmij,Bmj->Bmi", x, y)


def _cr_level_maps(M):
    """Static (numpy) index maps for one CR reduction level of M blocks.

    Eliminates odd-indexed blocks, keeps ceil(M/2) even survivors. When M is odd the
    wrap pair (blocks 0 and M-1) become adjacent survivors, so that one coupling passes
    through directly; all others get the Schur update.
    """
    surv = np.arange(0, M, 2)
    elim = np.arange(1, M, 2)
    lp = (surv - 1) % M
    rp = (surv + 1) % M
    left_is_e = lp % 2 == 1
    right_is_e = rp % 2 == 1
    left_epos = np.where(left_is_e, (lp - 1) // 2, 0)
    right_epos = np.where(right_is_e, (rp - 1) // 2, 0)
    sposL = ((elim - 1) % M) // 2
    sposR = ((elim + 1) % M) // 2
    return dict(
        surv=surv,
        elim=elim,
        lp=lp,
        rp=rp,
        left_is_e=left_is_e,
        right_is_e=right_is_e,
        left_epos=left_epos,
        right_epos=right_epos,
        sposL=sposL,
        sposR=sposR,
    )


def _cr_reduce(D, L, U):
    """Core cyclic reduction of a batch of block-tridiagonal matrices.

    D, L, U : (B, m, b, b) (L[:, 0], U[:, m-1] are the periodic wrap corners, zero for
    a non-periodic system). Returns (levels, root_inv), where each level holds the
    factors to reduce the rhs and back-substitute the eliminated blocks, and root_inv
    inverts the final single block.
    """
    levels = []
    M = D.shape[1]
    while M > 1:
        mp = _cr_level_maps(M)
        surv, elim = mp["surv"], mp["elim"]
        le = jnp.asarray(mp["left_is_e"])[None, :, None, None]
        re = jnp.asarray(mp["right_is_e"])[None, :, None, None]
        # gather along the block axis (axis 1); jnp.take keeps pyright happy
        Dinv_E = jnp.linalg.inv(jnp.take(D, elim, axis=1))
        D_p = jnp.take(D, surv, axis=1)
        L_p, U_p = jnp.take(L, surv, axis=1), jnp.take(U, surv, axis=1)
        L_lp, U_lp = jnp.take(L, mp["lp"], axis=1), jnp.take(U, mp["lp"], axis=1)
        L_rp, U_rp = jnp.take(L, mp["rp"], axis=1), jnp.take(U, mp["rp"], axis=1)
        Dinv_l = jnp.take(Dinv_E, mp["left_epos"], axis=1)
        Dinv_r = jnp.take(Dinv_E, mp["right_epos"], axis=1)
        # alpha/beta are zero where the neighbor is a survivor (wrap, odd M)
        alpha = jnp.where(le, _cr_mm(L_p, Dinv_l), 0.0)
        beta = jnp.where(re, _cr_mm(U_p, Dinv_r), 0.0)
        D_new = D_p - _cr_mm(alpha, U_lp) - _cr_mm(beta, L_rp)
        L_new = jnp.where(le, -_cr_mm(alpha, L_lp), L_p)
        U_new = jnp.where(re, -_cr_mm(beta, U_rp), U_p)
        L_E, U_E = jnp.take(L, elim, axis=1), jnp.take(U, elim, axis=1)
        levels.append((alpha, beta, Dinv_E, L_E, U_E, mp))
        D, L, U = D_new, L_new, U_new
        M = len(surv)
    root_inv = jnp.linalg.inv(D + L + U)
    return levels, root_inv


def _cr_forward_solve(levels, root_inv, rhs) -> jax.Array:
    """Apply A^-1 to rhs : (B, m, b) given (levels, root_inv)."""
    saved = []
    b = rhs
    for alpha, beta, _Dinv_E, _L_E, _U_E, mp in levels:
        b_new = (
            jnp.take(b, mp["surv"], axis=1)
            - _cr_mv(alpha, jnp.take(b, mp["lp"], axis=1))
            - _cr_mv(beta, jnp.take(b, mp["rp"], axis=1))
        )
        saved.append(jnp.take(b, mp["elim"], axis=1))
        b = b_new
    x = _cr_mv(root_inv, b)
    for (_alpha, _beta, Dinv_E, L_E, U_E, mp), b_e in zip(
        reversed(levels), reversed(saved)
    ):
        surv, elim = mp["surv"], mp["elim"]
        M = len(surv) + len(elim)
        x_elim = _cr_mv(
            Dinv_E,
            b_e
            - _cr_mv(L_E, jnp.take(x, mp["sposL"], axis=1))
            - _cr_mv(U_E, jnp.take(x, mp["sposR"], axis=1)),
        )
        # surv/elim are disjoint unique indices -> declare it so the scatter is
        # linear-transposable (needed by the trans=True path)
        full = jnp.zeros((x.shape[0], M) + x.shape[2:], x.dtype)
        full = full.at[:, surv].set(x, unique_indices=True)
        full = full.at[:, elim].set(x_elim, unique_indices=True)
        x = full
    return x


def _cr_row_scale(D, L, U):
    """Per-block-row max-magnitude reciprocal (row equilibration), (B, m, b)."""
    rowmax = jnp.maximum(
        jnp.max(jnp.abs(D), axis=-1),
        jnp.maximum(jnp.max(jnp.abs(L), axis=-1), jnp.max(jnp.abs(U), axis=-1)),
    )
    return jnp.where(rowmax == 0, 1.0, 1.0 / rowmax)


@functools.partial(jax.jit, static_argnames=("equilibrate",))
def cr_block_tridiag_factor(D, L, U, *, equilibrate=False):
    """Cyclic-reduction factorization of a batch of block-tridiagonal matrices.

    Parameters
    ----------
    D, L, U : jax.Array, shape (B, m, b, b)
        Diagonal, sub- and super-diagonal blocks. L[:, 0] and U[:, m-1] are
        the periodic wrap corners (pass zeros for a non-periodic system).
    equilibrate : bool
        If True, scale each block row to unit max-magnitude before factoring.
        Exact (row scaling does not change the solution), improves the unpivoted
        elimination on poorly scaled systems. Stored in the factors and applied
        by cr_block_tridiag_solve.

    Returns
    -------
    factors : tuple (levels, root_inv, s)
        levels and root_inv from the reduction; s is the row scaling (B, m, b)
        (ones when equilibrate=False).
    """
    if equilibrate:
        s = _cr_row_scale(D, L, U)
        D, L, U = D * s[..., None], L * s[..., None], U * s[..., None]
    else:
        s = jnp.ones(D.shape[:-1], D.dtype)
    levels, root_inv = _cr_reduce(D, L, U)
    return levels, root_inv, s


@functools.partial(jax.jit, static_argnames=("trans",))
def cr_block_tridiag_solve(factors, b, *, trans=False) -> jax.Array:
    """Solve A x = b (or A^T x = b if trans) from CR factors.

    Parameters
    ----------
    factors : tuple
        Output of cr_block_tridiag_factor.
    b : jax.Array, shape (B, m, b)
    trans : bool
        If True, solve with A^T. Uses the same stored factors.

    Returns
    -------
    x : jax.Array, shape (B, m, b)
    """
    levels, root_inv, s = factors
    if trans:
        # factors are of M = diag(s) A; A^T x = b  =>  x = s * (M^{-T} b)
        (y,) = jax.linear_transpose(
            lambda r: _cr_forward_solve(levels, root_inv, r), b
        )(b)
        return s * y
    return _cr_forward_solve(levels, root_inv, s * b)


def _banded_to_blocks(Ab, b, p, q):
    """Non-periodic banded storage (B, p+q+1, N) -> blocks (B, m, b, b).

    N must be a multiple of the block size b (which must be >= max(p, q) for the result
    to be block-tridiagonal); wrap/out-of-range entries are zeroed so
    L[:, 0] and U[:, m-1] come out zero.
    """
    N = Ab.shape[-1]
    H = p + q
    m = N // b
    k = jnp.arange(m)[:, None, None]
    a = jnp.arange(b)[None, :, None]
    c = jnp.arange(b)[None, None, :]

    def band_get(row, col) -> jax.Array:
        r = row - col + q
        valid = (r >= 0) & (r <= H) & (col >= 0) & (col < N) & (row >= 0) & (row < N)
        val = cast(jax.Array, Ab[:, jnp.clip(r, 0, H), jnp.clip(col, 0, N - 1)])
        return jnp.where(valid[None], val, 0.0)

    row = k * b + a
    D = band_get(row, k * b + c)  # (B, m, b, b), batch-first
    L = band_get(row, (k - 1) * b + c)
    U = band_get(row, (k + 1) * b + c)
    return D, L, U


def banded_to_block_tridiag(A_banded, p=None, q=None):
    """Extract padded non-periodic block-tridiagonal blocks from banded storage.

    Parameters
    ----------
    A_banded : jax.Array, shape (..., p+q+1, N)
        Banded storage (scipy/LAPACK format). The block size is b = max(p, q), the
        smallest that makes the matrix block-tridiagonal.
    p, q : int, optional
        Lower/upper bandwidth. When both are omitted they default to the symmetric
        bw = (H - 1) // 2 inferred from the storage height H = p + q + 1.

    Returns
    -------
    D, L, U : jax.Array, shape (B, m, b, b)
        Block-tridiagonal blocks with B = prod(leading dims) and m = ceil(N / b). If N
        is not a multiple of b the system is padded to m*b with decoupled identity rows
        (exact; L[:, 0]=U[:, m-1]=0).
    """
    *lead, H, N = A_banded.shape
    if p is None and q is None:
        p = q = (H - 1) // 2
    assert p is not None and q is not None, "pass both p and q, or neither"
    assert H == p + q + 1, "storage height must equal p + q + 1"
    b = max(p, q)
    B = int(np.prod(lead)) if lead else 1
    Ab = A_banded.reshape(B, H, N)
    Npad = -(-N // b) * b
    if Npad != N:
        Ab = jnp.zeros((B, H, Npad), Ab.dtype).at[:, :, :N].set(Ab)
        Ab = Ab.at[:, q, N:].set(1.0)  # appended rows are identity (diagonal at r=q)
    return _banded_to_blocks(Ab, b, p, q)


@functools.partial(jax.jit, static_argnames=("p", "q", "equilibrate"))
def cr_banded_factor(A_banded, p=None, q=None, *, equilibrate=False):
    """Cyclic-reduction factor of a non-periodic banded matrix.

    Parameters
    ----------
    A_banded : jax.Array, shape (..., p+q+1, N)
    p, q : int, optional
        Lower/upper bandwidth. Default to the symmetric bw = (H - 1) // 2.
    equilibrate : bool
        Row-equilibrate before factoring (see cr_block_tridiag_factor).

    Returns
    -------
    factors : tuple
        (levels, root_inv, s) from cr_block_tridiag_factor. The block size and padding
        are recovered from array shapes by cr_banded_solve.
    """
    D, L, U = banded_to_block_tridiag(A_banded, p, q)
    return cr_block_tridiag_factor(D, L, U, equilibrate=equilibrate)


@functools.partial(jax.jit, static_argnames=("trans",))
def cr_banded_solve(factors, b, *, trans=False):
    """Solve a non-periodic banded system from cr_banded_factor factors.

    Parameters
    ----------
    factors : tuple
        Output of cr_banded_factor.
    b : jax.Array, shape (..., N)
        RHS (original, unpadded size).
    trans : bool
        Solve with A^T if True.

    Returns
    -------
    x : jax.Array, shape (..., N)
    """
    root_inv = factors[1]
    blk = root_inv.shape[-1]  # block size b, static
    *lead, Nin = b.shape
    B = int(np.prod(lead)) if lead else 1
    Npad = -(-Nin // blk) * blk
    m = Npad // blk
    rhs = b.reshape(B, Nin)
    if Npad != Nin:
        rhs = jnp.zeros((B, Npad), rhs.dtype).at[:, :Nin].set(rhs)
    rhs = rhs.reshape(B, m, blk)  # (B, m, blk), batch-first
    x = cr_block_tridiag_solve(factors, rhs, trans=trans)
    x = cast(jax.Array, x).reshape(B, Npad)[:, :Nin]
    return x.reshape(*lead, Nin) if lead else x.reshape(Nin)


def _cr_wrap_lowrank(A, p, q):
    """Isolate the strictly-banded part and build the wrap low-rank update.

    Mirrors the construction in lu_factor_banded_periodic. A : (B, p+q+1, N) banded
    storage. Returns A_band (wrap zeroed), U : (N, k) indicator, VT : (B, k, N) wrap
    values, and the wrap_cols / wrap_rows index arrays (k = p + q), where
    A_periodic = A_band + U @ VT per batch.
    """
    B, H, N = A.shape
    dtype = A.dtype
    r_idx = jnp.arange(H)[:, None]
    j_idx = jnp.arange(N)[None, :]
    i_linear = j_idx + r_idx - q
    is_wrap = (i_linear < 0) | (i_linear >= N)
    i_cyclic = i_linear % N

    A_band = jnp.where(is_wrap[None], 0.0, A)

    k_dim = p + q
    U = jnp.zeros((N, k_dim), dtype)
    U = U.at[N - q + jnp.arange(q), jnp.arange(q)].set(1.0)  # bottom-left rows
    U = U.at[jnp.arange(p), q + jnp.arange(p)].set(1.0)  # top-right rows

    k_map = jnp.where(i_cyclic >= N - q, i_cyclic - (N - q), q + i_cyclic)
    safe_k = cast(jax.Array, jnp.where(is_wrap, k_map, 0)).reshape(-1)
    safe_j = jnp.broadcast_to(j_idx, (H, N)).reshape(-1)
    vals = cast(jax.Array, jnp.where(is_wrap[None], A, 0.0)).reshape(B, -1)
    VT = jnp.zeros((B, k_dim, N), dtype).at[:, safe_k, safe_j].add(vals)

    # rows carrying a 1 in U (in k-column order), for the transpose correction
    wrap_rows = jnp.concatenate([N - q + jnp.arange(q), jnp.arange(p)])
    # columns of VT that are nonzero (bottom-left cols 0..q-1, top-right N-p..N-1)
    wrap_cols = jnp.concatenate([jnp.arange(q), N - p + jnp.arange(p)])
    return A_band, U, VT, wrap_cols, wrap_rows


def _cr_solve_columns(bf, M, trans=False):
    """Apply A^{-1} (or A^{-T}) to each column of M : (B, N, k)."""
    cols = jnp.moveaxis(M, -1, 0)  # (k, B, N)
    out = jax.vmap(lambda col: cr_banded_solve(bf, col, trans=trans))(cols)
    return jnp.moveaxis(out, 0, -1)  # (B, N, k)


@functools.partial(jax.jit, static_argnames=("p", "q", "equilibrate"))
def cr_banded_periodic_factor(A_banded, p=None, q=None, *, equilibrate=False):
    """Cyclic-reduction factor of a periodic banded matrix.

    Parameters
    ----------
    A_banded : jax.Array, shape (..., p+q+1, N)
    p, q : int, optional
        Lower/upper bandwidth. Default to the symmetric bw = (H - 1) // 2.
    equilibrate : bool
        Row-equilibrate the full periodic system before factoring.

    Returns
    -------
    factors : tuple
        For N > p + q, the 8-tuple (sub, Z_U, Y, Z_Vt, Yt, s, wrap_cols, wrap_rows)
        the non-periodic sub-factors plus the forward/transpose capacitance pieces.
        For N <= p + q (where the wrap corners overlap and the low-rank split is
        invalid) the 2-tuple (lu, piv) of a batched dense LU; cr_banded_periodic_solve
        dispatches on the tuple length.
    """
    *lead, H, N = A_banded.shape
    if p is None and q is None:
        p = q = (H - 1) // 2
    assert p is not None and q is not None, "pass both p and q, or neither"
    assert H == p + q + 1, "storage height must equal p + q + 1"
    B = int(np.prod(lead)) if lead else 1
    A = A_banded.reshape(B, H, N)
    dtype = A.dtype

    if N <= p + q:
        # The rank-(p+q) wrap update needs disjoint top-right and bottom-left
        # indicator rows; they overlap once N <= p + q and the Woodbury correction
        # is then wrong. Fall back to a dense LU exactly as lu_factor_banded_periodic
        # does for n <= p+q. The 2-tuple return signals the dense path to
        # cr_banded_periodic_solve. (N is static, so this is a compile-time branch
        # and large-N factors never carry the dense matrix.)
        A_dense = jax.vmap(lambda M: banded_to_dense(p, q, M))(A)  # (B, N, N)
        return jax.vmap(jax.scipy.linalg.lu_factor)(A_dense)  # (lu, piv)

    # row equilibration of the full periodic system (scales A_band and U alike)
    if equilibrate:
        s = jax.vmap(lambda M: _banded_row_scale(p, q, M, periodic=True))(A)  # (B,N)
    else:
        s = jnp.ones((B, N), dtype)

    A_band, U, VT, wrap_cols, wrap_rows = _cr_wrap_lowrank(A, p, q)
    A_band = jax.vmap(lambda M, sc: _scale_banded_rows(p, q, M, sc, periodic=False))(
        A_band, s
    )
    Us = s[:, :, None] * U[None]  # (B, N, k), row-scaled indicator

    sub = cr_banded_factor(A_band, p, q, equilibrate=False)

    # forward capacitance: C = I + VT @ (M^-1 Us),  Y = C^-1 VT[:, wrap_cols].
    # VT is nonzero only in wrap_cols, so contracting Y against Binvb[wrap_cols]
    # reproduces the full C^-1 VT Binvb (see lu_solve_banded_periodic).
    Z_U = _cr_solve_columns(sub, Us)  # (B, N, k)
    C = jnp.eye(p + q, dtype=dtype)[None] + jnp.einsum("bkn,bnl->bkl", VT, Z_U)
    Y = jnp.linalg.solve(C, VT[:, :, wrap_cols])  # (B, k, len(wrap_cols))

    # transpose capacitance: (diag(s) A_periodic)^T = M^T + V @ Us^T (V = VT^T).
    # The Us wrap values are folded into sel at solve time, so Yt is plain Ct^{-1}.
    V = jnp.swapaxes(VT, -1, -2)  # (B, N, k)
    Z_Vt = _cr_solve_columns(sub, V, trans=True)  # M^{-T} V, (B, N, k)
    Ct = jnp.eye(p + q, dtype=dtype)[None] + jnp.einsum("bnk,bnl->bkl", Us, Z_Vt)
    Yt = jnp.linalg.inv(Ct)  # (B, k, k)

    return sub, Z_U, Y, Z_Vt, Yt, s, wrap_cols, wrap_rows


@functools.partial(jax.jit, static_argnames=("trans",))
def cr_banded_periodic_solve(factors, b, *, trans=False):
    """Solve a periodic banded system from cr_banded_periodic_factor factors.

    Parameters
    ----------
    factors : tuple
        Output of cr_banded_periodic_factor.
    b : jax.Array, shape (..., N)
    trans : bool
        Solve with A^T if True.

    Returns
    -------
    x : jax.Array, shape (..., N)
    """
    *lead, Nin = b.shape
    B = int(np.prod(lead)) if lead else 1
    bflat = b.reshape(B, Nin)

    if len(factors) == 2:
        # dense fallback from cr_banded_periodic_factor (N <= 2*bw); len is static
        lu, piv = factors
        tr = 1 if trans else 0
        x = jax.vmap(lambda a, p_, r: jax.scipy.linalg.lu_solve((a, p_), r, trans=tr))(
            lu, piv, bflat
        )
        return x.reshape(*lead, Nin) if lead else x.reshape(Nin)

    sub, Z_U, Y, Z_Vt, Yt, s, wrap_cols, wrap_rows = factors

    if trans:
        # A^T x = b  =>  x = s * (S^-T b), S = diag(s) A_periodic = M + Us VT
        Mtb = cr_banded_solve(sub, bflat, trans=True).reshape(B, Nin)  # M^-T b
        sel = (s * Mtb)[:, wrap_rows]  # Us^T (M^-T b)
        corr = jnp.einsum("bnk,bk->bn", Z_Vt, jnp.einsum("bkl,bl->bk", Yt, sel))
        x = s * (Mtb - corr)
    else:
        # A x = b  =>  S x = diag(s) b,  x = M^-1(sb) - Z_U (Y (M^-1(sb))[wrap])
        Minvsb = cr_banded_solve(sub, s * bflat).reshape(B, Nin)
        corr = jnp.einsum(
            "bnk,bk->bn", Z_U, jnp.einsum("bkl,bl->bk", Y, Minvsb[:, wrap_cols])
        )
        x = Minvsb - corr

    return x.reshape(*lead, Nin) if lead else x.reshape(Nin)


@functools.partial(jax.jit, static_argnames=("p", "q"))
@functools.partial(jnp.vectorize, signature="(k,n),(n)->(n)", excluded=(0, 1))
def banded_mv(p, q, A, x):
    """Matrix vector product w/ banded matrix.

    Parameters
    ----------
    p, q : int
        Lower and upper bandwidth of A
    A : jax.Array, shape(..., p+q+1, n)
        System matrix in banded storage format.
    x : jax.Array, shape(..., n)
        Vector to multiply

    Returns
    -------
    b : jax.Array, shape(..., n)
        b = A@x
    """
    if A.shape[-1] < (p + q + 1):
        # below isn't correct, just use dense matvec
        return banded_to_dense(p, q, A) @ x
    # Offsets from top row to bottom row: [p, p-1, ..., 0, ..., -q]
    # If p and q were swapped in your specific storage,
    # ensuring p is 'upper' and q is 'lower' here:
    offsets = jnp.arange(q, -p - 1, -1)

    def multiply_row(row_data, offset):
        # 1. Multiply the diagonal row by the vector element-wise
        # 2. Roll the result to align it with the output vector 'b'
        return jnp.roll(row_data * x, -offset)

    # Vectorized across all diagonals
    all_contributions = jax.vmap(multiply_row)(A, offsets)

    return jnp.sum(all_contributions, axis=0)


@functools.partial(jax.jit, static_argnames=["p1", "q1", "p2", "q2"])
@functools.partial(
    jnp.vectorize, signature="(k,n),(l,n)->(m,n),(),()", excluded=(0, 1, 2, 3)
)
def banded_mm(p1, q1, p2, q2, A, B):
    """Matrix-matrix product w/ banded matrices.

    Parameters
    ----------
    p1, q1 : int
        Lower and upper bandwidth of A
    p2, q2 : int
        Lower and upper bandwidth of B
    A : jax.Array, shape(..., p1+q1+1, n)
        Matrix 1 in banded storage format.
    B : jax.Array, shape(..., p2+q2+1, n)
        Matrix 2 in banded storage format.

    Returns
    -------
    C : jax.Array, shape(..., p1+p2+q1+q2+1, n)
        C = A@B
    p, q : int
        Lower and upper bandwidth of C
    """
    H_A, n = A.shape
    H_B, _ = B.shape
    if n < p1 + q1 + p2 + q2 + 1:
        # below is incorrect, just use dense matmul
        A = banded_to_dense(p1, q1, A)
        B = banded_to_dense(p2, q2, B)
        pc = min(n, p1 + p2)
        qc = min(n, q1 + q2)
        return dense_to_banded(pc, qc, A @ B), pc, qc

    # The output matrix inherently combines the total number of bands
    H_C = H_A + H_B - 1

    # Precompute column indices for the periodic wrap-around.
    # r2_idx represents the row indices of B, j_idx represents the columns.
    r2_idx = jnp.arange(H_B)[:, None]
    j_idx = jnp.arange(n)[None, :]

    # The shift applied to A's columns depends on B's upper bandwidth (q2)
    # and the specific row of B being evaluated (r2).
    col_idx = (j_idx - q2 + r2_idx) % n

    def compute_row(r):
        """Computes the r-th row (diagonal) of the output banded matrix."""
        r2 = jnp.arange(H_B)
        r1 = r - r2

        # Mask out indices where r1 falls outside the valid rows of A
        valid = (r1 >= 0) & (r1 < H_A)
        r1_safe = _where(valid, r1, jnp.array(0))

        # Gather elements from A. col_idx has shape (H_B, n)
        A_vals = A[r1_safe[:, None], col_idx]

        # Element-wise multiply by B and zero out invalid index contributions
        product = _where(valid[:, None], A_vals * B, jnp.array(0.0))

        # Summing over the intermediate dimension (r2) gives the dot product
        return jnp.sum(product, axis=0)

    # Vectorize the computation over every row of the output matrix C
    return jax.vmap(compute_row)(jnp.arange(H_C)), (p1 + p2), (q1 + q2)


@jax.jit
def banded_transpose(p, q, A):
    """Transposes a (periodic) banded matrix in compact format.

    Parameters
    ----------
    p, q : int
        Lower and upper bandwidth of A
    A : jax.Array, shape(..., p+q+1, n)
        Matrix in banded storage format.

    Returns
    -------
    A_T: jax.Array, shape (p + q + 1, n)
        Transpose of A
    p, q : int
        Lower and upper bandwidth of A_T
    """
    H, n = A.shape

    # Set up the grid of output coordinates
    r = jnp.arange(H)[:, None]
    c = jnp.arange(n)[None, :]

    # 1. Flip the rows vertically (H - 1 - r)
    r_A = H - 1 - r

    # 2. Shift the columns horizontally, wrapping around natively with % n
    c_A = (c + r - p) % n

    # Gather elements using advanced indexing
    return A[r_A, c_A], q, p


@jax.jit
def matrix_1norm(A):
    """1-norm ``||A||_1`` (max absolute column sum), per block.

    Works for dense storage ``(..., n, n)`` and banded storage ``(..., p+q+1, n)``:
    in both, matrix column ``j`` is ``A[..., :, j]`` (for banded, the entries of column
    ``j`` are the stored diagonals at column ``j``), so summing ``|.|`` over axis ``-2``
    and maxing over axis ``-1`` gives the maximum absolute column sum.
    """
    return jnp.max(jnp.sum(jnp.abs(A), axis=-2), axis=-1)


def _hager_1norm_est(apply_B, apply_BT, n, lead, dtype, iters):
    """Hager/Higham lower-bound estimate of ``||B||_1`` in a fixed number of iterations.

    Parameters
    ----------
    apply_B, apply_BT : callable
        Map ``(lead..., n) -> (lead..., n)``, applying ``B`` and ``B^T`` per block.
    n : int
        Block dimension.
    lead : tuple
        Batch (block) shape.
    dtype : jnp.dtype
    iters : int
        Number of iterations (LAPACK uses <= 5).
    """
    x = jnp.full(lead + (n,), 1.0 / n, dtype=dtype)

    def body(carry, _):
        x, est = carry
        y = apply_B(x)
        est = jnp.maximum(est, jnp.sum(jnp.abs(y), axis=-1))
        xi = jnp.where(y >= 0, 1.0, -1.0).astype(dtype)
        z = apply_BT(xi)
        j = jnp.argmax(jnp.abs(z), axis=-1)
        x = jax.nn.one_hot(j, n, dtype=dtype)
        return (x, est), None

    (x, est), _ = jax.lax.scan(body, (x, jnp.zeros(lead, dtype)), None, length=iters)

    # b_i = (-1)^i (1 + i/(n-1)) probes cancellation the unit vectors miss
    i = jnp.arange(n, dtype=dtype)
    b = (1.0 + i / max(n - 1, 1)) * (1.0 - 2.0 * (i % 2))
    y = apply_B(jnp.broadcast_to(b, lead + (n,)))
    alt = 2.0 * jnp.sum(jnp.abs(y), axis=-1) / (3.0 * n)
    return jnp.maximum(est, alt)


@functools.partial(jax.jit, static_argnames=["p", "q", "unroll"])
def cond_1norm_banded(p, q, A, lu_factors, *, iters=5, unroll=None):
    """Per-block 1-norm condition estimate for a (non-periodic) banded operator.

    ``||A||_1`` is exact from the banded storage; ``||A^{-1}||_1`` is estimated with a
    fixed-iteration Hager/Higham scheme.

    Parameters
    ----------
    p, q : int
        Lower/upper bandwidth of ``A``.
    A : jax.Array, shape ``(..., p+q+1, n)``
        Original banded-storage matrix (pre-factorization), batched over leading axes.
    lu_factors : tuple
        Factors ``(lu, s)`` from ``lu_factor_banded(p, q, A, ...)``.
    iters : int
        Fixed number of Hager iterations.
    unroll : int or None
        Loop unroll passed to the banded solves.

    Returns
    -------
    cond : jax.Array, shape ``(...)``
        Estimated ``kappa_1`` of each block.
    """
    lead = A.shape[:-2]
    n = A.shape[-1]
    a1 = matrix_1norm(A)

    def apply_B(V):
        return lu_solve_banded(p, q, lu_factors, V, unroll=unroll)

    def apply_BT(V):
        return lu_solve_banded(p, q, lu_factors, V, unroll=unroll, trans=True)

    ainv1 = _hager_1norm_est(apply_B, apply_BT, n, lead, A.dtype, iters)
    return a1 * ainv1


@jax.jit
def cond_1norm_cr(A, cr_factors, *, iters=5):
    """Per-block 1-norm condition estimate for a (non-periodic) banded operator.

    Identical to cond_1norm_banded but drives the Hager/Higham ||A^-1||_1 estimate with
    the cyclic-reduction solves cr_banded_solve instead of the banded LU. ||A||_1
    is exact from the banded storage.

    Parameters
    ----------
    A : jax.Array, shape (..., 2*bw+1, n)
        Original banded-storage matrix (pre-factorization), batched over leading axes.
    cr_factors : tuple
        Factors from cr_banded_factor(A, ...).
    iters : int
        Fixed number of Hager iterations.

    Returns
    -------
    cond : jax.Array, shape (...)
        Estimated ``kappa_1`` of each block.
    """
    lead = A.shape[:-2]
    n = A.shape[-1]
    a1 = matrix_1norm(A)

    def apply_B(V):
        return cr_banded_solve(cr_factors, V)

    def apply_BT(V):
        return cr_banded_solve(cr_factors, V, trans=True)

    ainv1 = _hager_1norm_est(apply_B, apply_BT, n, lead, A.dtype, iters)
    return a1 * ainv1
