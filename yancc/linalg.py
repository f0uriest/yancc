"""Linear algebra helpers."""

import functools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


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


@functools.partial(jax.jit, static_argnames=("p", "q", "unroll"))
@functools.partial(jnp.vectorize, signature="(m,n),(n)->(n)", excluded=(0, 1, "unroll"))
def _lu_solve_banded(p, q, lu, b, *, unroll=None):
    n = lu.shape[1]
    assert p <= n
    assert q <= n
    assert lu.shape[0] == (p + q + 1)

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


def lu_solve_banded(p, q, lu_factors, b, *, unroll=None):
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

    Returns
    -------
    x : jax.Array, shape(...,N)
        Solution to linear system.
    """
    lu, s = lu_factors
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
