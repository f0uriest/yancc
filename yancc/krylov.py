"""GCROTMK krylov solver."""

import operator
from functools import partial

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import lineax as lx
from jax import lax
from jax import scipy as jsp
from jax.tree_util import tree_leaves, tree_map, tree_reduce

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)

########
# this stuff is from jax,
# https://github.com/jax-ml/jax/blob/5aa339561871ccf037f1216237cf4e5db937376c/jax/_src/scipy/sparse/linalg.py#L594-L705
#
#
# used under the apache license:
#     https://www.apache.org/licenses/LICENSE-2.0
########


def _vdot_real_part(x, y):
    """Vector dot-product guaranteed to have a real valued result despite
    possibly complex input. Thus neglects the real-imaginary cross-terms.
    The result is a real float.
    """
    result = _vdot(x.real, y.real)
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
        result += _vdot(x.imag, y.imag)
    return result


def _norm(x):
    xs = tree_leaves(x)
    return jnp.sqrt(sum(map(_vdot_real_part, xs, xs)))


def _tree_vdot(x, y):
    xs = tree_leaves(x)
    ys = tree_leaves(y)
    return sum(map(_vdot, xs, ys))


_add = partial(tree_map, operator.add)
_sub = partial(tree_map, operator.sub)
_mul = partial(tree_map, operator.mul)


def _safe_normalize(x, thresh=None):
    """
    Returns the L2-normalized vector (which can be a pytree) x, and optionally
    the computed norm. If the computed norm is less than the threshold `thresh`,
    which by default is the machine precision of x's dtype, it will be
    taken to be 0, and the normalized x to be the zero vector.
    """
    norm = _norm(x)
    dtype = jnp.result_type(*tree_leaves(x))
    if thresh is None:
        thresh = jnp.finfo(norm.dtype).eps
    thresh = thresh.astype(dtype).real

    use_norm = norm > thresh
    normalized_x = tree_map(lambda y: jnp.where(use_norm, y / norm, 0.0), x)
    norm = jnp.where(use_norm, norm, 0.0)
    return normalized_x, norm


def _project_on_columns(A, v):
    """Returns A.T.conj() @ v."""
    v_proj = tree_map(
        lambda X, y: _einsum("...n,...->n", X.conj(), y),
        A,
        v,
    )
    return tree_reduce(operator.add, v_proj)


def _iterative_classical_gram_schmidt(Q, x, xnorm, max_iterations=2):
    """
    Orthogonalize x against the columns of Q. The process is repeated
    up to `max_iterations` times, or fewer if the condition
    ||r|| < (1/sqrt(2)) ||x|| is met earlier (see below for the meaning
    of r and x).

    Parameters
    ----------
    Q : array or tree of arrays
        A matrix of orthonormal columns.
    x : array or tree of arrays
        A vector. It will be replaced with a new vector q which is orthonormal
        to the columns of Q, such that x in span(col(Q), q).
    xnorm : float
        Norm of x.

    Returns
    -------
    q : array or tree of arrays
        A unit vector, orthonormal to each column of Q, such that
        x in span(col(Q), q).
    r : array
        Stores the overlaps of x with each vector in Q.
    """
    # "twice is enough"
    # http://slepc.upv.es/documentation/reports/str1.pdf

    # TODO(shoyer): consider switching to only one iteration, like SciPy?

    # This assumes that Q's leaves all have the same dimension in the last
    # axis.
    Q0 = tree_leaves(Q)[0]
    r = jnp.zeros(Q0.shape[-1], dtype=Q0.dtype)
    q = x
    xnorm_scaled = xnorm / jnp.sqrt(2.0)

    def body_function(carry):
        k, q, r, qnorm_scaled = carry
        h = _project_on_columns(Q, q)
        Qh = tree_map(lambda X: _dot(X, h), Q)
        q = _sub(q, Qh)
        r = _add(r, h)

        def qnorm_cond(carry):
            k, not_done, _, _ = carry
            return jnp.logical_and(not_done, k < (max_iterations - 1))

        def qnorm(carry):
            k, _, q, qnorm_scaled = carry
            _, qnorm = _safe_normalize(q)
            qnorm_scaled = qnorm / jnp.sqrt(2.0)
            return (k, False, q, qnorm_scaled)

        init = (k, True, q, qnorm_scaled)
        _, _, q, qnorm_scaled = lax.while_loop(qnorm_cond, qnorm, init)
        return (k + 1, q, r, qnorm_scaled)

    def cond_function(carry):
        k, _, r, qnorm_scaled = carry
        _, rnorm = _safe_normalize(r)
        return jnp.logical_and(k < (max_iterations - 1), rnorm < qnorm_scaled)

    k, q, r, qnorm_scaled = body_function((0, q, r, xnorm_scaled))
    k, q, r, _ = lax.while_loop(cond_function, body_function, (k, q, r, qnorm_scaled))
    return q, r


def _rotate_vectors(H, i, cs, sn):
    x1 = H[i]
    y1 = H[i + 1]
    x2 = cs.conj() * x1 - sn.conj() * y1
    y2 = sn * x1 + cs * y1
    H = H.at[i].set(x2)
    H = H.at[i + 1].set(y2)
    return H


def _givens_rotation(a, b):
    b_zero = abs(b) == 0
    a_lt_b = abs(a) < abs(b)
    t = -jnp.where(a_lt_b, a, b) / jnp.where(a_lt_b, b, a)
    r = lax.rsqrt(1 + abs(t) ** 2).astype(t.dtype)
    cs = jnp.where(b_zero, 1, jnp.where(a_lt_b, r * t, r))
    sn = jnp.where(b_zero, 0, jnp.where(a_lt_b, r, r * t))
    return cs, sn


def _apply_givens_rotations(H_row, givens, k):
    """
    Applies the Givens rotations stored in the vectors cs and sn to the vector
    H_row. Then constructs and applies a new Givens rotation that eliminates
    H_row's k'th element.
    """
    # This call successively applies each of the
    # Givens rotations stored in givens[:, :k] to H_col.

    def apply_ith_rotation(i, H_row):
        return _rotate_vectors(H_row, i, *givens[i, :])

    R_row = lax.fori_loop(0, k, apply_ith_rotation, H_row)

    givens_factors = _givens_rotation(R_row[k], R_row[k + 1])
    givens = givens.at[k, :].set(givens_factors)
    R_row = _rotate_vectors(R_row, k, *givens_factors)
    return R_row, givens


####
# begin non-jax stuff
####


@eqx.filter_jit
def _fgmres(
    matvec,
    v0,
    m,
    k,
    atol,
    C=None,
    lc=None,
    lpsolve=None,
    rpsolve=None,
):
    """FGMRES Arnoldi process, with optional projection or augmentation

    Parameters
    ----------
    matvec : callable
        Operation A*x
    v0 : pytree of jax.Array
        Initial residual.
    m : int
        Number of FGMRES rounds.
    k : int
        Number of vectors to carry between inner FGMRES iterations.
    atol : float
        Absolute tolerance for early exit
    C : pytree of jax.Array
        Matrix C in GCROTMK algorithm.
    lc : int
        Number of nonzero columns of C. Default C.shape[1]
    lpsolve : callable
        Left preconditioner L
    rpsolve : callable
        Right preconditioner R

    Returns
    -------
    H : ndarray
        Upper hessenberg matrix.
    B : ndarray
        Projections corresponding to matrix C
    V : pytree of jax.Array
        Columns of matrix V
    Z : pytree of jax.Array
        Columns of matrix Z
    y : ndarray
        Solution to ||H y - e_1||_2 = min!
    """
    if lpsolve is None:

        def lpsolve(x):
            return x

    if rpsolve is None:

        def rpsolve(x):
            return x

    # krylov space V = [V0, Av0, A^2v0, ...]
    V = tree_map(
        lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, m + k),)),
        v0,
    )
    # preconditioned krylov space [Mv0, M(AM)v0, M(AM)^2 v0, ...]
    Z = tree_map(lambda x: jnp.zeros_like(x[:, 1:]), V)

    if C is None:
        C = tree_map(lambda x: jnp.empty((*x.shape, 1)), v0)
        lc = 0
    else:
        C = tree_map(jnp.asarray, C)

    if lc is None:
        lc = tree_leaves(C)[0].shape[1]

    dtype = jnp.result_type(*tree_leaves(v0))
    eps = jnp.finfo(dtype).eps
    res = jnp.array(_norm(v0))

    # Orthogonal projection coefficients
    B = jnp.zeros((tree_leaves(C)[0].shape[1], m + k), dtype=v0.dtype)

    # H=QR. We only need R here but need H itself in outer loop, never need Q, we only
    # store Q*e1*beta = beta_vec
    R = jnp.eye(m + k, m + k + 1, dtype=v0.dtype)
    H = jnp.zeros((m + k, m + k + 1), dtype=v0.dtype)
    beta_vec = jnp.zeros((m + k + 1), dtype=dtype).at[0].set(res.astype(dtype))
    givens = jnp.zeros((m + k, 2), dtype=dtype)

    breakdown = False
    maxiter = m + jnp.maximum(k - lc, 0)
    # FGMRES Arnoldi process

    def arnoldi_cond(carry):
        j, _, _, _, _, _, _, _, res, breakdown = carry
        return jnp.logical_and(jnp.logical_and(j < maxiter, res > atol), ~breakdown)

    def arnoldi_loop(carry):
        # L A Z = C B + V H
        j, V, Z, B, R, H, givens, beta_vec, res, breakdown = carry
        v = tree_map(lambda x: x[..., j], V)  # Gets V[:, k]
        z = rpsolve(v)
        w = lpsolve(matvec(z))

        _, w_norm = _safe_normalize(w)

        # GCROT projection: L A -> (1 - C C^H) L A
        # i.e. orthogonalize against C
        def _C_loop(i, carry):
            B, w = carry
            c = tree_map(lambda x: jnp.asarray(x)[..., i], C)
            alpha = _tree_vdot(c, w)
            B = B.at[i, j].set(alpha)
            w = _sub(w, _mul(alpha, c))
            return B, w

        B, w = lax.fori_loop(0, lc, _C_loop, (B, w))

        w, h = _iterative_classical_gram_schmidt(V, w, w_norm, max_iterations=2)
        unit_w, w_norm_1 = _safe_normalize(w, thresh=eps * w_norm)
        V = tree_map(lambda X, y: X.at[..., j + 1].set(y), V, unit_w)
        Z = tree_map(lambda X, y: X.at[..., j].set(y), Z, z)
        h = h.at[j + 1].set(w_norm_1.astype(dtype))
        R = R.at[j, :].set(h)
        H = H.at[j, :].set(h)

        R_row, givens = _apply_givens_rotations(R[j, :], givens, j)
        R = R.at[j, :].set(R_row)
        beta_vec = _rotate_vectors(beta_vec, j, *givens[j, :])
        res = abs(beta_vec[j + 1])
        breakdown = H[j, j + 1] < eps * w_norm

        return j + 1, V, Z, B, R, H, givens, beta_vec, res, breakdown

    carry = (0, V, Z, B, R, H, givens, beta_vec, res, breakdown)
    _, V, Z, B, R, H, _, beta_vec, _, _ = lax.while_loop(
        arnoldi_cond, arnoldi_loop, carry
    )
    y = jsp.linalg.solve_triangular(R[:, :-1].T, beta_vec[:-1])

    return H, B, V, Z, y


@eqx.filter_jit
def gcrotmk(
    A,
    b,
    x0=None,
    *,
    rtol=jnp.array(1e-5),
    atol=jnp.array(0.0),
    maxiter=jnp.array(1000),
    M=None,
    m=20,
    k=None,
    C=None,
    U=None
):
    """
    Solve a matrix equation using flexible GCROT(m,k) algorithm.

    Parameters
    ----------
    A : lineax.LinearOperator
        System matrix as lineax.LinearOperator.
    b : jax.Array
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : jax.Array
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5`` and ``atol=0.0``.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved. The
        default is ``1000``.
    M : lineax.LinearOperator, optional
        Preconditioner for `A`.  The preconditioner should approximate the
        inverse of `A`. gcrotmk is a 'flexible' algorithm and the preconditioner
        can vary from iteration to iteration. Effective preconditioning
        dramatically improves the rate of convergence, which implies that
        fewer iterations are needed to reach a given error tolerance.
    m : int, optional
        Number of inner FGMRES iterations per each outer iteration.
        Default: 20
    k : int, optional
        Number of vectors to carry between inner FGMRES iterations.
        According to [2]_, good values are around `m`.
        Default: `m`
    C, U : pytree of jax.Array, optional
        Matrices C and U in the GCROT(m,k) algorithm. For details, see [2]_.
        If not given, start from empty matrices. If ``U`` is given but ``C`` is
        ``None`` then ``C`` is recomputed via ``C = A U`` on start and
        orthogonalized as described in [3]_. ``U`` should have the same tree structure
        as ``x`` but with a trailing dimension, and ``C`` should have the same
        structure as ``b`` but with a trailing dimension.

    Returns
    -------
    x : ndarray
        The solution found.
    info : int
        Provides convergence information:

        * 0  : successful exit
        * >0 : convergence to tolerance not achieved, number of iterations
    C, U : pytree of jax.Array, optional
        Matrices C and U in the GCROT(m,k) algorithm. For details, see [2]_.

    References
    ----------
    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
           methods'', SIAM J. Numer. Anal. 36, 864 (1999).
    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
           of GCROT for solving nonsymmetric linear systems'',
           SIAM J. Sci. Comput. 32, 172 (2010).
    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
           ''Recycling Krylov subspaces for sequences of linear systems'',
           SIAM J. Sci. Comput. 28, 1651 (2006).

    """
    structure = A.in_structure()
    if M is None:
        M = lx.IdentityLinearOperator(structure)
    matvec = A.mv
    psolve = M.mv

    if x0 is None:
        x = tree_map(jnp.zeros_like, b)
    else:
        x = x0

    r = _sub(b, matvec(x))
    if k is None:
        k = m

    b_norm = _norm(b)
    tol = jnp.maximum(atol, rtol * b_norm)

    if U is None:
        assert C is None
        lc = 0
        U = tree_map(lambda x: jnp.zeros((x.size, k)), x)
        C = tree_map(lambda x: jnp.zeros((x.size, k)), x)
    else:  # U provided
        U = tree_map(lambda x: jnp.atleast_2d(x.T).T, U)
        lc = tree_leaves(U)[0].shape[-1]  # number of supplied Us
        if C is None:
            C = jax.vmap(matvec, in_axes=1, out_axes=1)(U)
        C = tree_map(lambda x: jnp.atleast_2d(x.T).T, C)
        # Reorthogonalize old vectors
        c = tree_map(lambda x: x[..., 0], C)
        unflatten = jax.flatten_util.ravel_pytree(c)[1]
        C = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0], in_axes=1, out_axes=1
        )(C)

        Q, R, P = jsp.linalg.qr(C, mode="economic", pivoting=True)
        C = jax.vmap(unflatten, in_axes=1, out_axes=1)(Q)
        #   AUP = CP = Q R
        #   U' = U P R^-1
        tol = jnp.finfo(R.dtype).eps * jnp.abs(R[0, 0]) * max(Q.shape)
        mask = jnp.abs(jnp.diag(R)) > tol
        U = tree_map(
            lambda x: jsp.linalg.solve_triangular(R.T, x[:, P].T, lower=True).T, U
        )
        U = tree_map(lambda x: jnp.where(mask, x, 0), U)
        # pad to full size
        U = tree_map(lambda x: jnp.pad(x, ((0, 0), (0, k - lc))), U)
        C = tree_map(lambda x: jnp.pad(x, ((0, 0), (0, k - lc))), C)
        # if initial data wasn't full rank, only some are valid
        lc = jnp.sum(mask)

    def initial_projection(x, r):
        # Solve first the projection operation with respect to the C, U matrices
        #   y = argmin_y || b - A (x + U y) ||^2 = C^H (b - A x)
        #   x' = x + U y
        y = jax.vmap(lambda x: _tree_vdot(x, r), in_axes=1)(C)
        x = _add(x, tree_map(lambda x: _dot(x, y), U))
        r = _sub(r, tree_map(lambda x: _dot(x, y), C))
        return x, r

    x, r = lax.cond(lc, initial_projection, lambda *args: args, x, r)

    def gcmotmk_loop(carry):
        j_outer, x, r, beta, C, U, lc = carry

        H, B, V, Z, y = _fgmres(
            matvec,
            v0=_mul(1 / beta, r),
            m=m,
            k=k,
            rpsolve=psolve,
            atol=tol / beta,
            C=C,
            lc=lc,
        )
        y *= beta

        # ux := (Z - U B) y
        Zy = tree_map(lambda x: _dot(x, y), Z)
        By = B @ y
        UBy = tree_map(lambda x: _dot(x, By), U)
        ux = _sub(Zy, UBy)

        Hy = H.T @ y
        cx = tree_map(lambda x: _dot(x, Hy), V)

        # Normalize cx, maintaining cx = A ux
        # This new cx is orthogonal to the previous C, by construction
        alpha = 1 / _norm(cx)
        cx = _mul(alpha, cx)
        ux = _mul(alpha, ux)

        # Update residual and solution
        gamma = _dot(cx, r)
        r = _sub(r, _mul(gamma, cx))
        x = _add(x, _mul(gamma, ux))
        r = _sub(b, matvec(x))

        U = tree_map(_roll_prepend, U, ux)
        C = tree_map(_roll_prepend, C, cx)
        lc = jnp.minimum(lc + 1, k)
        beta = _norm(r)

        return j_outer + 1, x, r, beta, C, U, lc

    def gcrotmk_cond(carry):
        j_outer, _, _, beta, _, _, _ = carry
        return jnp.logical_and(j_outer < maxiter, beta > tol)

    beta = _norm(r)
    carry = (0, x, r, beta, C, U, lc)
    carry = lax.while_loop(gcrotmk_cond, gcmotmk_loop, carry)
    j_outer, x, r, beta, C, U, lc = carry
    # Include the solution vector to the span
    U = tree_map(_roll_prepend, U, x)
    C = tree_map(_roll_prepend, C, _sub(b, r))

    return x, j_outer, C, U


def _roll_prepend(X, y):
    return jnp.roll(X, shift=1, axis=1).at[:, 0].set(y)
