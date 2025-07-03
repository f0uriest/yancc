"""GCROT(m,k) and LGMRES krylov solvers."""

import operator
from collections.abc import Callable
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import lineax as lx
from jax import lax
from jax import scipy as jsp
from jax.tree_util import tree_leaves, tree_map, tree_reduce
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

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


def _givens_rotation(a: jax.Array, b: jax.Array):
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


def _roll_prepend(X: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.roll(X, shift=1, axis=1).at[:, 0].set(y)


def _identity(x: PyTree[ArrayLike]) -> PyTree[ArrayLike]:
    return x


def _maybe_print(flag, j, res, pre=""):

    def truefun():
        jax.debug.print(pre + "iter={j:3d}   res={res:.3e}", j=j, res=res)

    def falsefun():
        pass

    jax.lax.cond(flag, truefun, falsefun)


@eqx.filter_jit
def _fgmres(
    matvec: Callable[[PyTree[ArrayLike]], PyTree[ArrayLike]],
    v0: PyTree[ArrayLike],
    m: int,
    k: int,
    atol: Float[ArrayLike, ""],
    lpsolve: Optional[Callable[[PyTree[ArrayLike]], PyTree[ArrayLike]]] = None,
    rpsolve: Optional[Callable[[PyTree[ArrayLike]], PyTree[ArrayLike]]] = None,
    C: Optional[PyTree[ArrayLike]] = None,
    lc: Optional[int] = None,
    outer_v: Optional[PyTree[ArrayLike]] = None,
    outer_Av: Optional[PyTree[ArrayLike]] = None,
    lv: Optional[int] = None,
    print_every: ArrayLike = jnp.inf,
) -> tuple[Array, Array, PyTree[Array], PyTree[Array], Array, int, int, Array]:
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
    lpsolve : callable
        Left preconditioner L
    rpsolve : callable
        Right preconditioner R
    C : pytree of jax.Array
        Matrix C in GCROT(m,k) algorithm.
    lc : int
        Number of nonzero columns of C. Default C.shape[1]
    outer_v : pytree of jax.Array
        Augmentation vectors in LGMRES.
    outer_Av : pytree of jax.Array
        Augmentation vectors in LGMRES.
    lv : int
        Number of nonzero columns of outer_v. Default outer_v.shape[1]

    Returns
    -------
    H : ndarray
        Upper Hessenberg matrix.
    B : ndarray
        Projections corresponding to matrix C
    V : pytree of jax.Array
        Columns of matrix V
    Z : pytree of jax.Array
        Columns of matrix Z
    y : ndarray
        Solution to ||H y - e_1||_2 = min!
    j : int
        Number of iterations.
    nmv : int
        Number of matrix vector products
    res : float
        Final residual.
    """
    nmv = 0
    atol = jnp.asarray(atol)

    if lpsolve is None:
        lpsolve = _identity

    if rpsolve is None:
        rpsolve = _identity

    if outer_v is None:
        assert lv is None, "if outer_v is None, lv must also be None"
        assert outer_Av is None, "if outer_v is None, outer_Av must also be None"
        outer_v = tree_map(lambda x: jnp.empty((*x.shape, 1)), v0)
        outer_Av = tree_map(lambda x: jnp.empty((*x.shape, 1)), v0)
        lv = 0

    if lv is None:
        lv = tree_leaves(outer_v)[0].shape[1]

    assert lv is not None

    if outer_Av is None:
        outer_Av = jax.vmap(matvec, in_axes=1, out_axes=1)(outer_v)
        nmv += lv

    if C is None:
        assert lc is None, "if C is None, lc must also be None"
        C = tree_map(lambda x: jnp.empty((*x.shape, 1)), v0)
        lc = 0
    else:
        C = tree_map(jnp.asarray, C)

    if lc is None:
        lc = tree_leaves(C)[0].shape[1]

    assert lc is not None
    maxiter = m + jnp.maximum(k - lc, 0) + lv
    size = m + k + tree_leaves(outer_v)[0].shape[1]

    # krylov space V = [V0, Av0, A^2v0, ...]
    V = tree_map(
        lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, size),)),
        v0,
    )
    # preconditioned krylov space [Mv0, M(AM)v0, M(AM)^2 v0, ...]
    Z = tree_map(lambda x: jnp.zeros_like(x[:, 1:]), V)

    dtype = jnp.result_type(*tree_leaves(v0))
    eps = jnp.finfo(dtype).eps
    res = jnp.array(_norm(v0))

    # Orthogonal projection coefficients
    B = jnp.zeros((tree_leaves(C)[0].shape[1], size), dtype=v0.dtype)

    # H=QR. We only need R here but need H itself in outer loop, never need Q, we only
    # store Q*e1*beta = beta_vec
    R = jnp.eye(size, size + 1, dtype=v0.dtype)
    H = jnp.zeros((size, size + 1), dtype=v0.dtype)
    beta_vec = jnp.zeros((size + 1), dtype=dtype).at[0].set(res.astype(dtype))
    givens = jnp.zeros((size, 2), dtype=dtype)

    breakdown = jnp.array(False)

    # FGMRES Arnoldi process

    def arnoldi_cond(carry):
        j, _, _, _, _, _, _, _, _, res, breakdown = carry
        return jnp.logical_and(jnp.logical_and(j < maxiter, res > atol), ~breakdown)

    def arnoldi_loop(carry):
        # L A Z = C B + V H
        j, nmv, V, Z, B, R, H, givens, beta_vec, res, breakdown = carry

        def outer_v_iteration(j, nmv, V):
            z = lax.cond(
                j < lv,
                lambda: tree_map(lambda x: x[..., j], outer_v),
                lambda: rpsolve(v0),
            )
            w = lax.cond(
                j < lv,
                lambda: tree_map(lambda x: x[..., j], outer_Av),
                lambda: lpsolve(matvec(z)),
            )
            nmv = jnp.where(j < lv, nmv, nmv + 1)
            return z, w, nmv

        def regular_iteration(j, nmv, V):
            v = tree_map(lambda x: x[..., j], V)  # Gets V[:, k]
            z = rpsolve(v)
            w = lpsolve(matvec(z))
            nmv += 1
            return z, w, nmv

        z, w, nmv = lax.cond(j <= lv, outer_v_iteration, regular_iteration, j, nmv, V)
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
        _maybe_print(
            jnp.logical_and(print_every < jnp.inf, jnp.mod(j, print_every) == 0),
            j,
            res,
            pre="    FGMRES  ",
        )
        breakdown = H[j, j + 1] < eps * w_norm

        return j + 1, nmv, V, Z, B, R, H, givens, beta_vec, res, breakdown

    carry = (0, nmv, V, Z, B, R, H, givens, beta_vec, res, breakdown)
    j, nmv, V, Z, B, R, H, _, beta_vec, res, _ = lax.while_loop(
        arnoldi_cond, arnoldi_loop, carry
    )
    y = jsp.linalg.solve_triangular(R[:, :-1].T, beta_vec[:-1])

    return H, B, V, Z, y, j, nmv, res


@eqx.filter_jit
def gcrotmk(
    A: lx.AbstractLinearOperator,
    b: PyTree[ArrayLike],
    x0: Optional[PyTree[ArrayLike]] = None,
    *,
    rtol: Float[ArrayLike, ""] = jnp.array(1e-5),
    atol: Float[ArrayLike, ""] = jnp.array(0.0),
    maxiter: Int[ArrayLike, ""] = jnp.array(1000),
    ML: Optional[lx.AbstractLinearOperator] = None,
    MR: Optional[lx.AbstractLinearOperator] = None,
    m: int = 20,
    k: Optional[int] = None,
    C: Optional[PyTree[ArrayLike]] = None,
    U: Optional[PyTree[ArrayLike]] = None,
    print_every: ArrayLike = False,
) -> tuple[PyTree[Array], int, int, Array, PyTree[Array], PyTree[Array]]:
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
    ML, MR : lineax.LinearOperator, optional
        Left and/or right Preconditioner for `A`.  The preconditioners should be such
        that `ML @ A @ MR` is better conditioned than `A` itself. gcrotmk is a
        'flexible' algorithm and the right preconditioner `MR` can vary from iteration
        to iteration. Effective preconditioning dramatically improves the rate of
        convergence, which implies that fewer iterations are needed to reach a given
        error tolerance.
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
    x : pytree of ndarray
        The solution found.
    iters : int
        Number of iterations.
    nmv : int
        Number of matrix vector products.
    residual : float
        Residual of the linear system.
    C, U : pytree of jax.Array
        Matrices C and U in the GCROT(m,k) algorithm. For details, see [2]_.

    References
    ----------
    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
           methods'', SIAM J. Numerical Analysis 36, 864 (1999).
    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
           of GCROT for solving nonsymmetric linear systems'',
           SIAM Journal of  Scientific Computing 32, 172 (2010).
    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
           ''Recycling Krylov subspaces for sequences of linear systems'',
           SIAM Journal of  Scientific Computing 28, 1651 (2006).

    """
    if ML is None:
        ML = lx.IdentityLinearOperator(A.out_structure())
    if MR is None:
        MR = lx.IdentityLinearOperator(A.in_structure())

    matvec = A.mv
    lpsolve = ML.mv
    rpsolve = MR.mv
    print_every = jnp.asarray(print_every)
    print_every = jnp.where(print_every == 0, jnp.inf, print_every)

    if x0 is None:
        x = tree_map(jnp.zeros_like, A.in_structure())
    else:
        x = x0

    if k is None:
        k = m

    b_norm = _norm(b)
    tol = jnp.maximum(atol, rtol * b_norm)
    ptol_max_factor = 1.0
    r = _sub(b, matvec(x))
    dtype = jnp.result_type(*tree_leaves(r))
    eps = jnp.finfo(dtype).eps
    nmv = 1
    beta = _norm(r)
    _maybe_print(print_every < jnp.inf, 0, beta / b_norm, pre="GCROT  ")

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
            nmv += lc
        C = tree_map(lambda x: jnp.atleast_2d(x.T).T, C)
        # re-orthogonalize old vectors
        c = tree_map(lambda x: x[..., 0], C)
        unflatten = jax.flatten_util.ravel_pytree(c)[1]
        Carr: jax.Array = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0], in_axes=1, out_axes=1
        )(C)

        Q, R, P = jsp.linalg.qr(Carr, mode="economic", pivoting=True)
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

    def initial_projection(x, r, beta):
        # Solve first the projection operation with respect to the C, U matrices
        #   y = argmin_y || b - A (x + U y) ||^2 = C^H (b - A x)
        #   x' = x + U y
        y = jax.vmap(lambda x: _tree_vdot(x, r), in_axes=1)(C)
        x = _add(x, tree_map(lambda x: _dot(x, y), U))
        r = _sub(r, tree_map(lambda x: _dot(x, y), C))
        beta = _norm(r)
        return x, r, beta

    x, r, beta = lax.cond(lc, initial_projection, lambda *args: args, x, r, beta)

    def gcrotmk_cond(carry):
        j_outer, _, _, _, beta, _, _, _, _ = carry
        return jnp.logical_and(j_outer < maxiter, beta > tol)

    def gcmotmk_loop(carry):
        j_outer, nmv, x, r, beta, C, U, lc, ptol_max_factor = carry

        v0 = lpsolve(r)
        inner_res_0 = _norm(v0)

        v0 = _mul(1.0 / inner_res_0, v0)
        ptol = jnp.minimum(ptol_max_factor, tol / beta)

        H, B, V, Z, y, _, nmv_inner, pres = _fgmres(
            matvec,
            v0=v0,
            m=m,
            k=k,
            rpsolve=rpsolve,
            lpsolve=lpsolve,
            atol=ptol,
            C=C,
            lc=lc,
            print_every=print_every,
        )
        y *= inner_res_0
        nmv += nmv_inner

        # Inner loop tolerance control
        ptol_max_factor = jnp.where(
            pres > ptol,
            jnp.minimum(1.0, 1.5 * ptol_max_factor),
            jnp.maximum(eps, 0.25 * ptol_max_factor),
        )

        # u := (Z - U B) y
        Zy = tree_map(lambda x: _dot(x, y), Z)
        By = B @ y
        UBy = tree_map(lambda x: _dot(x, By), U)
        u = _sub(Zy, UBy)

        # c := V H y
        Hy = H.T @ y
        c = tree_map(lambda x: _dot(x, Hy), V)

        x = _add(x, u)
        r = _sub(b, matvec(x))
        nmv += 1
        beta = _norm(r)

        # Normalize cx, maintaining cx = A ux
        # This new cx is orthogonal to the previous C, by construction
        alpha = 1 / _norm(c)
        U = tree_map(_roll_prepend, U, _mul(alpha, u))
        C = tree_map(_roll_prepend, C, _mul(alpha, c))
        lc = jnp.minimum(lc + 1, k)

        _maybe_print(print_every < jnp.inf, j_outer + 1, beta / b_norm, pre="GCROT  ")

        return j_outer + 1, nmv, x, r, beta, C, U, lc, ptol_max_factor

    carry = (0, nmv, x, r, beta, C, U, lc, ptol_max_factor)
    carry = lax.while_loop(gcrotmk_cond, gcmotmk_loop, carry)
    j_outer, nmv, x, r, beta, C, U, _, _ = carry
    # Include the solution vector to the span
    U = tree_map(_roll_prepend, U, x)
    C = tree_map(_roll_prepend, C, _sub(b, r))

    return x, j_outer, nmv, beta, C, U


@eqx.filter_jit
def lgmres(
    A: lx.AbstractLinearOperator,
    b: PyTree[ArrayLike],
    x0: Optional[PyTree[ArrayLike]] = None,
    *,
    rtol: Float[ArrayLike, ""] = jnp.array(1e-5),
    atol: Float[ArrayLike, ""] = jnp.array(0.0),
    maxiter: Int[ArrayLike, ""] = jnp.array(1000),
    ML: Optional[lx.AbstractLinearOperator] = None,
    MR: Optional[lx.AbstractLinearOperator] = None,
    m: int = 30,
    k: int = 3,
    outer_v: Optional[PyTree[ArrayLike]] = None,
    outer_Av: Optional[PyTree[ArrayLike]] = None,
    print_every: ArrayLike = False,
) -> tuple[PyTree[Array], int, int, Array, PyTree[Array], PyTree[Array]]:
    """
    Solve a matrix equation using the LGMRES algorithm.

    The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems
    in the convergence in restarted GMRES, and often converges in fewer
    iterations.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5``, the default for ``atol`` is ``0.0``.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    ML, MR : lineax.LinearOperator, optional
        Left and/or right Preconditioner for `A`.  The preconditioners should be such
        that `ML @ A @ MR` is better conditioned than `A` itself. gcrotmk is a
        'flexible' algorithm and the right preconditioner `MR` can vary from iteration
        to iteration. Effective preconditioning dramatically improves the rate of
        convergence, which implies that fewer iterations are needed to reach a given
        error tolerance.
    m : int, optional
        Number of inner GMRES iterations per each outer iteration.
    k : int, optional
        Number of vectors to carry between inner GMRES iterations.
        According to [1]_, good values are in the range of 1...3.
        However, note that if you want to use the additional vectors to
        accelerate solving multiple similar problems, larger values may
        be beneficial.
    outer_v, outer_Av : pytree of jax.Array, optional
        Vectors and corresponding matrix-vector products, used to augment the Krylov
        subspace, and carried between inner GMRES iterations. If ``outer_v`` is given
        but ``outer_Av`` is ``None`` then ``outer_Av`` is recomputed automatically.
         ``outer_v`` should have the same tree structure as ``x`` but with a trailing
        dimension, and ``outer_Av`` should have the same structure as ``b`` but with
        a trailing dimension.

    Returns
    -------
    x : pytree of ndarray
        The solution found.
    iters : int
        Number of iterations.
    nmv : int
        Number of matrix vector products.
    residual : float
        Residual of the linear system.
    outer_v, outer_Av : pytree of jax.Array
        Vectors and corresponding matrix-vector products, used to augment the Krylov
        subspace, and carried between inner GMRES iterations.

    Notes
    -----
    The LGMRES algorithm [1]_ [2]_ is designed to avoid the
    slowing of convergence in restarted GMRES, due to alternating
    residual vectors. Typically, it often outperforms GMRES(m) of
    comparable memory requirements by some measure, or at least is not
    much worse.

    Another advantage in this algorithm is that you can supply it with
    'guess' vectors in the `outer_v` argument that augment the Krylov
    subspace. If the solution lies close to the span of these vectors,
    the algorithm converges faster. This can be useful if several very
    similar matrices need to be inverted one after another, such as in
    Newton-Krylov iteration where the Jacobian matrix often changes
    little in the nonlinear steps.

    References
    ----------
    .. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, "A Technique for
             Accelerating the Convergence of Restarted GMRES", SIAM J. Matrix
             Anal. Appl. 26, 962 (2005).
    .. [2] A.H. Baker, "On Improving the Performance of the Linear Solver
             restarted GMRES", PhD thesis, University of Colorado (2003).

    """
    if ML is None:
        ML = lx.IdentityLinearOperator(A.out_structure())
    if MR is None:
        MR = lx.IdentityLinearOperator(A.in_structure())

    matvec = A.mv
    lpsolve = ML.mv
    rpsolve = MR.mv
    print_every = jnp.asarray(print_every)
    print_every = jnp.where(print_every == 0, jnp.inf, print_every)

    if x0 is None:
        x = tree_map(jnp.zeros_like, A.in_structure())
    else:
        x = x0

    b_norm = _norm(b)
    tol = jnp.maximum(atol, rtol * b_norm)
    ptol_max_factor = 1.0
    r = _sub(b, matvec(x))
    dtype = jnp.result_type(*tree_leaves(r))
    eps = jnp.finfo(dtype).eps
    nmv = 1
    beta = _norm(r)
    _maybe_print(print_every < jnp.inf, 0, beta / b_norm, pre="LGMRES  ")

    if outer_v is None:
        assert outer_Av is None
        lv = 0
        outer_v = tree_map(lambda x: jnp.zeros((x.size, k)), x)
        outer_Av = tree_map(lambda x: jnp.zeros((x.size, k)), x)
    else:  # outer_v provided
        outer_v = tree_map(lambda x: jnp.atleast_2d(x.T).T, outer_v)
        lv = tree_leaves(outer_v)[0].shape[-1]  # number of supplied vs
        if outer_Av is None:
            outer_Av = jax.vmap(matvec, in_axes=1, out_axes=1)(outer_v)
            nmv += lv
        # pad to full size
        outer_v = tree_map(lambda x: jnp.pad(x, ((0, 0), (0, k - lv))), outer_v)
        outer_Av = tree_map(lambda x: jnp.pad(x, ((0, 0), (0, k - lv))), outer_Av)

    def lgmres_cond(carry):
        j_outer, nmv, x, r, beta, _, _, _, _ = carry
        return jnp.logical_and(j_outer < maxiter, beta > tol)

    def lgmres_loop(carry):
        j_outer, nmv, x, r, beta, outer_v, outer_Av, lv, ptol_max_factor = carry

        # -- inner LGMRES iteration
        v0 = lpsolve(r)
        inner_res_0 = _norm(v0)

        v0 = _mul(1.0 / inner_res_0, v0)
        ptol = jnp.minimum(ptol_max_factor, tol / beta)

        H, B, V, Z, y, _, nmv_inner, pres = _fgmres(
            matvec,
            v0=v0,
            m=m,
            k=0,
            lpsolve=lpsolve,
            rpsolve=rpsolve,
            atol=ptol,
            outer_v=outer_v,
            outer_Av=outer_Av,
            lv=lv,
            print_every=print_every,
        )
        y *= inner_res_0
        nmv += nmv_inner

        # Inner loop tolerance control
        ptol_max_factor = jnp.where(
            pres > ptol,
            jnp.minimum(1.0, 1.5 * ptol_max_factor),
            jnp.maximum(eps, 0.25 * ptol_max_factor),
        )

        # -- GMRES terminated: eval solution
        # dx = Z y
        dx = tree_map(lambda x: _dot(x, y), Z)

        # -- Store LGMRES augmentation vectors
        nx = _norm(dx)
        # ax = V H y
        ax = tree_map(lambda x: _dot(x, _dot(H.T, y)), V)
        outer_v = tree_map(_roll_prepend, outer_v, _mul(1 / nx, dx))
        outer_Av = tree_map(_roll_prepend, outer_Av, _mul(1 / nx, ax))
        lv = jnp.minimum(lv + 1, k)

        # -- Apply step
        x = _add(x, dx)
        r = _sub(b, matvec(x))
        nmv += 1
        beta = _norm(r)

        _maybe_print(print_every < jnp.inf, j_outer + 1, beta / b_norm, pre="LGMRES  ")

        return j_outer + 1, nmv, x, r, beta, outer_v, outer_Av, lv, ptol_max_factor

    carry = (0, nmv, x, r, beta, outer_v, outer_Av, lv, ptol_max_factor)
    carry = lax.while_loop(lgmres_cond, lgmres_loop, carry)
    j_outer, nmv, x, r, beta, outer_v, outer_Av, _, _ = carry

    return x, j_outer, nmv, beta, outer_v, outer_Av
