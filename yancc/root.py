"""Generic-ish functions for 1D rootfinding (eg for ambipolar Er)"""

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix
from jaxtyping import Array, PyTree, Scalar
from optimistix import RESULTS, AbstractRootFinder


class _Newton1DStatefullState(eqx.Module):
    y: Scalar
    f: Scalar
    diff: Scalar
    result: RESULTS
    step: Scalar
    aux: PyTree
    oob_count: jax.Array
    clipped: jax.Array
    fprev: Scalar
    xpos: Scalar
    xneg: Scalar
    bisected: jax.Array
    fbest: Scalar
    nstall: jax.Array


class Newton1DStatefull(AbstractRootFinder):
    """Find the root of a 1d function of the form f, new_state = fun(y, old_state)

    Once the evaluated points bracket a root, a step that would leave the bracket is
    replaced by bisection.

    With ``secant=True``, the derivative is estimated from the last two evaluations
    where that is reliable, and computed by forward mode differentiation otherwise.
    """

    ftol: float
    xatol: float
    xrtol: float
    atol: float
    rtol: float
    norm: Callable
    verbose: bool
    kappa: jax.Array
    max_stall: jax.Array
    secant: bool = eqx.field(static=True)

    def __init__(
        self,
        ftol,
        xatol,
        xrtol,
        verbose=False,
        kappa=1e-6,
        max_stall=jnp.inf,
        secant=False,
    ):
        self.ftol = ftol
        self.xatol = xatol
        self.xrtol = xrtol
        self.atol = xatol
        self.rtol = xrtol
        self.norm = optimistix.max_norm
        self.verbose = verbose
        self.kappa = jnp.asarray(kappa)
        self.max_stall = jnp.asarray(max_stall)
        self.secant = secant

    def init(
        self,
        fn,
        y,
        args,
        options,
        f_struct,
        aux_struct,
        tags=frozenset(),
    ):
        """Initialize the solver."""
        del fn, options, tags
        dtype = f_struct.dtype
        assert jax.tree_util.tree_structure(aux_struct) == jax.tree_util.tree_structure(
            args
        )
        return _Newton1DStatefullState(
            y=y,
            f=jnp.array(jnp.inf, dtype=dtype),
            diff=jnp.array(jnp.inf, dtype=dtype),
            result=RESULTS.successful,
            step=jnp.array(0),
            aux=args,
            oob_count=jnp.zeros(2).astype(jnp.int32),
            clipped=jnp.array(False),
            fprev=jnp.array(jnp.inf, dtype=dtype),
            xpos=jnp.array(jnp.nan, dtype=dtype),
            xneg=jnp.array(jnp.nan, dtype=dtype),
            bisected=jnp.array(False),
            fbest=jnp.array(jnp.inf, dtype=dtype),
            nstall=jnp.zeros((), dtype=int),
        )

    def step(
        self,
        fn,
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _Newton1DStatefullState,
        tags: frozenset[object] = frozenset(),
    ) -> tuple[Scalar, _Newton1DStatefullState, PyTree]:
        """Take 1 step of the solver to update y and state"""
        del tags, args
        lower = options.get("lower", -jnp.inf)
        upper = options.get("upper", jnp.inf)
        maxstep = options.get("maxstep", jnp.inf)
        roots = options.get("roots", jnp.array([jnp.inf]))
        del options
        # A step that uses up the step budget only records f, so it needs no derivative.
        last = state.step + 1 >= maxstep
        # state.y and state.f are the previous evaluation, and state.fprev the one
        # before it.
        has_prev = jnp.isfinite(state.f) & (state.y != y)
        # A secant through two points is a good estimate of the derivative only while
        # the iteration is converging. It is replaced by the exact derivative at the
        # start, after a step that had to be pulled back into the bounds, and after a
        # step that didn't reduce |f|: the secant then spans a region where the slope
        # changes, and following it can lead the iterates away, most often where two
        # roots are close together.
        progress = jnp.abs(state.f) < jnp.abs(state.fprev)
        use_secant = (
            self.secant & has_prev & ~state.clipped & ~state.bisected & progress
        )

        def exact(y):
            return jax.jvp(
                lambda _y: fn(_y, state.aux), [y], [jnp.ones_like(y)], has_aux=True
            )

        def approximate(y):
            fx, aux = fn(y, state.aux)
            dy = jnp.where(has_prev, y - state.y, 1.0)
            df = jnp.where(has_prev, (fx - state.f) / dy, 0.0)
            return fx, df, aux

        fx, df, aux = jax.lax.cond(use_secant | last, approximate, exact, y)
        # The sign of the derivative sets the step direction while its magnitude is
        # regularized, so a derivative that is exactly zero would otherwise give a
        # step of exactly zero, which the step size test below reads as convergence.
        # Functions that are constant over an interval do occur, for instance a
        # residual normalized by a sum of magnitudes saturates where the terms share
        # a sign, so pick a direction and take the full regularized step, leaving the
        # flat region through the out of bounds handling if it extends to a bound.
        # The newest point on each side of the root: together they bracket it. A point
        # at an already found root is not used, and neither is an interval containing
        # one: deflation cancels a known root only as well as the tolerance it was
        # found with, so a sign changing spike survives there, and bisecting into it
        # walks back to the root already found.
        # Iterates that stop reducing abs(f) are going nowhere: without a bracket to
        # bisect, a step that keeps missing can settle into a cycle and spend the whole
        # budget, which is how a search for a root that isn't there ends up costing as
        # much as one that converges.
        fbest = jnp.minimum(state.fbest, jnp.abs(fx))
        nstall = jnp.where(jnp.abs(fx) < state.fbest, 0, state.nstall + 1)
        at_root = jnp.isclose(y, roots, rtol=self.xrtol, atol=self.xatol).any()
        usable = ~at_root & jnp.isfinite(fx)
        xpos = jnp.where(usable & (fx > 0), y, state.xpos)
        xneg = jnp.where(usable & (fx < 0), y, state.xneg)
        blo = jnp.minimum(xpos, xneg)
        bhi = jnp.maximum(xpos, xneg)
        spans_root = ((roots > blo) & (roots < bhi)).any()
        has_bracket = jnp.isfinite(xpos) & jnp.isfinite(xneg) & ~spans_root
        sign = jnp.where(df == 0, jnp.ones_like(df), jnp.sign(df))
        diff = sign * fx / (jnp.abs(df) + self.kappa)
        new_y = y - diff
        # Once the root is bracketed, a step that leaves the bracket is replaced by its
        # midpoint. Bisection always makes progress, so the iteration can't be led away
        # by a slope that points outside, and the bracket also keeps the iterates within
        # the bounds, leaving the out of bounds handling below for the unbracketed case.
        bisected = has_bracket & ((new_y <= blo) | (new_y >= bhi))
        new_y = cast(jax.Array, jnp.where(bisected, (xpos + xneg) / 2, new_y))
        # The first step that leaves the bounds through a given side is damped to go
        # halfway from the current iterate to that bound. This keeps the Newton
        # direction, which typically overshoots where the function flattens out
        # before a root. If the iterates keep leaving through the same side, the
        # derivative near that bound is pushing outwards, so instead we move the
        # iterates progressively further towards the other bound on each retry to
        # explore the rest of the domain.
        oob_lower = new_y < lower
        oob_upper = new_y > upper
        clipped = oob_lower | oob_upper
        oob = jnp.array([oob_lower, oob_upper])
        oob_count = state.oob_count + oob
        sweep_lower = lower + oob_count[0] * (upper - lower) / (oob_count[0] + 1)
        sweep_upper = upper - oob_count[1] * (upper - lower) / (oob_count[1] + 1)
        new_yc = cast(
            jax.Array,
            jnp.where(
                oob_lower,
                jnp.where(oob_count[0] == 1, y + (lower - y) / 2, sweep_lower),
                new_y,
            ),
        )
        new_yc = cast(
            jax.Array,
            jnp.where(
                oob_upper,
                jnp.where(oob_count[1] == 1, y + (upper - y) / 2, sweep_upper),
                new_yc,
            ),
        )
        diff = y - new_yc
        # without a derivative the step isn't meaningful, so it can't signal convergence
        diff = cast(jax.Array, jnp.where(last & ~use_secant, jnp.inf, diff))
        new_state = _Newton1DStatefullState(
            y=y,
            f=fx,
            diff=diff,
            result=RESULTS.promote(lx.RESULTS.successful),
            step=state.step + 1,
            aux=aux,
            oob_count=oob_count,
            clipped=clipped,
            fprev=state.f,
            xpos=xpos,
            xneg=xneg,
            bisected=bisected,
            fbest=fbest,
            nstall=nstall,
        )
        if self.verbose:
            jax.debug.print(
                "Newton step {i:3d},  x={x: .6e},  xc={xc: .6e},  "
                "df={df: .6e},  f={f: .6e},  oob={oobl:3d},{oobu:3d}",
                i=state.step,
                xc=new_yc,
                x=new_y,
                df=df,
                f=fx,
                oobl=oob_count[0],
                oobu=oob_count[1],
            )
        return new_yc, new_state, aux

    def terminate(
        self,
        fn,
        y: PyTree[Array],
        args: PyTree,
        options: dict[str, Any],
        state: _Newton1DStatefullState,
        tags: frozenset[object] = frozenset(),
    ):
        """Check if we should stop iterating."""
        del fn, args, tags
        # A budget below the maximum number of steps is used to sample a point without
        # committing to a full search from it.
        maxstep = options.get("maxstep", jnp.inf)
        # Compare `f_val` against 0, not against some `f_prev`. This is because
        # we're doing a root-find and know that we're aiming to get close to zero.
        # Note that this does mean that the `rtol` is ignored in f-space, and only
        # `atol` matters.
        y_scale = self.xatol + self.xrtol * jnp.abs(y)
        f_scale = self.ftol
        y_converged = (jnp.abs(state.diff) < y_scale) & ~state.clipped
        f_converged = jnp.abs(state.f) < f_scale
        terminate = y_converged | f_converged

        # Repeatedly stepping out of bounds means there is likely no root in the
        # bounds reachable from here, so give up early rather than bouncing around.
        # A bracketed search can always fall back on bisection, so only the step budget
        # stops it.
        roots = options.get("roots", jnp.array([jnp.inf]))
        blo = jnp.minimum(state.xpos, state.xneg)
        bhi = jnp.maximum(state.xpos, state.xneg)
        has_bracket = (
            jnp.isfinite(blo)
            & jnp.isfinite(bhi)
            & ~((roots > blo) & (roots < bhi)).any()
        )
        diverged = ((state.nstall > self.max_stall) & ~has_bracket) | (
            state.step >= maxstep
        )
        terminate_result = RESULTS.where(
            diverged & ~(y_converged | f_converged),
            RESULTS.nonlinear_divergence,
            RESULTS.successful,
        )
        linsolve_fail = state.result != RESULTS.successful
        result = RESULTS.where(linsolve_fail, state.result, terminate_result)
        terminate = linsolve_fail | terminate | diverged
        return terminate, result

    def postprocess(
        self,
        fn,
        y: PyTree[Array],
        aux: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: _Newton1DStatefullState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[PyTree[Array], PyTree, dict[str, Any]]:
        """Do any postprocessing on y, aux."""
        del fn, y, aux, args, options, tags, result
        # The last function value and aux were computed at state.y, before the final
        # update. Return that point so that y, f and aux are all consistent.
        return state.y, state.aux, {}


class _DeflatedFun1D(eqx.Module):
    fn: Callable
    roots: jax.Array
    length: jax.Array

    def regularizer(self, x: jax.Array):
        # Shifted deflation operator, prod_i (L / (x - r_i) + 1). Distances are
        # measured relative to a length scale L of the search domain, so that away
        # from known roots the operator is O(1) regardless of the units of x and the
        # deflated function keeps roughly the magnitude of the original, rather than
        # being driven towards zero (which creates spurious roots) or blowing up.
        # The signed distance is used (rather than |x - r_i| as is common for systems)
        # so that the sign change of the operator cancels the sign change of f at a
        # simple root, keeping the deflated function continuous there. Otherwise it
        # has a jump at each known root that Newton iterations can't cross. The only
        # zero introduced is at x = r_i - L, which is outside the domain when L is the
        # domain width.
        dx = x - self.roots
        mask = jnp.isfinite(dx)
        # evaluating exactly at a known root would give 0 * inf
        tiny = jnp.finfo(dx.dtype).eps * self.length
        dx = jnp.where(mask, jnp.where(dx == 0, tiny, dx), 1.0)
        y: jax.Array = jnp.where(mask, self.length / dx + 1, 1.0)
        return jnp.prod(y)

    def __call__(self, x, *args, **kwargs):
        f, aux = self.fn(x, *args, **kwargs)
        r = self.regularizer(x)
        return r * f, aux


def _insert_history(hx, hf, x, f):
    """Insert (x, f) into sorted history arrays, dropping the least useful point.

    Empty slots are marked by hx = inf. When full, the point dropped is preferably an
    interior point with the same sign as both neighbors (it doesn't bound a sign
    change), with the smallest spacing to its neighbors.
    """
    size = hx.size
    skip = jnp.any(hx == x) | ~jnp.isfinite(f) | ~jnp.isfinite(x)
    ax = jnp.append(hx, x)
    af = jnp.append(hf, f)
    order = jnp.argsort(cast(jax.Array, ax))
    ax, af = cast(jax.Array, ax)[order], cast(jax.Array, af)[order]
    valid = jnp.isfinite(ax)
    idx = jnp.arange(size + 1)
    nvalid = valid.sum()
    interior = (idx > 0) & (idx < nvalid - 1)
    sign = jnp.sign(af)
    redundant = interior & (jnp.roll(sign, 1) == sign) & (jnp.roll(sign, -1) == sign)
    # spacing is only meaningful for interior points, whose neighbors are both valid
    safe_x = jnp.where(valid, ax, 0.0)
    spacing = jnp.minimum(safe_x - jnp.roll(safe_x, 1), jnp.roll(safe_x, -1) - safe_x)
    span = jnp.where(valid, ax, -jnp.inf).max() - jnp.where(valid, ax, jnp.inf).min()
    score = jnp.where(
        ~valid,
        -jnp.inf,
        jnp.where(
            interior,
            jnp.where(redundant, spacing, spacing + 2 * span),
            jnp.inf,
        ),
    )
    drop = jnp.argmin(cast(jax.Array, score))
    take = jnp.arange(size)
    take = jnp.where(take >= drop, take + 1, take)
    return jnp.where(skip, hx, ax[take]), jnp.where(skip, hf, af[take])


def _widest_gap(hx, lower, upper):
    """Midpoint of the widest interval between recorded points, bounds included.

    Sign changes can only reveal an odd number of roots, so a pair of roots close
    together is invisible until a point is recorded between them. Splitting the widest
    unexplored interval covers the domain in as few evaluations as possible.
    """
    xs = jnp.concatenate([hx, jnp.stack([lower, upper])])
    xs = jnp.sort(jnp.where(jnp.isfinite(xs), xs, jnp.inf))
    gaps = jnp.diff(xs)
    gaps = jnp.where(jnp.isfinite(gaps), gaps, -jnp.inf)
    j = jnp.argmax(cast(jax.Array, gaps))
    return (xs[j] + xs[j + 1]) / 2


def _best_point(hx, hf, roots, ftol, xrtol, xatol):
    """Recorded point with the smallest abs(f) that isn't at an already found root.

    Where no sign change points at a root, the closest approach to zero recorded so far
    is the best place to search from: it is the one point known to be near a root.
    """
    at_root = jnp.isclose(hx[:, None], roots[None, :], rtol=xrtol, atol=xatol).any(
        axis=1
    )
    usable = jnp.isfinite(hx) & (jnp.abs(hf) >= ftol) & ~at_root
    score = jnp.where(usable, jnp.abs(hf), jnp.inf)
    return hx[jnp.argmin(cast(jax.Array, score))], usable.any()


def _find_bracket(hx, hf, roots, ftol, xrtol, xatol):
    """Find the narrowest interval of sorted history containing an unfound root.

    An interval between adjacent points contains an odd number of unfound roots if
    the sign change of f across it isn't accounted for by the known roots inside it.
    Points with abs(f) < ftol are not used, since their sign is unreliable, and
    neither are points at an already found root: f there is only zero to the tolerance
    of the search, so its sign is arbitrary, and an interval ending at one would
    otherwise look like it held a sign change that no known root explains.
    """
    at_root = jnp.isclose(hx[:, None], roots[None, :], rtol=xrtol, atol=xatol).any(
        axis=1
    )
    usable = jnp.isfinite(hx) & (jnp.abs(hf) >= ftol) & ~at_root
    # compress usable points to the front, keeping them sorted
    order = jnp.argsort(cast(jax.Array, jnp.where(usable, hx, jnp.inf)))
    px, pf, pu = hx[order], hf[order], usable[order]
    a, b, fa, fb = px[:-1], px[1:], pf[:-1], pf[1:]
    valid = pu[:-1] & pu[1:]
    known = jnp.isfinite(roots)
    inside = (
        known[None, :] & (roots[None, :] > a[:, None]) & (roots[None, :] < b[:, None])
    )
    parity = jnp.where(inside.sum(axis=1) % 2 == 0, 1.0, -1.0)
    has_root = valid & (jnp.sign(fa) * jnp.sign(fb) * parity < 0)
    j = jnp.argmin(cast(jax.Array, jnp.where(has_root, b - a, jnp.inf)))
    aj, bj, faj, fbj = a[j], b[j], fa[j], fb[j]
    width = bj - aj
    no_known = inside[j].sum() == 0
    safe_df = jnp.where(fbj != faj, fbj - faj, 1.0)
    falsi = jnp.clip(aj - faj * width / safe_df, aj + 0.1 * width, bj - 0.1 * width)
    start = jnp.where(no_known, falsi, (aj + bj) / 2)
    return has_root.any(), cast(jax.Array, start), aj, bj


@eqx.filter_jit
def deflated_root_scalar(
    fun,
    x0,
    num_roots,
    args=(),
    bounds=(-jnp.inf, jnp.inf),
    ftol=jnp.array(1e-6),
    xatol=jnp.array(0.0),
    xrtol=jnp.array(1e-6),
    maxiter=20,
    max_stall=3,
    probe_steps=1,
    interior_samples=6,
    best_searches=2,
    history_size=12,
    method="secant",
    carry_state=False,
    full_output=False,
    verbose: bool | int = False,
):
    """Find multiple roots x where fun(x, args) == 0.

    Roots are found one at a time by a Newton type method, using deflation to avoid
    converging to already found roots. The first searches start from the given guesses,
    one each. The points where fun is evaluated are recorded, and when they show a sign
    change that isn't explained by the known roots, the next search is started inside
    that interval and restricted to it. When no sign change is left to follow, each of
    the bounds in turn is sampled to look for one, rather than searched from, and then
    searched from. A sign change only reveals an odd number of roots, so what can be
    left is a pair that none of the recorded points separates: the widest unexplored
    intervals are then sampled in turn to split it, and finally the recorded point that
    came closest to zero is searched from. The search stops when num_roots roots are
    found, or when all of that is exhausted.

    The function is assumed to be stateful, with the state returned as a second
    output, eg
    f, new_state = fun(x, old_state)

    Parameters
    ----------
    fun : callable
        Function to find the root of. Should have a signature of the form
        fun(x, args)- > float, aux.
    x0 : float or array of float
        Initial guesses, one per search. Searches beyond the number of guesses given
        start from a sign change in the recorded values, or from a bound.
    num_roots : int
        Number of roots to find.
    args : tuple, optional
        Initial state to pass to fun.
    bounds : tuple of float, optional
        Lower and upper bounds for the search.
    ftol : float, optional
        Absolute stopping tolerance on f. Stops when abs(fun(x)) < ftol.
    xatol : float, optional
        Absolute stopping tolerance on change in x. Stops when abs(dx) < xatol.
    xrtol : float, optional
        Relative stopping tolerance on change in x. Stops when abs(dx) < xrtol * abs(x).
    maxiter : int > 0, optional
        Maximum number of iterations per search.
    max_stall : int, optional
        A search is stopped as unsuccessful once this many consecutive iterations fail
        to reduce abs(f) below the smallest value it has reached, which covers both
        iterates that settle into a cycle and iterates pushed against a bound. A search
        whose iterates bracket a root is never stopped this way, since it can always
        fall back on bisection.
    probe_steps : int, optional
        Number of iterations allowed when sampling a bound, which is done to look for
        a sign change rather than to converge to a root.
    interior_samples : int, optional
        Number of points sampled inside the bounds, one per search, once the guesses,
        the sign changes and both bounds are exhausted. Each splits the widest interval
        not yet explored, which is the only way to find a pair of roots that no sign
        change points at.
    best_searches : int, optional
        Number of searches started from the recorded point with the smallest abs(f),
        run once the interior samples are used up.
    history_size : int, optional
        Number of evaluated points kept for detecting sign changes.
    method : {"secant", "newton"}, optional
        How the derivative of fun is found at each iteration. "newton" differentiates
        fun in forward mode at every iteration. "secant" estimates it from the last
        two evaluations while the iteration is making progress, and differentiates fun
        otherwise. It usually needs a similar number of iterations with far fewer
        derivatives, but is less reliable at finding roots that are close together.
    carry_state : bool, optional
        If False, each search starts from ``args``. If True, each search starts from
        the state returned by the previous search, so that information accumulated in
        the state, such as a warm start for an iterative solver, carries over.
    full_output : bool, optional
        If True, also return additional information about the search.
    verbose : bool, optional
        Whether to print iteration info.

    Returns
    -------
    xk : jax.Array, shape (num_roots,)
        Roots, in the order they were found. Missing roots are set to jnp.inf.
    info : tuple
        Only returned if full_output is True. Contains the number of searches, and for
        each root the success flag, residual of fun at xk, number of iterations, and
        the state returned by fun at xk. The last element is a tuple containing, for
        each search (up to ``num_roots + len(x0) + 4 + interior_samples +
        best_searches``), whether the search was run,
        and the final state returned by fun.

    """
    if method not in ("secant", "newton"):
        raise ValueError(f"method must be 'secant' or 'newton', got '{method}'")
    dtype = jnp.asarray(x0, dtype=float).dtype
    starts = jnp.atleast_1d(jnp.asarray(x0, dtype=dtype))
    nguess = starts.size
    lower = jnp.asarray(bounds[0], dtype=dtype)
    upper = jnp.asarray(bounds[1], dtype=dtype)
    width = upper - lower
    length = jnp.where(jnp.isfinite(width), width, 1.0)
    # Searches that only sample a bound don't find roots, so they get their own budget.
    # Guesses likewise: several of them can lead to the same root, and the fallback
    # still needs its full budget afterwards to reach the roots they missed.
    max_searches = num_roots + nguess + 4 + interior_samples + best_searches

    def recorded_fun(x, state):
        user_state, (hx, hf) = state
        f, user_aux = fun(x, user_state)
        hx, hf = _insert_history(
            hx, hf, jax.lax.stop_gradient(x), jax.lax.stop_gradient(f)
        )
        return f, (user_aux, (hx, hf))

    def _invalid(x, n):
        # inf has no integer representation, so integer leaves are filled with zero
        fill = jnp.inf if jnp.issubdtype(x.dtype, jnp.inexact) else 0
        return jnp.repeat(jnp.full_like(x, fill)[None], n, axis=0)

    _, aux_struct = jax.eval_shape(fun, starts[0], args)
    xs = jnp.full(num_roots, jnp.inf, dtype=dtype)
    fs = jnp.full(num_roots, jnp.inf, dtype=dtype)
    ks = jnp.zeros(num_roots, dtype=int)
    auxs = jax.tree.map(lambda x: _invalid(x, num_roots), aux_struct)
    searched = jnp.zeros(max_searches, dtype=bool)
    search_auxs = jax.tree.map(lambda x: _invalid(x, max_searches), aux_struct)
    history = (
        jnp.full(history_size, jnp.inf, dtype=dtype),
        jnp.full(history_size, jnp.nan, dtype=dtype),
    )

    solver = Newton1DStatefull(
        ftol,
        xatol,
        xrtol,
        verbose=verbose > 1,
        kappa=1e-6,
        max_stall=max_stall,
        secant=method == "secant",
    )

    def phase(carry, inputs):
        # One Newton solve of fun deflated by the given roots, skipped if the previous
        # phase failed. Deflating by all-inf roots gives the original function.
        x, f, aux, k, ok = carry
        i, p, roots, lo, hi, budget = inputs
        deflated_fun = _DeflatedFun1D(recorded_fun, roots, length)

        def solve(x, aux):
            sol = optimistix.root_find(
                deflated_fun,
                solver,
                y0=x,
                has_aux=True,
                args=aux,
                options={
                    "lower": lo,
                    "upper": hi,
                    "maxstep": budget,
                    "roots": roots,
                },
                throw=False,
                max_steps=maxiter,
            )
            return (
                sol.value,
                sol.state.f / deflated_fun.regularizer(sol.value),
                sol.state.aux,
                k + sol.stats["num_steps"].astype(k.dtype),
                sol.result == RESULTS.successful,
            )

        def skip(x, aux):
            return x, f, aux, k, jnp.array(False)

        x, f, aux, k, ok = jax.lax.cond(ok, solve, skip, x, aux)
        if verbose > 1:
            jax.debug.print(
                "Search {i:3d}, phase {p:1d}, x={x: .4e}, f={f: .4e}, steps={k:3d}",
                i=i,
                p=p,
                x=x,
                f=f,
                k=k,
                ordered=True,
            )
        return (x, f, aux, k, ok), None

    def condfun(state):
        i, nfound, xs, fs, ks, auxs, history, probed, tried, failed, bracket = state[
            :11
        ]
        nscan, nbest = state[14], state[15]
        exhausted = (
            failed
            & ~bracket
            & probed.all()
            & tried.all()
            & (nscan >= interior_samples)
            & (nbest >= best_searches)
            & (i >= nguess)
        )
        return (i < max_searches) & (nfound < num_roots) & ~exhausted

    def bodyfun(state):
        (
            i,
            nfound,
            xs,
            fs,
            ks,
            auxs,
            history,
            probed,
            tried,
            _,
            _,
            searched,
            search_auxs,
            search_args,
            nscan,
            nbest,
        ) = state
        has_bracket, bracket_start, bracket_lo, bracket_hi = _find_bracket(
            history[0], history[1], xs, ftol, xrtol, xatol
        )
        # The supplied guesses are used first, one per search. They carry information
        # the search cannot derive for itself, so they come before the bracket that
        # the recorded values may already show.
        use_guess = i < nguess
        # Without a sign change to follow, each bound is first sampled with a small
        # step budget: that records f there, which is often all that is needed to
        # bracket a root, and costs a fraction of a search that has nothing to
        # converge to. A sign change can only reveal an odd number of roots though, so
        # a bound whose sample brackets nothing is searched from afterwards, which is
        # the only way to reach a pair of roots that the recorded signs can't see.
        loose = ~use_guess & ~has_bracket
        use_bracket = ~use_guess & has_bracket
        probe_lower = loose & ~probed[0]
        probe_upper = loose & probed[0] & ~probed[1]
        probing = probe_lower | probe_upper
        use_lower = loose & ~probing & ~tried[0]
        use_upper = loose & ~probing & tried[0] & ~tried[1]
        # Once both bounds have been searched from, what is left is a root pair that no
        # sign change can point at. Splitting the widest unexplored interval records a
        # point inside it, which brackets both of its roots, and costs one evaluation
        # rather than a whole search.
        sampling = loose & ~probing & tried.all() & (nscan < interior_samples)
        # With the domain sampled and still no sign change to follow, the recorded point
        # that came closest to zero is the one place known to be near a root, so it is
        # the best start left for a full search.
        best_start, have_best = _best_point(
            history[0], history[1], xs, ftol, xrtol, xatol
        )
        from_best = (
            loose
            & ~probing
            & ~sampling
            & tried.all()
            & have_best
            & (nbest < best_searches)
        )
        at_lower = probe_lower | use_lower
        at_upper = probe_upper | use_upper
        guess = starts[jnp.clip(i, 0, nguess - 1)]
        # the cases are mutually exclusive, so these are alternatives, not overrides
        start = cast(jax.Array, guess)
        start = cast(jax.Array, jnp.where(from_best, best_start, start))
        gap = _widest_gap(history[0], lower, upper)
        start = cast(jax.Array, jnp.where(sampling, gap, start))
        start = cast(jax.Array, jnp.where(at_upper, upper, start))
        start = cast(jax.Array, jnp.where(at_lower, lower, start))
        start = cast(jax.Array, jnp.where(use_bracket, bracket_start, start))
        lo = cast(jax.Array, jnp.where(use_bracket, bracket_lo, lower))
        hi = cast(jax.Array, jnp.where(use_bracket, bracket_hi, upper))
        probed = probed | jnp.array([probe_lower, probe_upper])
        tried = tried | jnp.array([use_lower, use_upper])
        budget = jnp.where(probing | sampling, probe_steps, maxiter)

        # Once deflation finds a root we use that as a warm start for a search with
        # the un-deflated function, to avoid returning spurious roots created by
        # deflation. Both phases run through a single scan so that fun is only traced
        # (and compiled) once. First pass deflates out found roots, second pass deflates
        # inf roots (ie, no deflation)
        roots = jnp.stack([xs, jnp.full_like(xs, jnp.inf)])
        carry = (
            start,
            jnp.asarray(jnp.inf, dtype=dtype),
            (search_args, history),
            jnp.zeros((), dtype=ks.dtype),
            jnp.array(True),
        )
        (x, f, (aux, history), k, ok), _ = jax.lax.scan(
            phase,
            carry,
            (
                jnp.full(2, i),
                jnp.arange(2),
                roots,
                jnp.full(2, lo),
                jnp.full(2, hi),
                jnp.full(2, budget),
            ),
        )

        # if we converged to a previously found root we consider that a failure.
        # Usually this means deflation found a spurious root, and the refinement
        # brought us to one we've already seen.
        status = ok & ~jnp.isclose(cast(jax.Array, x), xs, rtol=xrtol, atol=xatol).any()

        xs = jnp.where(status, xs.at[nfound].set(x), xs)
        fs = jnp.where(status, fs.at[nfound].set(f), fs)
        ks = jnp.where(status, ks.at[nfound].set(k), ks)
        auxs = jax.tree.map(
            lambda x, y: jnp.where(status, x.at[nfound].set(y), x), auxs, aux
        )
        searched = searched.at[i].set(True)
        search_auxs = jax.tree.map(lambda x, y: x.at[i].set(y), search_auxs, aux)
        if verbose:
            jax.debug.print(
                "Search {i:3d}: start={x0: .4e}, bounds=({lo: .4e},{hi: .4e}), "
                "success={s}, x={x: .4e}, f={f: .4e}, steps={k:3d}",
                i=i,
                x0=start,
                lo=lo,
                hi=hi,
                s=status,
                x=x,
                f=f,
                k=k,
                ordered=True,
            )
        has_bracket, *_ = _find_bracket(history[0], history[1], xs, ftol, xrtol, xatol)
        return (
            i + 1,
            nfound + status,
            xs,
            fs,
            ks,
            auxs,
            history,
            probed,
            tried,
            ~status,
            has_bracket,
            searched,
            search_auxs,
            aux if carry_state else search_args,
            nscan + sampling,
            nbest + from_best,
        )

    # bounds that are infinite can't be used as starting points
    tried = ~jnp.isfinite(jnp.array([lower, upper]))
    state = (
        0,
        0,
        xs,
        fs,
        ks,
        auxs,
        history,
        tried,
        tried,
        jnp.array(False),
        jnp.array(False),
        searched,
        search_auxs,
        args,
        jnp.array(0),
        jnp.array(0),
    )
    state = eqx.internal.while_loop(
        condfun, bodyfun, state, max_steps=max_searches, kind="bounded"
    )
    i, _, xs, fs, ks, auxs, *_, searched, search_auxs, _, _, _ = state
    if full_output:
        return xs, (i, jnp.isfinite(xs), fs, ks, auxs, (searched, search_auxs))

    return xs
