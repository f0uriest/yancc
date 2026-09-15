"""Generic-ish functions for 1D rootfinding (eg for ambipolar Er)"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix
from jaxtyping import Array, PyTree, Scalar
from optimistix import RESULTS, AbstractRootFinder


class _Newton1DStatefullState(eqx.Module):
    f: Scalar
    diff: Scalar
    result: RESULTS
    step: Scalar
    aux: PyTree
    oob_count: jax.Array
    clipped: jax.Array


class Newton1DStatefull(AbstractRootFinder):
    """Find the root of a 1d function of the form f, new_state = fun(y, old_state)"""

    ftol: float
    xatol: float
    xrtol: float
    atol: float
    rtol: float
    norm: Callable
    verbose: bool
    kappa: jax.Array

    def __init__(self, ftol, xatol, xrtol, verbose=False, kappa=1e-4):
        self.ftol = ftol
        self.xatol = xatol
        self.xrtol = xrtol
        self.atol = xatol
        self.rtol = xrtol
        self.norm = optimistix.max_norm
        self.verbose = verbose
        self.kappa = jnp.asarray(kappa)

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
        del fn, y, options, tags
        dtype = f_struct.dtype
        assert jax.tree_util.tree_structure(aux_struct) == jax.tree_util.tree_structure(
            args
        )
        return _Newton1DStatefullState(
            f=jnp.array(jnp.inf, dtype=dtype),
            diff=jnp.array(jnp.inf, dtype=dtype),
            result=RESULTS.successful,
            step=jnp.array(0),
            aux=args,
            oob_count=jnp.zeros(2).astype(jnp.int32),
            clipped=jnp.array(False),
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
        del options
        fx, df, aux = jax.jvp(
            lambda _y: fn(_y, state.aux), [y], [jnp.ones_like(y)], has_aux=True
        )
        diff = jnp.sign(df) * fx / (jnp.abs(df) + self.kappa)
        new_y = y - diff
        # we want to keep the iterates in the bounds, but simply clipping to the bounds
        # can lead to stagnation if the derivative on the boundary wants to keep
        # pushing farther out. Instead we keep track of how often it goes oob and move
        # the iterates towards the other bound on each retry.
        oob_lower = new_y < lower
        oob_upper = new_y > upper
        clipped = oob_lower | oob_upper
        oob = jnp.array([oob_lower, oob_upper])
        oob_count = jnp.where(oob, state.oob_count + oob, state.oob_count)
        new_yc = jnp.where(
            oob_lower,
            lower + oob_count[0] * (upper - lower) / (oob_count[0] + 1),
            new_y,
        )
        new_yc = jnp.where(
            oob_upper,
            upper - oob_count[1] * (upper - lower) / (oob_count[1] + 1),
            new_yc,
        )
        diff = y - new_yc
        new_state = _Newton1DStatefullState(
            f=fx,
            diff=diff,
            result=RESULTS.promote(lx.RESULTS.successful),
            step=state.step + 1,
            aux=aux,
            oob_count=oob_count,
            clipped=clipped,
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
        del fn, args, options, tags
        # Compare `f_val` against 0, not against some `f_prev`. This is because
        # we're doing a root-find and know that we're aiming to get close to zero.
        # Note that this does mean that the `rtol` is ignored in f-space, and only
        # `atol` matters.
        y_scale = self.xatol + self.xrtol * jnp.abs(y)
        f_scale = self.ftol
        y_converged = (jnp.abs(state.diff) < y_scale) & ~state.clipped
        f_converged = jnp.abs(state.f) < f_scale
        terminate = y_converged | f_converged

        terminate_result = RESULTS.successful
        linsolve_fail = state.result != RESULTS.successful
        result = RESULTS.where(linsolve_fail, state.result, terminate_result)
        terminate = linsolve_fail | terminate
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
        del fn, aux, args, options, tags, result
        return y, state.aux, {}


class _DeflatedFun1D(eqx.Module):
    fn: Callable
    roots: jax.Array

    def regularizer(self, x: jax.Array):
        dx = x - self.roots
        # each deflation makes f smaller because we're dividing by x.
        # this can lead to fake roots where f is smaller than the root finding tol
        # To combat this, we rescale f after each deflation by the magnitude of the
        # found root. This is only a heuristic, and may still fail if searching very
        # far from known roots. But assuming the domain is bounded this seems to do
        # reasonably well
        scale = jnp.where(
            jnp.isfinite(self.roots) & (jnp.abs(self.roots) > 1),
            jnp.abs(self.roots),
            jnp.array(1.0),
        )
        mask = jnp.isfinite(dx)
        y: jax.Array = jnp.where(mask, 1 / dx, jnp.ones_like(dx))
        return jnp.prod(y) * jnp.prod(scale)

    def __call__(self, x, *args, **kwargs):
        f, aux = self.fn(x, *args, **kwargs)
        r = self.regularizer(x)
        return r * f, aux


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
    full_output=False,
    verbose=False,
):
    """Find multiple roots x where fun(x, args) == 0.

    Uses deflation (dividing out already found roots).

    The function is assumed to be stateful, with the state returned as a second
    output, eg
    f, new_state = fun(x, old_state)

    Parameters
    ----------
    fun : callable
        Function to find the root of. Should have a signature of the form
        fun(x, args)- > float, aux.
    x0 : float
        Initial guess for the root.
    nroots : int
        Number of roots to find.
    args : tuple, optional
        Initial state to pass to fun.
    ftol : float, optional
        Absolute stopping tolerance on f. Stops when abs(fun(x)) < ftol.
    xatol : float, optional
        Absolute stopping tolerance on change in x. Stops when abs(dx) < xatol.
    xrtol : float, optional
        Relative stopping tolerance on change in x. Stops when abs(dx) < xrtol * abs(x).
    maxiter : int > 0, optional
        Maximum number of iterations.
    full_output : bool, optional
        If True, also return a tuple where the first element is the residual from
        the root finding and the second is the number of iterations.
    verbose : bool, optional
        Whether to print iteration info.

    Returns
    -------
    xk : jax.Array
        Roots. Invalid values will be replaced with jnp.inf
    info : tuple of (bool, float, int)
        Success flag, residual of fun at xk and number of iterations for each deflation.

    """
    xs = jnp.full(num_roots, jnp.inf)
    fs = jnp.full(num_roots, jnp.inf)
    ss = jnp.full(num_roots, False)
    ks = jnp.full(num_roots, 0)

    _, aux_struct = jax.eval_shape(fun, x0, args)
    auxs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.full_like(x, jnp.inf)[None], num_roots, axis=0),
        aux_struct,
    )

    def condfun(state):
        i, xs, fs, ss, ks, auxs = state
        stop = (i > num_roots) | ((i > 0) & ~ss[i - 1])
        return ~stop

    def bodyfun(state):
        i, xs, fs, ss, ks, auxs = state

        deflated_fun = _DeflatedFun1D(fun, xs)

        sol = optimistix.root_find(
            deflated_fun,
            Newton1DStatefull(ftol, xatol, xrtol, verbose=verbose > 1, kappa=1e-6),
            y0=x0,
            has_aux=True,
            args=args,
            options={"lower": bounds[0], "upper": bounds[1]},
            throw=False,
            max_steps=maxiter,
        )
        if verbose:
            jax.debug.print(
                "Deflation {i:3d}, x={x: .4e}, f={f: .4e}, steps={k:3d}",
                i=i,
                x=sol.value,
                f=sol.state.f / deflated_fun.regularizer(sol.value),
                k=sol.stats["num_steps"],
                ordered=True,
            )

        def root_found(solution):
            # deflation modifies the original function like 1/x*f, so far from previous
            # roots the deflated function is small, which may lead to spurious roots.
            # if we find one we double check by using it as an initial guess for root
            # finding with the original function
            sol = optimistix.root_find(
                fun,
                Newton1DStatefull(ftol, xatol, xrtol, verbose=verbose > 1, kappa=1e-6),
                y0=solution.value,
                has_aux=True,
                args=solution.state.aux,
                options={"lower": bounds[0], "upper": bounds[1]},
                throw=False,
                max_steps=10,
            )
            return sol

        def root_not_found(solution):
            return solution

        sol = jax.lax.cond(
            (sol.result == RESULTS.successful), root_found, root_not_found, sol
        )

        x = sol.value
        f = sol.state.f
        # if we converged to a previously found root we consider that a failure and
        # stop usually this means deflation found a spurious root, and the refinement
        # brought us to one we've already seen. If we continue, we'll just keep
        # getting the same roots.
        status = (sol.result == RESULTS.successful) & ~jnp.isclose(
            x, xs, rtol=xrtol, atol=xatol
        ).any()
        k = sol.stats["num_steps"]
        aux = sol.state.aux

        xs = jnp.where(status, xs.at[i].set(x), xs)
        fs = jnp.where(status, fs.at[i].set(f), fs)
        ss = ss.at[i].set(status)
        ks = ks.at[i].set(k)
        if verbose:
            jax.debug.print(
                "Deflation {i:3d}, x={x: .4e}, f={f: .4e}, steps={k:3d}",
                i=i,
                x=xs[i],
                f=fs[i],
                k=ks[i],
                ordered=True,
            )
        auxs = jax.tree.map(
            lambda x, y: jnp.where(status, x.at[i].set(y), x), auxs, aux
        )
        return i + 1, xs, fs, ss, ks, auxs

    state = (0, xs, fs, ss, ks, auxs)

    i, xs, fs, ss, ks, auxs = eqx.internal.while_loop(
        condfun, bodyfun, state, max_steps=num_roots, kind="bounded"
    )
    if full_output:
        return xs, (i, ss, fs, ks, auxs)

    return xs
