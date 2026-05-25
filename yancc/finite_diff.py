"""Functions for computing finite difference derivatives."""

import functools
import math
import operator

import jax
import jax.numpy as jnp
from jax import config

# need this here as well so that consts use 64 bit
config.update("jax_enable_x64", True)

y1b = 0.15
y3c = 0.2
y5c = 0.04

fd_coeffs = {
    # these are for forward difference,
    # with the reference point at the center of the stencil
    # to get backward version flip coeffs left/right and negate
    1: {
        "1a": jnp.array([0, -1, 1]),
        "1b": jnp.array([-1 / 2 + y1b, -2 * y1b, 1 / 2 + y1b]),
        "2a": jnp.array([0, 0, -3 / 2, 2, -1 / 2]),
        "2b": jnp.array([0, -1 / 4, -3 / 4, 5 / 4, -1 / 4]),  # Fromm's
        "2c": jnp.array([0, -3 / 8, -3 / 8, 7 / 8, -1 / 8]),  # QUICK
        "2d": jnp.array(  # optimized, d=0.88
            [0, 0, 0, 0, -5 / 4, 4 / 3, 0, 0, -1 / 12]
        ),
        "2e": jnp.array([0, 0, 0, -4 / 5, 3 / 4, 0, 0, 0, 1 / 20]),  # optimized, d=0.88
        "2f": jnp.array([0, 0, -3 / 4, 2 / 3, 0, 0, 1 / 12]),  # optimized, d=0.80
        "2g": jnp.array([0, 0, 0, -4 / 3, 3 / 2, 0, -1 / 6]),  # optimized, d=0.80
        "2z": jnp.array([-1 / 2, 0, 1 / 2]),  # centered, bad
        "3a": jnp.array([0, 0, 0, -11 / 6, 3, -3 / 2, 1 / 3]),
        "3b": jnp.array([0, -1 / 3, -1 / 2, 1, -1 / 6]),  # CUI
        "3c": jnp.array(
            [1 / 12 - y3c, -2 / 3 + 4 * y3c, -6 * y3c, 2 / 3 + 4 * y3c, -1 / 12 - y3c]
        ),
        "3d": jnp.array([0, -1 / 10, 0, -5 / 6, 1, 0, -1 / 15]),  # optimized, d=0.71
        "3e": jnp.array(
            [0, 2 / 21, 0, -6 / 5, 13 / 12, 0, 0, 0, 3 / 140]
        ),  # optimized, d=0.82
        "3f": jnp.array(
            [0, -1 / 21, 0, 0, -11 / 12, 1, 0, 0, -1 / 28]
        ),  # optimized, d=0.84
        "4a": jnp.array([0, 0, 0, 0, -25 / 12, 4, -3, 4 / 3, -1 / 4]),
        "4b": jnp.array([0, 0, -1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12]),
        "4d": jnp.array(  # optimized, d=0.62
            [0, 0, -1 / 15, 0, -13 / 12, 4 / 3, 0, -4 / 15, 1 / 12]
        ),
        "4e": jnp.array(
            [0, 0, 1 / 6, -16 / 15, 3 / 4, 0, 1 / 6, 0, -1 / 60]
        ),  # optimized, d=0.52
        "4z": jnp.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),  # centered, bad
        "5a": jnp.array([0, 0, 0, 0, 0, -137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5]),
        "5b": jnp.array([0, 1 / 20, -1 / 2, -1 / 3, 1, -1 / 4, 1 / 30]),
        "5c": jnp.array(
            [
                -1 / 60 + y5c,
                3 / 20 - 6 * y5c,
                -3 / 4 + 15 * y5c,
                -20 * y5c,
                3 / 4 + 15 * y5c,
                -3 / 20 - 6 * y5c,
                1 / 60 + y5c,
            ]
        ),
        "5d": jnp.array(  # optimized, d=0.53
            [0, 1 / 21, -1 / 5, 0, -3 / 4, 1 / 1, 0, -2 / 15, 1 / 28]
        ),
        "5e": jnp.array(
            [0, 4 / 105, -1 / 6, 0, -11 / 12, 4 / 3, -3 / 10, 0, 1 / 84]
        ),  # optimized, d=0.49
        "6a": jnp.array(
            [0, 0, 0, 0, 0, 0, -49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]
        ),
        "6b": jnp.array(
            [0, -5 / 168, 5 / 42, 0, -1, 4 / 5, 0, 0, 5 / 21, -9 / 56, 1 / 30]
        ),  # optimized, d=0.50
        "6c": jnp.array(
            [0, -5 / 648, 0, 5 / 21, -32 / 27, 4 / 5, 0, 5 / 27, 0, -1 / 24, 32 / 2835]
        ),  # optimized, d=0.48
        "6z": jnp.array(  # centered, bad
            [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]
        ),
        "7a": jnp.array(  # optimized, d=0.38
            [-3 / 280, 8 / 105, -1 / 5, 0, -1 / 1, 8 / 5, -3 / 5, 16 / 105, -1 / 56]
        ),
        "7b": jnp.array(  # optimized, d=0.42
            [
                0,
                -1 / 72,
                2 / 21,
                -5 / 21,
                0,
                -13 / 15,
                4 / 3,
                -1 / 3,
                0,
                5 / 168,
                -2 / 315,
            ]
        ),
    },
    # these are centered coeffs for 2nd derivatives.
    2: {
        2: jnp.array([1, -2, 1]),
        4: jnp.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
        6: jnp.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]),
        8: jnp.array(
            [
                -1 / 560,
                8 / 315,
                -1 / 5,
                8 / 5,
                -205 / 72,
                8 / 5,
                -1 / 5,
                8 / 315,
                -1 / 560,
            ]
        ),
    },
}

# these are kwargs to build_advection_matrix that give the same stencils
# as the above for uniform spacing. Note that 2b, 2c, 3b are kappa methods
# and can't easily be handled in the same way, but we never really use them so its fine.

fd_kwargs = {
    "1a": {
        "stencil": jnp.array([0, 1]),
        "order": 1,
    },
    "1b": {
        "stencil": jnp.array([-1, 0, 1]),
        "order": 1,
        "hyper_nu": jnp.array(0.15),
        "hyper_order": 2,
        "hyper_accuracy": 2,
        "scale_hyper_nu": 1,
    },
    "2a": {
        "stencil": jnp.array([0, 1, 2]),
        "order": 2,
    },
    "2d": {
        "stencil": jnp.array([0, 1, 4]),
        "order": 2,
    },
    "2e": {
        "stencil": jnp.array([-1, 0, 4]),
        "order": 2,
    },
    "2f": {
        "stencil": jnp.array([-1, 0, 3]),
        "order": 2,
    },
    "2g": {
        "stencil": jnp.array([0, 1, 3]),
        "order": 2,
    },
    "2z": {
        "stencil": jnp.array([-1, 0, 1]),
        "order": 2,
    },
    "3a": {
        "stencil": jnp.array([0, 1, 2, 3]),
        "order": 3,
    },
    "3c": {
        "stencil": jnp.array([-2, -1, 0, 1, 2]),
        "order": 4,
        "hyper_nu": jnp.array(0.025),
        "hyper_order": 4,
        "hyper_accuracy": 2,
        "scale_hyper_nu": 3,
    },
    "3d": {
        "stencil": jnp.array([-2, 0, 1, 3]),
        "order": 3,
    },
    "3e": {
        "stencil": jnp.array([-3, -1, 0, 4]),
        "order": 3,
    },
    "3f": {
        "stencil": jnp.array([-3, 0, 1, 4]),
        "order": 3,
    },
    "4a": {
        "stencil": jnp.array([0, 1, 2, 3, 4]),
        "order": 4,
    },
    "4b": {
        "stencil": jnp.array([-1, 0, 1, 2, 3]),
        "order": 4,
    },
    "4d": {
        "stencil": jnp.array([-2, 0, 1, 3, 4]),
        "order": 4,
    },
    "4e": {
        "stencil": jnp.array([-2, -1, 0, 2, 4]),
        "order": 4,
    },
    "5a": {
        "stencil": jnp.array([0, 1, 2, 3, 4, 5]),
        "order": 5,
    },
    "5b": {
        "stencil": jnp.array([-2, -1, 0, 1, 2, 3]),
        "order": 5,
    },
    "5c": {
        "stencil": jnp.array([-3, -2, -1, 0, 1, 2, 3]),
        "order": 6,
        "hyper_nu": jnp.array(0.04 / 243),
        "hyper_order": 6,
        "hyper_accuracy": 2,
        "scale_hyper_nu": 5,
    },
    "5d": {
        "stencil": jnp.array([-3, -2, 0, 1, 3, 4]),
        "order": 5,
    },
    "5e": {
        "stencil": jnp.array([-3, -2, 0, 1, 2, 4]),
        "order": 5,
    },
    "6a": {
        "stencil": jnp.array([0, 1, 2, 3, 4, 5, 6]),
        "order": 6,
    },
    "6b": {
        "stencil": jnp.array([-4, -3, -1, 0, 3, 4, 5]),
        "order": 6,
    },
    "6c": {
        "stencil": jnp.array([-4, -2, -1, 0, 2, 4, 5]),
        "order": 6,
    },
    "6z": {
        "stencil": jnp.array([-3, -2, -1, 0, 1, 2, 3]),
        "order": 6,
    },
    "7a": {
        "stencil": jnp.array([-4, -3, -2, 0, 1, 2, 3, 4]),
        "order": 7,
    },
    "7b": {
        "stencil": jnp.array([-4, -3, -2, 0, 1, 2, 4, 5]),
        "order": 7,
    },
}


@functools.partial(jax.jit, static_argnames=("p", "bc", "axis"))
def fd2(f, p, h=1, bc="periodic", axis=0):
    """Centered finite differences for second derivatives

    Parameters
    ----------
    f : jax.Array
        Function to differentiate at equally spaced points.
    p : str
        Order of accuracy
    h : float
        Grid spacing.
    bc : {"periodic", "symmetric"}
        Type of boundary conditions.
    axis : int
        Axis along which to differentiate f

    Returns
    -------
    df : jax.Array
        Centered finite difference approximation to the derivative of f.

    """
    axis = operator.index(axis)
    f = jnp.moveaxis(f, axis, -1)
    stencil = fd_coeffs[2][p] / h**2
    df = _fdctr(f, stencil, bc)
    return jnp.moveaxis(df, -1, axis)


@functools.partial(jnp.vectorize, signature="(n)->(n)", excluded=[1, 2])
def _fdctr(f, dx, bc="periodic"):
    assert bc in {"periodic", "symmetric"}
    m = len(dx)
    f1 = f2 = jnp.zeros(m)
    if bc == "symmetric":
        f1 = f[:m][::-1]
        f2 = f[-m:][::-1]
    elif bc == "periodic":
        f1 = f[-m:]
        f2 = f[:m]
    fpad = jnp.concatenate([f1, f, f2])
    df = jnp.convolve(fpad, dx[::-1], "valid")
    offset = (df.size - f.size) // 2
    return df[offset : f.size + offset]


@functools.partial(jax.jit, static_argnames=("p", "bc", "axis"))
def fdbwd(f, p, h=1, bc="periodic", axis=0):
    """Backward finite differences for first derivatives.

    Parameters
    ----------
    f : jax.Array
        Function to differentiate at equally spaced points.
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    h : float
        Grid spacing.
    bc : {"periodic", "symmetric"}
        Type of boundary conditions.
    axis : int
        Axis along which to differentiate f

    Returns
    -------
    df : jax.Array
        Backward finite difference approximation to the derivative of f.

    """
    axis = operator.index(axis)
    f = jnp.moveaxis(f, axis, -1)
    stencil = -fd_coeffs[1][p][::-1] / h
    df = _fdctr(f, stencil, bc)
    return jnp.moveaxis(df, -1, axis)


@functools.partial(jax.jit, static_argnames=("p", "bc", "axis"))
def fdfwd(f, p, h=1, bc="periodic", axis=0):
    """Forward finite differences for first derivatives.

    Parameters
    ----------
    f : jax.Array
        Function to differentiate at equally spaced points.
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    h : float
        Grid spacing.
    bc : {"periodic", "symmetric"}
        Type of boundary conditions.
    axis : int
        Axis along which to differentiate f

    Returns
    -------
    df : jax.Array
        Forward finite difference approximation to the derivative of f.

    """
    axis = operator.index(axis)
    f = jnp.moveaxis(f, axis, -1)
    stencil = fd_coeffs[1][p] / h
    df = _fdctr(f, stencil, bc)
    return jnp.moveaxis(df, -1, axis)


@functools.partial(jax.jit, static_argnames=["order"])
def build_lorentz_matrix(a: jax.Array, order: int = 2) -> jax.Array:
    """Finite difference Lorentz scattering operator.

    1/sin(a) ∂ₐ [sin(a) ∂ₐ]

    Assumes symmetric boundary conditions on [0,π]:
    f(-a) = f(a),  f(π-a) = f(π+a)

    Parameters
    ----------
    a : jax.Array
        Coordinates of pitch angle grid. May be non-uniform,
        but should NOT include endpoints at [0,π].
    order : int
        Order of accuracy.

    Returns
    -------
    L : jax.Array
        Finite difference Lorentz operator.
    """
    assert order % 2 == 0
    n = a.shape[0]
    N = order + 1

    # We pad the grid by N points on both sides to guarantee the stencil
    # always has enough room to center itself without overflowing.
    P = N

    # left ghost points (mirrored across 0)
    a_left = -a[:P][::-1]
    idx_left = jnp.arange(P)[::-1]
    # right ghost points (mirrored across π)
    a_right = 2 * jnp.pi - a[-P:][::-1]
    idx_right = jnp.arange(n - P, n)[::-1]

    # augment the grid and create a mapping back to interior indices
    a_aug = jnp.concatenate([a_left, a, a_right])
    map_aug = jnp.concatenate([idx_left, jnp.arange(n), idx_right])

    indices = jnp.arange(n)

    def get_row_weights(i):
        # Center the stencil around the target point in the augmented grid.
        # Since we padded by P, the target interior point x[i] is at index i + P.
        start = i + P - N // 2

        # Dynamically slice the stencil from the augmented grid
        a_stencil = jax.lax.dynamic_slice(a_aug, (start,), (N,))
        mapped_idx = jax.lax.dynamic_slice(map_aug, (start,), (N,))

        da = a_stencil - a[i]

        # Vandermonde solve (same stabilizing logic as before)
        h_scale = jnp.max(jnp.abs(da))
        h_scale = jnp.where(h_scale == 0, 1.0, h_scale)
        da_scaled = da / h_scale

        m = jnp.arange(N)[:, None]
        V = da_scaled[None, :] ** m

        b1_scaled = jnp.zeros(N).at[1].set(1.0)
        w1 = jnp.linalg.solve(V, b1_scaled) / h_scale

        b2_scaled = jnp.zeros(N).at[2].set(2.0)
        w2 = jnp.linalg.solve(V, b2_scaled) / (h_scale**2)

        cot_x = jnp.cos(a[i]) / jnp.sin(a[i])
        w_op = w2 + cot_x * w1

        return mapped_idx, w_op

    mapped_idx, w_op = jax.vmap(get_row_weights)(indices)

    # Assemble the matrix.
    D = jnp.zeros((n, n))
    row_idx = jnp.arange(n)[:, None]

    # important: use .add() instead of .set().
    # If a stencil crosses the boundary, it will contain both a_k and its
    # ghost equivalent. Both share the same 'mapped_idx'. .add() ensures
    # their weights are summed together, perfectly enforcing the symmetry.
    D = D.at[row_idx, mapped_idx].add(w_op)

    return D


@functools.partial(
    jax.jit,
    static_argnames=["direction", "order", "bc_type", "hyper_order", "hyper_accuracy"],
)
def build_advection_matrix(
    x,
    stencil=(-1, 0, 1),
    direction="fwd",
    bc_type="periodic",
    domain=None,
    order=2,
    penalty_ratio=10.0,
    hyper_nu=0.0,
    hyper_order=None,
    hyper_accuracy=None,
    scale_hyper_nu="auto",
):
    """Constructs a global finite difference matrix for advection.

    Contains explicit hyperdiffusion, supporting arbitrary stencils, grid scaling,
    and independent hyperdiffusion accuracy.

    Parameters
    ----------
    x : array_like
        A 1D array of the spatial coordinates of the grid points.
    stencil : array_like, default=(-1, 0, 1)
        A 1D array or tuple of relative integer indices defining the footprint.
    direction : {"fwd", "bwd"}
        Direction of advective flow, for forward/backward differences. "bwd" reverses
        the stencil.
    bc_type : str, default='periodic'
        Boundary condition type: 'periodic' (wraps domain) or 'symmetric' (mirror
        reflection).
    domain : tuple of float, optional
        Where the boundary conditions should be applied. Defaults to x[0] and x[-1]
    order : int, default=2
        The formal order of accuracy for the advection (first derivative) approximation.
    penalty_ratio : float, default=10.0
        The weighting ratio used to enforce diagonal dominance for underdetermined
        stencils.
    hyper_nu : float, default=0.0
        The artificial viscosity coefficient (acts as a grid-independent dampener).
    hyper_order : int, default=None
        The even derivative used for hyperdiffusion (defaults to order+2 or order+1).
    hyper_accuracy : int, default=None
        The formal order of accuracy for the hyperdiffusion operator. Defaults to
        matching the advection 'order'.
    scale_hyper_nu : int, default hyper_order
        hyper_nu is scaled by h**scale_hyper_nu to make hyperdiffusion grid independent.

    Returns
    -------
    jax.numpy.ndarray
        A dense 2D JAX array of shape (N, N) representing the fully constructed
        operator.
    """
    x = jnp.asarray(x)
    stencil = jnp.asarray(stencil, dtype=jnp.int32)
    assert direction in {"fwd", "bwd"}
    if direction == "bwd":
        stencil = -stencil
        hyper_nu *= -1
    stencil = jnp.sort(stencil)

    N = x.shape[0]
    M = stencil.shape[0]

    if domain is None:
        domain = (x[0], x[-1])
    period = domain[1] - domain[0]

    if hyper_order == "auto":
        hyper_order = order + 2 if order % 2 == 0 else order + 1
    if hyper_accuracy is None:
        hyper_accuracy = order
    if scale_hyper_nu == "auto":
        scale_hyper_nu = hyper_order
    if scale_hyper_nu is None:
        scale_hyper_nu = 0
    assert isinstance(scale_hyper_nu, (jax.Array, float, int))

    # The desired Taylor polynomial degree to achieve the requested accuracy
    desired_hyper_req = (
        (hyper_order + hyper_accuracy - 1) if hyper_order is not None else 0
    )

    # Cap the constraints by the available degrees of freedom (M).
    # For symmetric stencils, unconstrained odd derivatives will naturally cancel.
    req_hyper_order = min(desired_hyper_req, M - 1)

    if M < order + 1:
        raise ValueError(
            f"Stencil size M={M} is too small to support advection order={order}."
        )
    if hyper_order is not None and M < req_hyper_order + 1:
        raise ValueError(
            f"Stencil size M={M} is too small to support hyper_order={hyper_order} "
            f"with hyper_accuracy={hyper_accuracy}. You need at least "
            f"{req_hyper_order + 1} points."
        )

    # Maximum required derivative to generate Taylor factorial constraints
    K_max = max(order + 1, (req_hyper_order + 1) if hyper_order is not None else 0)
    facts = jnp.array([math.factorial(k) for k in range(K_max)])

    # Setup Penalty matrix Pi
    Pi = jnp.full(M, penalty_ratio)
    Pi = jnp.where(stencil == 0, 1.0, Pi)

    def solve_weights(dx, target_deriv, req_order):
        """Generates scale-invariant finite difference weights via lstsq."""
        # Scale dx to avoid Vandermonde ill-conditioning at high powers
        h = jnp.max(jnp.abs(dx))
        h = jnp.where(h == 0.0, 1.0, h)
        dx_scaled = dx / h

        K_local = req_order + 1
        powers = jnp.arange(K_local)[:, None]

        # Build Vandermonde Matrix: V[k, j] = (dx_j)^k / k!
        V = (dx_scaled[None, :] ** powers) / facts[:K_local, None]

        # We want to solve C * y = b, where C = V * Pi^{-1/2} and y = Pi^{1/2} * w
        Pi_inv_sqrt = 1.0 / jnp.sqrt(Pi)
        C = V * Pi_inv_sqrt[None, :]

        b = jnp.zeros(K_local).at[target_deriv].set(1.0)

        # jnp.linalg.lstsq finds the minimum L2-norm solution for underdetermined
        # systems, y will have shape (M,)
        y, _, _, _ = jnp.linalg.lstsq(C, b, rcond=None)

        w_scaled = y * Pi_inv_sqrt
        w = w_scaled / (h**target_deriv)
        return w, h

    def get_row(i):
        raw_idx = i + stencil

        if bc_type == "periodic":
            idx = raw_idx % N
            dx = x[idx] - x[i]
            dx = dx - period * jnp.round(dx / period)
        elif bc_type == "symmetric":
            idx = jnp.where(
                raw_idx < 0,
                -raw_idx - 1,
                jnp.where(raw_idx >= N, 2 * N - 1 - raw_idx, raw_idx),
            )
            x_ghost = jnp.where(
                raw_idx < 0,
                2 * domain[0] - x[idx],
                jnp.where(raw_idx >= N, 2 * domain[1] - x[idx], x[idx]),
            )
            dx = x_ghost - x[i]
        else:
            idx = jnp.clip(raw_idx, 0, N - 1)
            dx = x[idx] - x[i]

        # advection
        w_total, _ = solve_weights(dx, target_deriv=1, req_order=order)

        # hyperdiffusion
        if hyper_order is not None:
            w_hyp, max_h = solve_weights(
                dx, target_deriv=hyper_order, req_order=req_hyper_order
            )
            sign = (-1.0) ** (hyper_order // 2 + 1)
            grid_scale = max_h**scale_hyper_nu
            w_total = w_total + hyper_nu * sign * w_hyp * grid_scale

        return idx, w_total

    indices, weights = jax.vmap(get_row)(jnp.arange(N))

    D = jnp.zeros((N, N))
    row_indices = jnp.arange(N)[:, None]
    # boundary conditions mean indices may contain duplicates due to reflection,
    # so use .add rather than .set to combine weights correctly.
    D = D.at[row_indices, indices].add(weights)

    return D
