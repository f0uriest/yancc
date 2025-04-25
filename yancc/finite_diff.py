"""Functions for computing finite difference derivatives."""

import functools
import operator

import jax
import jax.numpy as jnp

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
        "3a": jnp.array([0, 0, 0, -11 / 6, 3, -3 / 2, 1 / 3]),
        "3b": jnp.array([0, -1 / 3, -1 / 2, 1, -1 / 6]),  # CUI
        "3c": jnp.array(
            [1 / 12 - y3c, -2 / 3 + 4 * y3c, -6 * y3c, 2 / 3 + 4 * y3c, -1 / 12 - y3c]
        ),
        "4a": jnp.array([0, 0, 0, 0, -25 / 12, 4, -3, 4 / 3, -1 / 4]),
        "4b": jnp.array([0, 0, -1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12]),
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
        "6a": jnp.array(
            [0, 0, 0, 0, 0, 0, -49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]
        ),
    },
    # these are centered coeffs for 2nd derivatives.
    2: {
        2: jnp.array([1, -2, 1]),
        4: jnp.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
        6: jnp.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]),
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
