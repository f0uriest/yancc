"""Functions for computing finite difference derivatives."""

import functools
import operator

import jax
import jax.numpy as jnp

fwd_coeffs = {
    1: {
        1: jnp.array([-1, 1]),
        2: jnp.array([-3 / 2, 2, -1 / 2]),
        3: jnp.array([-11 / 6, 3, -3 / 2, 1 / 3]),
        4: jnp.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]),
        5: jnp.array([-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5]),
        6: jnp.array([-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]),
    },
    2: {
        1: jnp.array([1, -2, 1]),
        2: jnp.array([2, -5, 4, -1]),
        3: jnp.array([35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12]),
        4: jnp.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6]),
        5: jnp.array(
            [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180]
        ),
        6: jnp.array(
            [
                469 / 90,
                -223 / 10,
                879 / 20,
                -949 / 18,
                41,
                -201 / 10,
                1019 / 180,
                -7 / 10,
            ]
        ),
    },
}
bwd_coeffs = {
    1: {
        1: -jnp.array([-1, 1])[::-1],
        2: -jnp.array([-3 / 2, 2, -1 / 2])[::-1],
        3: -jnp.array([-11 / 6, 3, -3 / 2, 1 / 3])[::-1],
        4: -jnp.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])[::-1],
        5: -jnp.array([-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5])[::-1],
        6: -jnp.array([-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6])[::-1],
    },
    2: {
        1: jnp.array([1, -2, 1])[::-1],
        2: jnp.array([2, -5, 4, -1])[::-1],
        3: jnp.array([35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12])[::-1],
        4: jnp.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6])[::-1],
        5: jnp.array(
            [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180]
        )[::-1],
        6: jnp.array(
            [
                469 / 90,
                -223 / 10,
                879 / 20,
                -949 / 18,
                41,
                -201 / 10,
                1019 / 180,
                -7 / 10,
            ]
        )[::-1],
    },
}

ctr_coeffs = {
    1: {
        2: jnp.array([-1 / 2, 0, 1 / 2]),
        4: jnp.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
        6: jnp.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
    },
    2: {
        2: jnp.array([1, -2, 1]),
        4: jnp.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
        6: jnp.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]),
    },
}


@functools.partial(jax.jit, static_argnames=("p", "d", "bc", "axis"))
def fdbwd(f, p, d=1, h=1, bc="periodic", axis=0):
    """Backward finite differences.

    Parameters
    ----------
    f : jax.Array
        Function to differentiate at equally spaced points.
    p : {1,2,3,4,5,6}
        Order of accuracy.
    d : {1,2}
        Order of derivative.
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
    df = _fdbwd(f, p=p, d=d, h=h, bc=bc)
    return jnp.moveaxis(df, -1, axis)


@functools.partial(jnp.vectorize, signature="(n)->(n)", excluded=["p", "d", "h", "bc"])
def _fdbwd(f, *, p, d=1, h=1, bc="periodic"):
    assert bc in {"periodic", "symmetric"}
    dx = bwd_coeffs[d][p] / h**d
    m = len(dx)
    if bc == "symmetric":
        fp = f[:m][::-1]
    elif bc == "periodic":
        fp = f[-m:]
    fpad = jnp.concatenate([fp, f])
    df = jnp.convolve(fpad, dx[::-1], "valid")
    return df[-len(f) :]


@functools.partial(jax.jit, static_argnames=("p", "d", "bc", "axis"))
def fdfwd(f, p, d=1, h=1, bc="periodic", axis=0):
    """Forward finite differences.

    Parameters
    ----------
    f : jax.Array
        Function to differentiate at equally spaced points.
    p : {1,2,3,4,5,6}
        Order of accuracy.
    d : {1,2}
        Order of derivative.
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
    df = _fdfwd(f, p=p, d=d, h=h, bc=bc)
    return jnp.moveaxis(df, -1, axis)


@functools.partial(jnp.vectorize, signature="(n)->(n)", excluded=["p", "d", "h", "bc"])
def _fdfwd(f, *, p, d=1, h=1, bc="periodic"):
    assert bc in {"periodic", "symmetric"}
    dx = fwd_coeffs[d][p] / h**d
    m = len(dx)
    if bc == "symmetric":
        fp = f[-m:][::-1]
    elif bc == "periodic":
        fp = f[:m]
    fpad = jnp.concatenate([f, fp])
    df = jnp.convolve(fpad, dx[::-1], "valid")
    return df[: len(f)]


@functools.partial(jax.jit, static_argnames=("p", "d", "bc", "axis"))
def fdctr(f, p, d=1, h=1, bc="periodic", axis=0):
    """Centered finite differences.

    Parameters
    ----------
    f : jax.Array
        Function to differentiate at equally spaced points.
    p : {2,4,6}
        Order of accuracy.
    d : {1,2}
        Order of derivative.
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
    df = _fdctr(f, p=p, d=d, h=h, bc=bc)
    return jnp.moveaxis(df, -1, axis)


@functools.partial(jnp.vectorize, signature="(n)->(n)", excluded=["p", "d", "h", "bc"])
def _fdctr(f, *, p, d=1, h=1, bc="periodic"):
    assert bc in {"periodic", "symmetric"}
    dx = ctr_coeffs[d][p] / h**d
    m = len(dx)
    if bc == "symmetric":
        f1 = f[:m][::-1]
        f2 = f[-m:][::-1]
    elif bc == "periodic":
        f1 = f[-m:]
        f2 = f[:m]
    fpad = jnp.concatenate([f1, f, f2])
    df = jnp.convolve(fpad, dx[::-1], "valid")
    return df[m // 2 + 1 : f.size + m // 2 + 1]
