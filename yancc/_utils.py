"""Simple utility functions."""

import jax
import jax.numpy as jnp
import numpy as np

###############################
### reshape / permutation utils
###############################


def _parse_axorder_shape_3d(
    nt: int, nz: int, na: int, axorder: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = np.empty(3, dtype=int)
    shape[axorder.index("a")] = na
    shape[axorder.index("t")] = nt
    shape[axorder.index("z")] = nz
    caxorder = (axorder.index("a"), axorder.index("t"), axorder.index("z"))
    return tuple(shape), caxorder


def _parse_axorder_shape_4d(
    nt: int, nz: int, na: int, nx: int, ns: int, axorder: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = np.empty(5, dtype=int)
    shape[axorder.index("a")] = na
    shape[axorder.index("t")] = nt
    shape[axorder.index("z")] = nz
    shape[axorder.index("x")] = nx
    shape[axorder.index("s")] = ns
    caxorder = (
        axorder.index("s"),
        axorder.index("x"),
        axorder.index("a"),
        axorder.index("t"),
        axorder.index("z"),
    )
    return tuple(shape), caxorder


### misc utils


def safediv(a: jax.Array, b: jax.Array, fill=jnp.array(0), threshold=jnp.array(0)):
    """Divide a/b with guards for division by zero.

    Parameters
    ----------
    a, b : ndarray
        Numerator and denominator.
    fill : float, ndarray, optional
        Value to return where b is zero.
    threshold : float >= 0
        How small is b allowed to be.

    """
    mask = jnp.abs(b) <= threshold
    num = jnp.where(mask, fill, a)
    den = jnp.where(mask, 1, b)
    return num / den
