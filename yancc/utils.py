"""Simple utility functions."""

import jax
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


def _refold(a, k):
    N, M, _ = a.shape
    a = a.reshape((N // k, k, M, M))
    # TODO: make this better
    return jax.vmap(lambda x: jax.scipy.linalg.block_diag(*x))(a)
