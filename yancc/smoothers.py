"""Smoothing operators for multigrid."""

import functools

import jax
import jax.numpy as jnp

from .trajectories import _parse_axorder_shape, dfdpitch, dfdtheta, dfdxi, dfdzeta


@functools.partial(jax.jit, static_argnames=["axorder", "p1", "p2"])
def get_block_diag(
    field, pitchgrid, E_psi, nu, axorder="atz", p1="1a", p2=2, flip=False, gauge=False
):
    """Extract the block diagonal part of the MDKE operator for a given ordering."""
    assert axorder[-1] in "atz"

    if axorder[-1] == "a":
        x = jnp.tile(jnp.eye(pitchgrid.nxi)[None], (field.ntheta * field.nzeta, 1, 1))
        f = jnp.ones(field.ntheta * field.nzeta * pitchgrid.nxi)
        f1 = jax.vmap(
            lambda x: dfdxi(
                x,
                field,
                pitchgrid,
                nu,
                axorder=axorder,
                p=p1,
                flip=flip,
                gauge=gauge,
            )
        )(x.reshape((-1, pitchgrid.nxi)).T).T.reshape(
            (-1, pitchgrid.nxi, pitchgrid.nxi)
        )
        f2 = jax.vmap(
            lambda x: dfdpitch(
                x,
                field,
                pitchgrid,
                nu,
                axorder=axorder,
                p=p2,
                flip=flip,
                gauge=gauge,
            )
        )(x.reshape((-1, pitchgrid.nxi)).T).T.reshape(
            (-1, pitchgrid.nxi, pitchgrid.nxi)
        )
        f3 = dfdtheta(
            f,
            field,
            pitchgrid,
            E_psi,
            axorder=axorder,
            p=p1,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f3 = jax.vmap(jnp.diag)(f3.reshape((-1, pitchgrid.nxi)))
        f4 = dfdzeta(
            f,
            field,
            pitchgrid,
            E_psi,
            axorder=axorder,
            p=p1,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f4 = jax.vmap(jnp.diag)(f4.reshape((-1, pitchgrid.nxi)))
        return f1 + f2 + f3 + f4

    elif axorder[-1] == "t":
        x = jnp.tile(jnp.eye(field.ntheta)[None], (pitchgrid.nxi * field.nzeta, 1, 1))
        f = jnp.ones(field.ntheta * field.nzeta * pitchgrid.nxi)
        f1 = dfdxi(
            f,
            field,
            pitchgrid,
            nu,
            axorder=axorder,
            p=p1,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f1 = jax.vmap(jnp.diag)(f1.reshape((-1, field.ntheta)))
        f2 = dfdpitch(
            f,
            field,
            pitchgrid,
            nu,
            axorder=axorder,
            p=p2,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f2 = jax.vmap(jnp.diag)(f2.reshape((-1, field.ntheta)))
        f3 = jax.vmap(
            lambda x: dfdtheta(
                x,
                field,
                pitchgrid,
                E_psi,
                axorder=axorder,
                p=p1,
                flip=flip,
                gauge=gauge,
            )
        )(x.reshape((-1, field.ntheta)).T).T.reshape((-1, field.ntheta, field.ntheta))
        f4 = dfdzeta(
            f,
            field,
            pitchgrid,
            E_psi,
            axorder=axorder,
            p=p1,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f4 = jax.vmap(jnp.diag)(f4.reshape((-1, field.ntheta)))
        return f1 + f2 + f3 + f4

    elif axorder[-1] == "z":
        x = jnp.tile(jnp.eye(field.nzeta)[None], (pitchgrid.nxi * field.ntheta, 1, 1))
        f = jnp.ones(field.ntheta * field.nzeta * pitchgrid.nxi)
        f1 = dfdxi(
            f,
            field,
            pitchgrid,
            nu,
            axorder=axorder,
            p=p1,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f1 = jax.vmap(jnp.diag)(f1.reshape((-1, field.nzeta)))
        f2 = dfdpitch(
            f,
            field,
            pitchgrid,
            nu,
            axorder=axorder,
            p=p2,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f2 = jax.vmap(jnp.diag)(f2.reshape((-1, field.nzeta)))
        f3 = dfdtheta(
            f,
            field,
            pitchgrid,
            E_psi,
            axorder=axorder,
            p=p1,
            diag=True,
            flip=flip,
            gauge=gauge,
        )
        f3 = jax.vmap(jnp.diag)(f3.reshape((-1, field.nzeta)))
        f4 = jax.vmap(
            lambda x: dfdzeta(
                x,
                field,
                pitchgrid,
                E_psi,
                axorder=axorder,
                p=p1,
                flip=flip,
                gauge=gauge,
            )
        )(x.reshape((-1, field.nzeta)).T).T.reshape((-1, field.nzeta, field.nzeta))
        return f1 + f2 + f3 + f4


def permute_f(f, field, pitchgrid, axorder, flip):
    """Rearrange elements of f to a given grid ordering."""
    shape, caxorder = _parse_axorder_shape(
        field.ntheta, field.nzeta, pitchgrid.nxi, axorder
    )
    f = f.reshape(shape)
    f = jax.lax.cond(flip, lambda: f[..., ::-1], lambda: f)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2))
    return f.flatten()
