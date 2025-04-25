"""Smoothing operators for multigrid."""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .field import Field
from .finite_diff import fd_coeffs
from .linalg import tridiag_solve_dense
from .trajectories import _parse_axorder_shape, dfdpitch, dfdtheta, dfdxi, dfdzeta
from .velocity_grids import PitchAngleGrid


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


class MDKEJacobiSmoother(lx.AbstractLinearOperator):
    """Block diagonal smoother for MDKE.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    nu : float
        Normalized collisionality, nu/v
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    flip : bool
        If True, assume f is ordered backwards in each coordinate.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    """

    field: Field
    pitchgrid: PitchAngleGrid
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    flip: bool
    axorder: str = eqx.field(static=True)
    mats: jax.Array

    def __init__(
        self,
        field,
        pitchgrid,
        E_psi,
        nu,
        p1="1a",
        p2=2,
        axorder="atz",
        flip=False,
        gauge=True,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.flip = jnp.array(flip)

        mats = get_block_diag(
            field,
            pitchgrid,
            E_psi,
            nu,
            axorder=axorder,
            p1=p1,
            p2=p2,
            flip=flip,
            gauge=gauge,
        )
        if fd_coeffs[1][self.p1].size <= 3 and fd_coeffs[2][self.p2].size <= 3:
            self.mats = mats
        else:
            self.mats = jnp.linalg.inv(mats)

    @eqx.filter_jit
    def mv(self, x):
        """Matrix vector product."""
        permute = lambda f: permute_f(
            f, self.field, self.pitchgrid, self.axorder, self.flip
        )
        x = jax.linear_transpose(permute, x)(x)[0]
        size, N, M = self.mats.shape
        x = x.reshape(size, N)
        if fd_coeffs[1][self.p1].size <= 3 and fd_coeffs[2][self.p2].size <= 3:
            b = tridiag_solve_dense(self.mats, x)
        else:
            b = jnp.einsum("ijk,ik -> ij", self.mats, x[:, :])
        return permute(b.flatten())

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(MDKEJacobiSmoother)
def _(operator):
    return False


@lx.is_diagonal.register(MDKEJacobiSmoother)
def _(operator):
    return False
