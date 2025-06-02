"""Smoothing operators for multigrid."""

import functools

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import lineax as lx

from .field import Field
from .finite_diff import fd_coeffs
from .linalg import lu_factor_banded_periodic, lu_solve_banded_periodic
from .trajectories import _parse_axorder_shape, dfdpitch, dfdtheta, dfdxi, dfdzeta
from .velocity_grids import UniformPitchAngleGrid


@functools.partial(jax.jit, static_argnames=["axorder", "p1", "p2"])
def get_block_diag(
    field, pitchgrid, E_psi, nu, axorder="atz", p1="1a", p2=2, gauge=False
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
                gauge=gauge,
            )
        )(x.reshape((-1, field.nzeta)).T).T.reshape((-1, field.nzeta, field.nzeta))
        return f1 + f2 + f3 + f4


def permute_f(f, field, pitchgrid, axorder):
    """Rearrange elements of f to a given grid ordering."""
    shape, caxorder = _parse_axorder_shape(
        field.ntheta, field.nzeta, pitchgrid.nxi, axorder
    )
    f = f.reshape(shape)
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
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    smooth_solver : {"banded", "dense"}
        Solver to use for inverting the smoother. "banded" is significantly faster in
        most cases but may be numerically unstable in some edge cases. "dense" is
        slower but more robust.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    bandwidth: int = eqx.field(static=True)
    smooth_solver: str = eqx.field(static=True)
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
        gauge=True,
        smooth_solver="banded",
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        assert smooth_solver in {"banded", "dense"}
        self.smooth_solver = smooth_solver
        self.bandwidth = max(
            fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2
        )

        mats = get_block_diag(
            field,
            pitchgrid,
            E_psi,
            nu,
            axorder=axorder,
            p1=p1,
            p2=p2,
            gauge=gauge,
        )

        if self.smooth_solver == "banded":
            self.mats = lu_factor_banded_periodic(self.bandwidth, mats)
        else:
            self.mats = jnp.linalg.inv(mats)

    @eqx.filter_jit
    def mv(self, x):
        """Matrix vector product."""
        permute = lambda f: permute_f(f, self.field, self.pitchgrid, self.axorder)
        x = jax.linear_transpose(permute, x)(x)[0]

        if self.smooth_solver == "banded":
            size, N, M = self.mats[0].shape
            x = x.reshape(size, M)
            b = lu_solve_banded_periodic(self.bandwidth, self.mats, x)
        else:
            size, N, M = self.mats.shape
            x = x.reshape(size, M)
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


def optimal_smoothing_parameter(p1, p2, nuhat, axorder):
    """Approximate best relaxation parameter for block jacobi smoother for MDKE."""
    method = p1  # smoothing seems to be the same for any p2 so ignore that
    nus = jnp.array([-6, -4, -2, 0])
    nu = jnp.log10(nuhat)
    if method not in OPTIMAL_SMOOTHING_COEFFS:
        return jnp.array(0.1)  # conservative guess
    if axorder not in OPTIMAL_SMOOTHING_COEFFS[method]:
        return jnp.array(0.1)  # conservative guess
    c = OPTIMAL_SMOOTHING_COEFFS[method][axorder]
    w = interpax.interp1d(nu, nus, c, method="linear", extrap=True)
    return jnp.clip(w, 0.1, 1.0)


OPTIMAL_SMOOTHING_COEFFS = {
    "1a": {
        "atz": jnp.array([0.68421053, 0.68421053, 0.68421053, 0.63157895]),
        "zat": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.63157895]),
        "tza": jnp.array([0.52631579, 0.57894737, 0.63157895, 0.89473684]),
    },
    "1b": {
        "atz": jnp.array([0.36842105, 0.36842105, 0.47368421, 0.57894737]),
        "zat": jnp.array([0.21052632, 0.21052632, 0.36842105, 0.57894737]),
        "tza": jnp.array([0.21052632, 0.21052632, 0.36842105, 0.84210526]),
    },
    "2a": {
        "atz": jnp.array([0.52631579, 0.52631579, 0.52631579, 0.57894737]),
        "zat": jnp.array([0.42105263, 0.42105263, 0.47368421, 0.57894737]),
        "tza": jnp.array([0.42105263, 0.42105263, 0.47368421, 0.84210526]),
    },
    "2b": {
        "atz": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.57894737]),
        "zat": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.57894737]),
        "tza": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.84210526]),
    },
    "2c": {
        "atz": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.57894737]),
        "zat": jnp.array([0.21052632, 0.21052632, 0.31578947, 0.57894737]),
        "tza": jnp.array([0.15789474, 0.21052632, 0.31578947, 0.78947368]),
    },
    "2d": {
        "atz": jnp.array([0.63157895, 0.63157895, 0.68421053, 0.57894737]),
        "zat": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.57894737]),
        "tza": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.89473684]),
    },
    "3a": {
        "atz": jnp.array([0.36842105, 0.36842105, 0.42105263, 0.52631579]),
        "zat": jnp.array([0.26315789, 0.26315789, 0.31578947, 0.52631579]),
        "tza": jnp.array([0.26315789, 0.26315789, 0.31578947, 0.73684211]),
    },
    "3b": {
        "atz": jnp.array([0.36842105, 0.36842105, 0.47368421, 0.57894737]),
        "zat": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.57894737]),
        "tza": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.78947368]),
    },
    "3c": {
        "atz": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.57894737]),
        "zat": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.57894737]),
        "tza": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.89473684]),
    },
    "3d": {
        "atz": jnp.array([0.57894737, 0.57894737, 0.63157895, 0.57894737]),
        "zat": jnp.array([0.47368421, 0.47368421, 0.52631579, 0.57894737]),
        "tza": jnp.array([0.47368421, 0.47368421, 0.57894737, 0.89473684]),
    },
    "3e": {
        "atz": jnp.array([0.68421053, 0.68421053, 0.68421053, 0.63157895]),
        "zat": jnp.array([0.63157895, 0.63157895, 0.63157895, 0.57894737]),
        "tza": jnp.array([0.63157895, 0.63157895, 0.63157895, 0.78947368]),
    },
    "4a": {
        "atz": jnp.array([0.21052632, 0.21052632, 0.26315789, 0.47368421]),
        "zat": jnp.array([0.15789474, 0.15789474, 0.21052632, 0.47368421]),
        "tza": jnp.array([0.15789474, 0.15789474, 0.21052632, 0.57894737]),
    },
    "4b": {
        "atz": jnp.array([0.36842105, 0.36842105, 0.47368421, 0.52631579]),
        "zat": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.52631579]),
        "tza": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.78947368]),
    },
    "4d": {
        "atz": jnp.array([0.57894737, 0.57894737, 0.63157895, 0.57894737]),
        "zat": jnp.array([0.47368421, 0.47368421, 0.52631579, 0.57894737]),
        "tza": jnp.array([0.47368421, 0.47368421, 0.52631579, 0.89473684]),
    },
    "5a": {
        "atz": jnp.array([0.10526316, 0.10526316, 0.15789474, 0.36842105]),
        "zat": jnp.array([0.10526316, 0.10526316, 0.10526316, 0.36842105]),
        "tza": jnp.array([0.10526316, 0.10526316, 0.10526316, 0.42105263]),
    },
    "5b": {
        "atz": jnp.array([0.21052632, 0.21052632, 0.36842105, 0.57894737]),
        "zat": jnp.array([0.10526316, 0.10526316, 0.26315789, 0.57894737]),
        "tza": jnp.array([0.10526316, 0.10526316, 0.21052632, 0.68421053]),
    },
    "5c": {
        "atz": jnp.array([0.42105263, 0.42105263, 0.47368421, 0.57894737]),
        "zat": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.57894737]),
        "tza": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.84210526]),
    },
    "5d": {
        "atz": jnp.array([0.52631579, 0.52631579, 0.63157895, 0.57894737]),
        "zat": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.57894737]),
        "tza": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.89473684]),
    },
    "6a": {
        "atz": jnp.array([0.05263158, 0.05263158, 0.05263158, 0.26315789]),
        "zat": jnp.array([0.05263158, 0.05263158, 0.05263158, 0.26315789]),
        "tza": jnp.array([0.05263158, 0.05263158, 0.05263158, 0.26315789]),
    },
}
