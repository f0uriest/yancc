"""Smoothing operators for multigrid."""

import warnings

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import lineax as lx

from .field import Field
from .finite_diff import fd_coeffs
from .linalg import lu_factor_banded_periodic, lu_solve_banded_periodic
from .trajectories import MDKE, _parse_axorder_shape
from .velocity_grids import UniformPitchAngleGrid


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

        mats = MDKE(
            field, pitchgrid, E_psi, nu, p1, p2, axorder, gauge
        ).block_diagonal()

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
@lx.is_diagonal.register(MDKEJacobiSmoother)
@lx.is_tridiagonal.register(MDKEJacobiSmoother)
def _(operator):
    return False


def optimal_smoothing_parameter(p1, p2, nuhat, axorder):
    """Approximate best relaxation parameter for block jacobi smoother for MDKE."""
    method = p1  # smoothing seems to be the same for any p2 so ignore that
    nus = jnp.array([-6, -4, -2, 0])
    nu = jnp.log10(nuhat)
    if method not in OPTIMAL_SMOOTHING_COEFFS:
        warnings.warn(
            f"No optimal smoothing parameter for stencil={method}, using "
            "conservative default of w=0.1"
        )
        return jnp.array(0.1)  # conservative guess
    if axorder[-1] not in OPTIMAL_SMOOTHING_COEFFS[method]:
        warnings.warn(
            f"No optimal smoothing parameter for axorder={axorder}, using "
            "conservative default of w=0.1"
        )
        return jnp.array(0.1)  # conservative guess
    c = OPTIMAL_SMOOTHING_COEFFS[method][axorder[-1]]
    w = interpax.interp1d(nu, nus, c, method="linear", extrap=True)
    return jnp.clip(w, 0.1, 1.0)


OPTIMAL_SMOOTHING_COEFFS = {
    "1a": {
        "z": jnp.array([0.68421053, 0.68421053, 0.68421053, 0.63157895]),
        "t": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.63157895]),
        "a": jnp.array([0.52631579, 0.57894737, 0.63157895, 0.89473684]),
    },
    "1b": {
        "z": jnp.array([0.36842105, 0.36842105, 0.47368421, 0.57894737]),
        "t": jnp.array([0.21052632, 0.21052632, 0.36842105, 0.57894737]),
        "a": jnp.array([0.21052632, 0.21052632, 0.36842105, 0.84210526]),
    },
    "2a": {
        "z": jnp.array([0.52631579, 0.52631579, 0.52631579, 0.57894737]),
        "t": jnp.array([0.42105263, 0.42105263, 0.47368421, 0.57894737]),
        "a": jnp.array([0.42105263, 0.42105263, 0.47368421, 0.84210526]),
    },
    "2b": {
        "z": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.57894737]),
        "t": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.57894737]),
        "a": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.84210526]),
    },
    "2c": {
        "z": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.57894737]),
        "t": jnp.array([0.21052632, 0.21052632, 0.31578947, 0.57894737]),
        "a": jnp.array([0.15789474, 0.21052632, 0.31578947, 0.78947368]),
    },
    "2d": {
        "z": jnp.array([0.63157895, 0.63157895, 0.68421053, 0.57894737]),
        "t": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.57894737]),
        "a": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.89473684]),
    },
    "3a": {
        "z": jnp.array([0.36842105, 0.36842105, 0.42105263, 0.52631579]),
        "t": jnp.array([0.26315789, 0.26315789, 0.31578947, 0.52631579]),
        "a": jnp.array([0.26315789, 0.26315789, 0.31578947, 0.73684211]),
    },
    "3b": {
        "z": jnp.array([0.36842105, 0.36842105, 0.47368421, 0.57894737]),
        "t": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.57894737]),
        "a": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.78947368]),
    },
    "3c": {
        "z": jnp.array([0.52631579, 0.52631579, 0.57894737, 0.57894737]),
        "t": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.57894737]),
        "a": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.89473684]),
    },
    "3d": {
        "z": jnp.array([0.57894737, 0.57894737, 0.63157895, 0.57894737]),
        "t": jnp.array([0.47368421, 0.47368421, 0.52631579, 0.57894737]),
        "a": jnp.array([0.47368421, 0.47368421, 0.57894737, 0.89473684]),
    },
    "4a": {
        "z": jnp.array([0.21052632, 0.21052632, 0.26315789, 0.47368421]),
        "t": jnp.array([0.15789474, 0.15789474, 0.21052632, 0.47368421]),
        "a": jnp.array([0.15789474, 0.15789474, 0.21052632, 0.57894737]),
    },
    "4b": {
        "z": jnp.array([0.36842105, 0.36842105, 0.47368421, 0.52631579]),
        "t": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.52631579]),
        "a": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.78947368]),
    },
    "4d": {
        "z": jnp.array([0.57894737, 0.57894737, 0.63157895, 0.57894737]),
        "t": jnp.array([0.47368421, 0.47368421, 0.52631579, 0.57894737]),
        "a": jnp.array([0.47368421, 0.47368421, 0.52631579, 0.89473684]),
    },
    "5a": {
        "z": jnp.array([0.10526316, 0.10526316, 0.15789474, 0.36842105]),
        "t": jnp.array([0.10526316, 0.10526316, 0.10526316, 0.36842105]),
        "a": jnp.array([0.10526316, 0.10526316, 0.10526316, 0.42105263]),
    },
    "5b": {
        "z": jnp.array([0.21052632, 0.21052632, 0.36842105, 0.57894737]),
        "t": jnp.array([0.10526316, 0.10526316, 0.26315789, 0.57894737]),
        "a": jnp.array([0.10526316, 0.10526316, 0.21052632, 0.68421053]),
    },
    "5c": {
        "z": jnp.array([0.42105263, 0.42105263, 0.47368421, 0.57894737]),
        "t": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.57894737]),
        "a": jnp.array([0.31578947, 0.31578947, 0.42105263, 0.84210526]),
    },
    "5d": {
        "z": jnp.array([0.52631579, 0.52631579, 0.63157895, 0.57894737]),
        "t": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.57894737]),
        "a": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.89473684]),
    },
    "6a": {
        "z": jnp.array([0.05263158, 0.05263158, 0.05263158, 0.26315789]),
        "t": jnp.array([0.05263158, 0.05263158, 0.05263158, 0.26315789]),
        "a": jnp.array([0.05263158, 0.05263158, 0.05263158, 0.26315789]),
    },
}
