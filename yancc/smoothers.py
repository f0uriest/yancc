"""Smoothing operators for multigrid."""

import warnings
from typing import Optional

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import ArrayLike, Bool, Float

from .field import Field
from .finite_diff import fd_coeffs
from .linalg import lu_factor_banded_periodic, lu_solve_banded_periodic
from .species import LocalMaxwellian, nuD_ab
from .trajectories import DKE, MDKE, _parse_axorder_shape_3d, _parse_axorder_shape_4d
from .velocity_grids import SpeedGrid, UniformPitchAngleGrid


def permute_f_3d(
    f: jax.Array, field: Field, pitchgrid: UniformPitchAngleGrid, axorder: str
) -> jax.Array:
    """Rearrange elements of f to a given grid ordering."""
    shape, caxorder = _parse_axorder_shape_3d(
        field.ntheta, field.nzeta, pitchgrid.nxi, axorder
    )
    f = f.reshape(shape)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2))
    return f.flatten()


def permute_f_4d(
    f: jax.Array,
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: SpeedGrid,
    species: list[LocalMaxwellian],
    axorder: str,
) -> jax.Array:
    """Rearrange elements of f to a given grid ordering."""
    shape, caxorder = _parse_axorder_shape_4d(
        field.ntheta, field.nzeta, pitchgrid.nxi, speedgrid.nx, len(species), axorder
    )
    f = f.reshape(shape)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
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
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    smooth_solver : {"banded", "dense"}
        Solver to use for inverting the smoother. "banded" is significantly faster in
        most cases but may be numerically unstable in some edge cases. "dense" is
        slower but more robust.
    weight : array-like, optional
        Under-relaxation parameter.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    bandwidth: int = eqx.field(static=True)
    smooth_solver: str = eqx.field(static=True)
    weight: jax.Array
    mats: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        E_psi: Float[ArrayLike, ""],
        nu: Float[ArrayLike, ""],
        p1: str = "2d",
        p2: int = 2,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = True,
        smooth_solver: str = "banded",
        weight: Optional[jax.Array] = None,
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
        if weight is None:
            weight = optimal_smoothing_parameter(p1, p2, nu, axorder)
        self.weight = jnp.atleast_1d(jnp.array(weight))

        mats = MDKE(
            field, pitchgrid, E_psi, nu, p1, p2, axorder, gauge
        ).block_diagonal()

        if self.smooth_solver == "banded":
            self.mats = lu_factor_banded_periodic(self.bandwidth, mats)
        else:
            self.mats = jnp.linalg.inv(mats)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        x = vector
        permute = lambda f: permute_f_3d(f, self.field, self.pitchgrid, self.axorder)
        x = jax.linear_transpose(permute, x)(x)[0]

        if self.smooth_solver == "banded":
            size, N, M = self.mats[0].shape
            x = x.reshape(size, M)
            b = lu_solve_banded_periodic(self.bandwidth, self.mats, x)
        else:
            size, N, M = self.mats.shape
            x = x.reshape(size, M)
            b = jnp.einsum("ijk,ik -> ij", self.mats, x[:, :])

        return self.weight * permute(b.flatten())

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


class DKEJacobiSmoother(lx.AbstractLinearOperator):
    """Block diagonal smoother for DKE.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Normalized electric field, E_psi/v
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    smooth_solver : {"banded", "dense"}
        Solver to use for inverting the smoother. "banded" is significantly faster in
        most cases but may be numerically unstable in some edge cases. "dense" is
        slower but more robust.
    weight : array-like, optional
        Under-relaxation parameter.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    bandwidth: int = eqx.field(static=True)
    smooth_solver: str = eqx.field(static=True)
    mats: jax.Array
    weight: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        p1="2d",
        p2=2,
        axorder="sxatz",
        smooth_solver="banded",
        weight: Optional[jax.Array] = None,
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        assert smooth_solver in {"banded", "dense"}
        self.smooth_solver = smooth_solver
        self.bandwidth = max(
            fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2
        )
        if weight is None:
            x = speedgrid.x
            nus = []
            for spa in species:
                nu = 0.0
                for spb in species:
                    nu += nuD_ab(spa, spb, x * spa.v_thermal) / spa.v_thermal
                nus.append(nu)
            nus = jnp.asarray(nus)
            _fun = lambda y: optimal_smoothing_parameter(p1, p2, y, axorder)
            weight = jnp.vectorize(_fun)(nus)[:, :, None, None, None]
            weight = weight * jnp.ones((1, 1, pitchgrid.nxi, field.ntheta, field.nzeta))
        self.weight = jnp.asarray(weight).flatten()

        mats = DKE(
            field, pitchgrid, speedgrid, species, E_psi, p1, p2, axorder
        ).block_diagonal()

        if self.smooth_solver == "banded":
            self.mats = lu_factor_banded_periodic(self.bandwidth, mats)
        else:
            self.mats = jnp.linalg.inv(mats)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        x = vector
        permute = lambda f: permute_f_3d(f, self.field, self.pitchgrid, self.axorder)
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
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(DKEJacobiSmoother)
@lx.is_diagonal.register(DKEJacobiSmoother)
@lx.is_tridiagonal.register(DKEJacobiSmoother)
@lx.is_symmetric.register(MDKEJacobiSmoother)
@lx.is_diagonal.register(MDKEJacobiSmoother)
@lx.is_tridiagonal.register(MDKEJacobiSmoother)
def _(operator):
    return False


def optimal_smoothing_parameter(p1, p2, nuhat, axorder):
    """Approximate best relaxation parameter for block jacobi smoother for MDKE."""
    method = p1  # smoothing seems to be the same for any p2 so ignore that
    nus = jnp.array([-6, -4, -2, 0, 2])
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
    w = interpax.interp1d(nu, nus, c, method="linear", extrap=(c[0], c[-1]))
    return jnp.clip(w, 0.1, 1.0)


OPTIMAL_SMOOTHING_COEFFS = {
    "1a": {
        "z": jnp.array([0.73684211, 0.73684211, 0.68421053, 0.63157895, 0.63157895]),
        "t": jnp.array([0.52631579, 0.52631579, 0.63157895, 0.63157895, 0.63157895]),
        "a": jnp.array([0.52631579, 0.57894737, 0.68421053, 0.94736842, 1.00000000]),
    },
    "1b": {
        "z": jnp.array([0.47368421, 0.47368421, 0.63157895, 0.63157895, 0.63157895]),
        "t": jnp.array([0.21052632, 0.21052632, 0.52631579, 0.63157895, 0.63157895]),
        "a": jnp.array([0.21052632, 0.21052632, 0.47368421, 0.89473684, 0.89473684]),
    },
    "2a": {
        "z": jnp.array([0.57894737, 0.57894737, 0.68421053, 0.57894737, 0.57894737]),
        "t": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.57894737, 0.57894737]),
        "a": jnp.array([0.42105263, 0.42105263, 0.52631579, 0.89473684, 0.94736842]),
    },
    "2b": {
        "z": jnp.array([0.52631579, 0.52631579, 0.68421053, 0.57894737, 0.57894737]),
        "t": jnp.array([0.31578947, 0.31578947, 0.47368421, 0.57894737, 0.57894737]),
        "a": jnp.array([0.31578947, 0.31578947, 0.52631579, 0.89473684, 0.94736842]),
    },
    "2c": {
        "z": jnp.array([0.42105263, 0.47368421, 0.68421053, 0.57894737, 0.57894737]),
        "t": jnp.array([0.21052632, 0.21052632, 0.47368421, 0.57894737, 0.57894737]),
        "a": jnp.array([0.21052632, 0.21052632, 0.42105263, 0.89473684, 0.89473684]),
    },
    "2d": {
        "z": jnp.array([0.52631579, 0.57894737, 0.63157895, 0.63157895, 0.63157895]),
        "t": jnp.array([0.52631579, 0.52631579, 0.63157895, 0.63157895, 0.63157895]),
        "a": jnp.array([0.47368421, 0.47368421, 0.57894737, 0.68421053, 0.78947368]),
    },
    "3a": {
        "z": jnp.array([0.42105263, 0.42105263, 0.57894737, 0.52631579, 0.52631579]),
        "t": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.52631579, 0.52631579]),
        "a": jnp.array([0.26315789, 0.26315789, 0.36842105, 0.84210526, 0.94736842]),
    },
    "3b": {
        "z": jnp.array([0.47368421, 0.47368421, 0.68421053, 0.57894737, 0.57894737]),
        "t": jnp.array([0.26315789, 0.26315789, 0.47368421, 0.57894737, 0.57894737]),
        "a": jnp.array([0.26315789, 0.26315789, 0.47368421, 0.89473684, 0.89473684]),
    },
    "3c": {
        "z": jnp.array([0.63157895, 0.63157895, 0.68421053, 0.57894737, 0.57894737]),
        "t": jnp.array([0.42105263, 0.42105263, 0.57894737, 0.57894737, 0.57894737]),
        "a": jnp.array([0.42105263, 0.42105263, 0.57894737, 0.89473684, 0.94736842]),
    },
    "3d": {
        "z": jnp.array([0.68421053, 0.68421053, 0.68421053, 0.57894737, 0.57894737]),
        "t": jnp.array([0.47368421, 0.47368421, 0.63157895, 0.57894737, 0.57894737]),
        "a": jnp.array([0.47368421, 0.47368421, 0.63157895, 0.94736842, 0.94736842]),
    },
    "3e": {
        "z": jnp.array([0.73684211, 0.73684211, 0.73684211, 0.63157895, 0.57894737]),
        "t": jnp.array([0.63157895, 0.63157895, 0.63157895, 0.57894737, 0.57894737]),
        "a": jnp.array([0.63157895, 0.63157895, 0.63157895, 0.84210526, 0.94736842]),
    },
    "4a": {
        "z": jnp.array([0.26315789, 0.26315789, 0.47368421, 0.47368421, 0.47368421]),
        "t": jnp.array([0.15789474, 0.15789474, 0.26315789, 0.47368421, 0.47368421]),
        "a": jnp.array([0.15789474, 0.15789474, 0.26315789, 0.73684211, 0.84210526]),
    },
    "4b": {
        "z": jnp.array([0.47368421, 0.47368421, 0.63157895, 0.57894737, 0.57894737]),
        "t": jnp.array([0.26315789, 0.26315789, 0.47368421, 0.57894737, 0.57894737]),
        "a": jnp.array([0.26315789, 0.26315789, 0.47368421, 0.89473684, 0.94736842]),
    },
    "4d": {
        "z": jnp.array([0.63157895, 0.68421053, 0.68421053, 0.57894737, 0.57894737]),
        "t": jnp.array([0.47368421, 0.47368421, 0.57894737, 0.57894737, 0.57894737]),
        "a": jnp.array([0.47368421, 0.47368421, 0.63157895, 0.94736842, 0.94736842]),
    },
    "5a": {
        "z": jnp.array([0.15789474, 0.15789474, 0.26315789, 0.42105263, 0.42105263]),
        "t": jnp.array([0.10526316, 0.10526316, 0.15789474, 0.42105263, 0.42105263]),
        "a": jnp.array([0.10526316, 0.10526316, 0.15789474, 0.57894737, 0.73684211]),
    },
    "5b": {
        "z": jnp.array([0.31578947, 0.31578947, 0.63157895, 0.57894737, 0.57894737]),
        "t": jnp.array([0.10526316, 0.10526316, 0.36842105, 0.57894737, 0.57894737]),
        "a": jnp.array([0.10526316, 0.10526316, 0.31578947, 0.78947368, 0.78947368]),
    },
    "5c": {
        "z": jnp.array([0.52631579, 0.52631579, 0.63157895, 0.57894737, 0.57894737]),
        "t": jnp.array([0.31578947, 0.31578947, 0.47368421, 0.57894737, 0.57894737]),
        "a": jnp.array([0.31578947, 0.31578947, 0.47368421, 0.89473684, 0.94736842]),
    },
    "5d": {
        "z": jnp.array([0.63157895, 0.63157895, 0.63157895, 0.57894737, 0.57894737]),
        "t": jnp.array([0.42105263, 0.42105263, 0.57894737, 0.57894737, 0.57894737]),
        "a": jnp.array([0.42105263, 0.42105263, 0.63157895, 0.94736842, 0.94736842]),
    },
}
