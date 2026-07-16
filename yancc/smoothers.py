"""Smoothing operators for multigrid."""

import warnings
from typing import Any

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import lineax as lx
from jax import config
from jaxtyping import ArrayLike, Bool, Float

from .collisions import RosenbluthPotentials
from .field import Field
from .finite_diff import fd2, fd_coeffs
from .linalg import (
    TransposedLinearOperator,
    cond_1norm_banded,
    cond_1norm_cr,
    cr_banded_factor,
    cr_banded_periodic_factor,
    cr_banded_periodic_solve,
    cr_banded_solve,
    dense_to_banded,
    lu_factor_banded,
    lu_factor_banded_periodic,
    lu_solve_banded,
    lu_solve_banded_periodic,
    matrix_1norm,
)
from .species import LocalMaxwellian, nustar
from .trajectories import DKE, MDKE, _parse_axorder_shape_3d, _parse_axorder_shape_4d
from .velocity_grids import AbstractSpeedGrid, MaxwellSpeedGrid, UniformPitchAngleGrid

# need this here as well so that consts use 64 bit
config.update("jax_enable_x64", True)

# axorder string convention: s=species, x=speed, a=pitch, t=theta, z=zeta
# Shape annotations like (ns, nx, na, nt, nz) follow the same axis-letter mapping;
# field.ntheta / field.nzeta are the underlying attribute names for nt / nz.


OPTIMAL_SMOOTHING_COEFFS_3D = {
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


# the full DKE spans many orders of magnitude in collisionality. We could allow for
# collisionality dependent weights but it seems sensitive and can lead to divergence
# if not tuned carefully, and tuning carefully for all possible problems is a nightmare
# simpler to just set a constant weight for each axis. In the future could make this
# depend on the thermal collisionality maybe (not local)?
OPTIMAL_SMOOTHING_COEFFS_4D = {
    "2d": {
        "z": jnp.array([0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70]),
        "t": jnp.array([0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70]),
        "a": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
        "x": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
        "s": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
    },
    "4d": {
        "z": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
        "t": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
        "a": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
        "x": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
        "s": jnp.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]),
    },
}


def permute_f_3d(
    f: jax.Array, field: Field, pitchgrid: UniformPitchAngleGrid, axorder: str
) -> jax.Array:
    """Rearrange elements of f to a given grid ordering."""
    shape, caxorder = _parse_axorder_shape_3d(
        field.ntheta, field.nzeta, pitchgrid.nalpha, axorder
    )
    f = f.reshape(shape)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2))
    return f.flatten()


def permute_f_4d(
    f: jax.Array,
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: AbstractSpeedGrid,
    species: list[LocalMaxwellian],
    axorder: str,
) -> jax.Array:
    """Rearrange elements of f to a given grid ordering."""
    shape, caxorder = _parse_axorder_shape_4d(
        field.ntheta, field.nzeta, pitchgrid.nalpha, speedgrid.nx, len(species), axorder
    )
    f = f.reshape(shape)
    f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
    return f.flatten()


def inverse_permute_f_3d(
    f: jax.Array, field: Field, pitchgrid: UniformPitchAngleGrid, axorder: str
) -> jax.Array:
    """Inverse of permute_f_3d: canonical (a,t,z) layout back to axorder layout."""
    nt, nz, na = field.ntheta, field.nzeta, pitchgrid.nalpha
    _, caxorder = _parse_axorder_shape_3d(nt, nz, na, axorder)
    f = f.reshape((na, nt, nz))
    f = jnp.moveaxis(f, (0, 1, 2), caxorder)
    return f.flatten()


def inverse_permute_f_4d(
    f: jax.Array,
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: AbstractSpeedGrid,
    species: list[LocalMaxwellian],
    axorder: str,
) -> jax.Array:
    """Inverse of permute_f_4d: canonical (s,x,a,t,z) layout back to axorder layout."""
    nt, nz, na = field.ntheta, field.nzeta, pitchgrid.nalpha
    nx, ns = speedgrid.nx, len(species)
    _, caxorder = _parse_axorder_shape_4d(nt, nz, na, nx, ns, axorder)
    f = f.reshape((ns, nx, na, nt, nz))
    f = jnp.moveaxis(f, (0, 1, 2, 3, 4), caxorder)
    return f.flatten()


class MDKEJacobiSmoother(lx.AbstractLinearOperator):
    """Block diagonal smoother for MDKE.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v in units of V*s/m
    nuhat : float
        Normalized collisionality, nu/v
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    smooth_solver : {None, "banded", "cr", "dense"}
        Solver to use for inverting the smoother. "banded" uses the least memory but
        can be the slowest on GPU. "dense" uses the most memory but is often the fastest
        on GPU, and competitive on CPU at moderate resolution. "cr" uses ~2x more
        memory than banded but is significantly faster on both CPU and GPU. None
        selects "cr" for large matrices or "dense" when the memory savings are small.
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
    mats: Any

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        nuhat: Float[ArrayLike, ""],
        p1: str = "2d",
        p2: int = 2,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = True,
        smooth_solver: str | None = None,
        weight: jax.Array | None = None,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.bandwidth = max(
            fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2
        )
        assert smooth_solver in {None, "banded", "cr", "dense"}
        if smooth_solver is None:
            sizes = {
                "a": self.pitchgrid.nalpha,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            # use cr solver once it actually saves memory. For the s/x axes bw = dim//2,
            # so 6*bw+1 >= dim keeps them dense.
            if sizes[self.axorder[-1]] > 6 * self.bandwidth + 1:
                smooth_solver = "cr"
            else:
                smooth_solver = "dense"
        self.smooth_solver = smooth_solver
        if weight is None:
            weight = optimal_smoothing_parameter_3d(p1, p2, nuhat, axorder[-1])
        self.weight = jnp.atleast_1d(jnp.array(weight))

        mats = MDKE(
            field, pitchgrid, erhohat, nuhat, p1, p2, axorder, gauge
        ).block_diagonal()
        # TODO: implement banded block diagonal for MDKE

        if self.smooth_solver == "banded":
            mats = dense_to_banded(self.bandwidth, self.bandwidth, mats)
            self.mats = lu_factor_banded_periodic(
                self.bandwidth,
                self.bandwidth,
                mats,
                equilibrate=True,
                pivot_tol=jnp.finfo(mats.dtype).eps ** (1 / 2),
                # unroll has little effect on CPU but ~2x faster on GPU
                unroll=4,
            )
        elif self.smooth_solver == "cr":
            mats = dense_to_banded(self.bandwidth, self.bandwidth, mats)
            self.mats = cr_banded_periodic_factor(mats, equilibrate=True)
        else:
            self.mats = jnp.linalg.inv(mats)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        with jax.named_scope(f"MDKEJacobiSmoother.mv, axorder={self.axorder}"):
            x = inverse_permute_f_3d(vector, self.field, self.pitchgrid, self.axorder)

            if self.smooth_solver == "banded":
                size, N, M = self.mats[0].shape
                x = x.reshape(size, M)
                b = lu_solve_banded_periodic(
                    self.bandwidth,
                    self.bandwidth,
                    self.mats,
                    x,
                    # unroll here has little effect on GPU but modest gain on CPU
                    unroll=8,
                )
            elif self.smooth_solver == "cr":
                M = {
                    "a": self.pitchgrid.nalpha,
                    "t": self.field.ntheta,
                    "z": self.field.nzeta,
                }[self.axorder[-1]]
                x = x.reshape(-1, M)
                b = cr_banded_periodic_solve(self.mats, x)
            else:
                size, N, M = self.mats.shape
                x = x.reshape(size, M)
                b = jnp.einsum("ijk,ik -> ij", self.mats, x[:, :])

            b = permute_f_3d(b.flatten(), self.field, self.pitchgrid, self.axorder)
            return self.weight * b

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nalpha,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nalpha,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


# We've found empirically that for certain cases, some blocks of the pitch line smoother
# amplify error significantly rather than smoothing it, causing the multigrid
# preconditioner to return garbage and stalling krylov. The problematic cases seem to
# always be 2 species, thermal collisionality ~1e-1. The problematic blocks seem to
# correspond to the slowest electrons, and the modes that get amplified tend to live
# near the turning points (b*gradB ~= 0), but gating purely based on electron speed
# or turning points doesn't seem to catch them, since its a very specific resonance.
# The best filter I've come up with is just based on the condition number of the
# smoother blocks. The condition number naturally scales with na^2 from the finite
# difference matrices, so we normalize by that. Healthy blocks usually have
# cond/na^2 ~ 1-10, asymptoting to ~40 at high collisionality. The blocks that amplify
# error are usually around cond/na^2 ~ 400, so we set a threshold at 150. Anything
# above this we switch from block jacobi to point jacobi which seems to avoid the
# blowup and stalling. Zeroing the weight for the flagged blocks also fixes it but
# point jacobi seems to do marginally better in some cases.
_PITCH_COND_GATE = 150.0


def _pitch_cond_gate(cond, na):
    """Flag pitch blocks with 1-norm cond above ``_PITCH_COND_GATE * na**2``."""
    return cond > _PITCH_COND_GATE * na**2


class DKEJacobiSmoother(lx.AbstractLinearOperator):
    """Block diagonal smoother for DKE.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
    background : list[LocalMaxwellian]
        Background species to include in the collision operator without solving for df.
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 5d array is flattened. The last axis
        denotes which direction the smoother is applied.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    smooth_solver : {None, "banded", "cr", "dense"}
        Solver to use for inverting the smoother. "banded" uses the least memory but
        can be the slowest on GPU. "dense" uses the most memory but is often the fastest
        on GPU, and competitive on CPU at moderate resolution. "cr" uses ~2x more
        memory than banded but is significantly faster on both CPU and GPU. None
        selects "cr" for large matrices or "dense" when the memory savings are small.
    weight : array-like, optional
        Under-relaxation parameter.
    operator_weights : array-like, optional


    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    background: list[LocalMaxwellian]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    bandwidth: int = eqx.field(static=True)
    smooth_solver: str = eqx.field(static=True)
    mats: Any
    weight: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: list[LocalMaxwellian] | None = None,
        potentials: RosenbluthPotentials | None = None,
        p1="2d",
        p2=2,
        axorder="sxatz",
        gauge: Bool[ArrayLike, ""] = True,
        smooth_solver: str | None = None,
        weight: jax.Array | None = None,
        operator_weights: jax.Array | None = None,
        coulomb_log=None,
    ):
        assert len(axorder) == 5 and set(axorder) == set("sxatz")
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        if self.axorder[-1] == "s":
            self.bandwidth = len(self.species) // 2
        elif self.axorder[-1] == "x":
            self.bandwidth = self.speedgrid.nx // 2
        else:
            self.bandwidth = max(
                fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2
            )
        assert smooth_solver in {None, "banded", "cr", "dense"}
        if smooth_solver is None:
            sizes = {
                "s": len(self.species),
                "x": self.speedgrid.nx,
                "a": self.pitchgrid.nalpha,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            # use cr solver once it actually saves memory. For the s/x axes bw = dim//2,
            # so 6*bw+1 >= dim keeps them dense.
            if sizes[self.axorder[-1]] > 6 * self.bandwidth + 1:
                smooth_solver = "cr"
            else:
                smooth_solver = "dense"
        if operator_weights is None:
            # defaults, zero out krook diffusion term
            operator_weights = jnp.ones(8).at[-1].set(0)

        self.smooth_solver = smooth_solver

        if weight is None:
            x = speedgrid.x
            nus = []
            for i, spa in enumerate(species):
                others = species[:i] + species[i + 1 :] + background
                nu = nustar(spa, field, x, *others, lnlambda=coulomb_log)
                nus.append(nu)
            nus = jnp.asarray(nus)
            _fun = lambda y: optimal_smoothing_parameter_4d(p1, p2, y, axorder[-1])
            _weight = jnp.vectorize(_fun)(nus)[:, :, None, None, None]
            _weight = _weight * jnp.ones(
                (1, 1, pitchgrid.nalpha, field.ntheta, field.nzeta)
            )
        else:
            _weight = weight
        self.weight = jnp.asarray(_weight).flatten()

        # "cr" consumes the same banded storage as "banded"
        bd_fmt = "banded" if self.smooth_solver in ("banded", "cr") else "dense"
        mats = DKE(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho,
            background=background,
            potentials=potentials,
            p1=p1,
            p2=p2,
            axorder=axorder,
            gauge=gauge,
            operator_weights=operator_weights,
            coulomb_log=coulomb_log,
        ).block_diagonal(bd_fmt, self.bandwidth)

        # The pitch line smoother (convolved axis "a") is the only case with a
        # non-periodic band and the only one prone to blowing up, so it gets the
        # standard banded factor/solve plus a condition-number gate replacing ill-
        # conditioned blocks with point Jacobi (drop off-diagonals).
        pitch = self.axorder[-1] == "a"
        pivot_tol = jnp.finfo(mats.dtype).eps ** (1 / 2)
        if self.smooth_solver == "banded" and not pitch:
            self.mats = lu_factor_banded_periodic(
                self.bandwidth,
                self.bandwidth,
                mats,
                equilibrate=True,
                pivot_tol=pivot_tol,
                # unroll has little effect on CPU but ~2x faster on GPU
                unroll=4,
            )
        elif self.smooth_solver == "banded":
            lu, s = lu_factor_banded(
                self.bandwidth,
                self.bandwidth,
                mats,
                equilibrate=True,
                pivot_tol=pivot_tol,
                unroll=4,
            )
            cond = cond_1norm_banded(self.bandwidth, self.bandwidth, mats, (lu, s))
            flag = _pitch_cond_gate(cond, self.pitchgrid.nalpha)
            # point Jacobi: LU of diag(A) is L = I (zero bands), U = diag(A); s = 1
            lu_pj = (
                jnp.zeros_like(lu)
                .at[:, self.bandwidth, :]
                .set(mats[:, self.bandwidth, :])
            )
            lu = jnp.where(flag[:, None, None], lu_pj, lu)
            s = jnp.where(flag[:, None], jnp.ones_like(s), s)
            self.mats = (lu, s)
        elif self.smooth_solver == "cr" and not pitch:
            self.mats = cr_banded_periodic_factor(mats, equilibrate=True)
        elif self.smooth_solver == "cr":
            # Unlike banded LU (point Jacobi = zero the off-diagonal bands) or the dense
            # inverse (point Jacobi = diag(1/diag)), the CR factor is a multi-level
            # reduction tree with no local point-Jacobi form, so we can't splice the
            # gate into it without a second factorization. Instead we keep the single
            # full-matrix factor plus the per-block flag and diagonal, and apply point
            # Jacobi (x/diag) for flagged blocks directly in `mv`.
            factors = cr_banded_factor(mats, equilibrate=True)
            cond = cond_1norm_cr(mats, factors)
            flag = _pitch_cond_gate(cond, self.pitchgrid.nalpha)
            diag = mats[:, self.bandwidth, :]
            self.mats = (factors, flag, diag)
        elif pitch:
            # dense pitch: same gate, but blocks are stored as inverses.
            anorm = matrix_1norm(mats)
            inv = jnp.linalg.inv(mats)
            cond = anorm * matrix_1norm(inv)
            na = mats.shape[-1]
            flag = _pitch_cond_gate(cond, na)
            pj = (1.0 / jnp.diagonal(mats, axis1=-2, axis2=-1))[:, None, :] * jnp.eye(
                na, dtype=mats.dtype
            )
            self.mats = jnp.where(flag[:, None, None], pj, inv)
        else:
            self.mats = jnp.linalg.inv(mats)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        with jax.named_scope(f"DKEJacobiSmoother.mv, axorder={self.axorder}"):
            x = inverse_permute_f_4d(
                vector,
                self.field,
                self.pitchgrid,
                self.speedgrid,
                self.species,
                self.axorder,
            )

            if self.smooth_solver == "banded":
                size, N, M = self.mats[0].shape
                x = x.reshape(size, M)
                # pitch ("...a") is non-periodic -> standard banded solve; periodic axes
                # (theta/zeta line smoothers) keep the wrap-aware periodic solve.
                if self.axorder[-1] == "a":
                    b = lu_solve_banded(
                        self.bandwidth, self.bandwidth, self.mats, x, unroll=8
                    )
                else:
                    b = lu_solve_banded_periodic(
                        self.bandwidth,
                        self.bandwidth,
                        self.mats,
                        x,
                        # unroll here has little effect on GPU but modest gain on CPU
                        unroll=8,
                    )
            elif self.smooth_solver == "cr":
                sizes = {
                    "s": len(self.species),
                    "x": self.speedgrid.nx,
                    "a": self.pitchgrid.nalpha,
                    "t": self.field.ntheta,
                    "z": self.field.nzeta,
                }
                M = sizes[self.axorder[-1]]
                x = x.reshape(-1, M)
                # pitch ("...a") is non-periodic; theta/zeta are periodic line smoothers
                if self.axorder[-1] == "a":
                    # gate flagged (ill-conditioned) blocks to point Jacobi (x/diag)
                    factors, flag, diag = self.mats
                    b = jnp.where(flag[:, None], x / diag, cr_banded_solve(factors, x))
                else:
                    b = cr_banded_periodic_solve(self.mats, x)
            else:
                size, N, M = self.mats.shape
                x = x.reshape(size, M)
                b = jnp.einsum("ijk,ik -> ij", self.mats, x[:, :])

            b = permute_f_4d(
                b.flatten(),
                self.field,
                self.pitchgrid,
                self.speedgrid,
                self.species,
                self.axorder,
            )
            return self.weight * b

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nalpha
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
                * self.pitchgrid.nalpha
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class DKEFrozenPlaneSmoother(lx.AbstractLinearOperator):
    """Frozen (theta, zeta)-plane smoother for DKE, applied via FFT.

    The exact (theta, zeta) plane block couples the two angle directions through the
    variable-coefficient streaming/ExB winds, giving distinct (nt*nz)^2 dense blocks per
    (s, x, a), which is too expensive to store or invert at practical resolution.
    This smoother freezes the winds to their flux-surface average c_theta, c_zeta so
    the block becomes a single constant-coefficient operator
    ``c_theta Dtheta + c_zeta Dzeta + d I`` shared across (theta, zeta). Because Dtheta,
    Dzeta are periodic-circulant, it is diagonalized by the 2d FFT: only the
    per-(s, x, a) inverse symbol 1/lambda(k_theta, k_zeta) is stored (O(N) memory) and
    the solve is FFT2 / divide / IFFT2 (O(N log(nt*nz))). The collision part enters
    exactly through the operator diagonal; only the geometry winds are frozen to their
    mean. Pair with the exact angle lines (DKEJacobiSmoother), which damp the
    frozen-approximation error the plane discards.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered.
    Erho : float
        Radial electric field, Erho = -d Phi / d rho, in Volts.
    background : list[LocalMaxwellian]
        Background species to include in the collision operator without solving for df.
    potentials : RosenbluthPotentials, optional
        Precomputed Rosenbluth potentials for the collision operator.
    p1 : str
        Stencil for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    gauge : bool
        Whether to impose the gauge constraint by fixing f at a single point.
    weight : array-like, optional
        Under-relaxation parameter, scalar. Defaults to 0.7
    operator_weights : array-like, optional
        Per-term weights for the DKE operator.
    coulomb_log : float, optional
        Coulomb logarithm for the collision operator.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    invsym: jax.Array
    weight: jax.Array

    # label used by the (verbose) multigrid smoothing loop
    axorder = "plane"

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: list[LocalMaxwellian] | None = None,
        potentials: RosenbluthPotentials | None = None,
        p1="2d",
        p2=2,
        gauge: Bool[ArrayLike, ""] = True,
        weight: jax.Array | None = None,
        operator_weights: jax.Array | None = None,
        coulomb_log=None,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        if operator_weights is None:
            operator_weights = jnp.ones(8).at[-1].set(0)
        self.weight = jnp.asarray(0.7 if weight is None else weight)

        op = DKE(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho,
            background=background,
            potentials=potentials,
            p1=p1,
            p2=p2,
            axorder="sxatz",
            gauge=gauge,
            operator_weights=operator_weights,
            coulomb_log=coulomb_log,
        )

        ns, nx, na = len(species), speedgrid.nx, pitchgrid.nalpha
        nt, nz = field.ntheta, field.nzeta
        # frozen winds: operator-weighted plane-average of the streaming/ExB coeffs,
        # flattened to (ns*nx*na,) in (s, x, a) order (matching the sxatz plane layout)
        cbar_t = (operator_weights[2] * op._opt._w).mean(axis=(3, 4)).reshape(-1)
        cbar_z = (operator_weights[3] * op._opz._w).mean(axis=(3, 4)).reshape(-1)
        # per-(s, x, a) plane mean of the full operator diagonal (collisions enter here)
        fulldiag = (
            op.diagonal().reshape(ns, nx, na, nt, nz).mean(axis=(3, 4)).reshape(-1)
        )
        # circulant symbols of the forward/backward upwind stencils (first column
        # generates it); pick the upwind stencil per block by the frozen wind's sign
        et_fd = jnp.fft.fft(op._opt._fd[:, 0])
        et_bd = jnp.fft.fft(op._opt._bd[:, 0])
        ez_fd = jnp.fft.fft(op._opz._fd[:, 0])
        ez_bd = jnp.fft.fft(op._opz._bd[:, 0])
        et = jnp.where((cbar_t > 0)[:, None], et_bd[None, :], et_fd[None, :])  # n1,nt
        ez = jnp.where((cbar_z > 0)[:, None], ez_bd[None, :], ez_fd[None, :])  # n1,nz
        # remove the frozen stencil's own diagonal so d matches the exact block mean
        # diagonal without double counting the theta/zeta stencil diagonal
        Dt00 = jnp.where(cbar_t > 0, op._opt._bd[0, 0], op._opt._fd[0, 0])
        Dz00 = jnp.where(cbar_z > 0, op._opz._bd[0, 0], op._opz._fd[0, 0])
        dbar = fulldiag - (cbar_t * Dt00 + cbar_z * Dz00)
        lam = (
            cbar_t[:, None, None] * et[:, :, None]
            + cbar_z[:, None, None] * ez[:, None, :]
            + dbar[:, None, None]
        )
        self.invsym = jnp.where(
            jnp.abs(lam) > jnp.finfo(lam.real.dtype).eps, 1.0 / lam, 0.0
        )

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        with jax.named_scope("DKEFrozenPlaneSmoother.mv"):
            n1, nt, nz = self.invsym.shape
            # native sxatz flatten -> (ns*nx*na, nt, nz); theta, zeta are inner axes
            x = vector.reshape(n1, nt, nz)
            y = jnp.fft.ifft2(
                jnp.fft.fft2(x, axes=(1, 2)) * self.invsym, axes=(1, 2)
            ).real
            return (self.weight * y).reshape(-1)

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nalpha
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return self.in_structure()

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class DKELaplacian(lx.AbstractLinearOperator):
    """Normalized Laplacian operator on 4d phase space."""

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    norm: jax.Array

    def __init__(self, field, pitchgrid, speedgrid, species, normalize=True):
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if normalize:
            na = self.pitchgrid.nalpha
            nt = self.field.ntheta
            nz = self.field.nzeta
            ha = jnp.pi / na
            ht = 2 * jnp.pi / nt
            hz = 2 * jnp.pi / nz / self.field.NFP
            # want |L@f| / |L| ~ |f|
            # uses approx 2 norm for matrix, though a bit off because D2x is not
            # symmetric like the others
            self.norm = (
                4 / ha**2 * jnp.sin(jnp.pi / 2 * (na - 1) / na) ** 2
                + 4 / ht**2 * jnp.sin(jnp.pi / 2 * (nt - 1) / nt) ** 2
                + 4 / hz**2 * jnp.sin(jnp.pi / 2 * (nz - 1) / nz) ** 2
                + jnp.max(
                    jnp.linalg.svd(self.speedgrid.D2x_pseudospectral, compute_uv=False)
                )
            )
        else:
            self.norm = jnp.array(1)

    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shape = (
            len(self.species),
            self.speedgrid.nx,
            self.pitchgrid.nalpha,
            self.field.ntheta,
            self.field.nzeta,
        )

        na = self.pitchgrid.nalpha
        nt = self.field.ntheta
        nz = self.field.nzeta
        ha = jnp.pi / na
        ht = 2 * jnp.pi / nt
        hz = 2 * jnp.pi / nz / self.field.NFP

        f = f.reshape(shape)
        fxx = jnp.einsum("yx,sxatz->syatz", self.speedgrid.D2x_pseudospectral, f)
        faa = fd2(f, 2, h=ha, bc="symmetric", axis=2)
        ftt = fd2(f, 2, h=ht, bc="periodic", axis=3)
        fzz = fd2(f, 2, h=hz, bc="periodic", axis=4)

        df = fxx + faa + ftt + fzz
        df /= self.norm
        return df.reshape(vector.shape)

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nalpha
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
                * self.pitchgrid.nalpha
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class MDKEFrozenPlaneSmoother(lx.AbstractLinearOperator):
    """Frozen (theta, zeta)-plane smoother for MDKE, applied via FFT.

    The monoenergetic analog of :class:`DKEFrozenPlaneSmoother`. The exact
    (theta, zeta) plane block couples the two angle directions through the
    variable-coefficient streaming/ExB winds, giving distinct (nt*nz)^2 dense blocks
    per pitch node, which is too expensive to store or invert at practical resolution.
    This smoother freezes the winds to their flux-surface average c_theta, c_zeta so
    the block becomes a single constant-coefficient operator
    ``c_theta Dtheta + c_zeta Dzeta + d I`` shared across (theta, zeta). Because Dtheta,
    Dzeta are periodic-circulant, it is diagonalized by the 2d FFT: only the per-pitch
    inverse symbol 1/lambda(k_theta, k_zeta) is stored (O(N) memory) and the solve is
    FFT2 / divide / IFFT2 (O(N log(nt*nz))). The pitch-angle scattering enters exactly
    through the operator diagonal; only the geometry winds are frozen to their mean.
    Pair with the exact angle lines (MDKEJacobiSmoother), which damp the
    frozen-approximation error the plane discards.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v in units of V*s/m.
    nuhat : float
        Monoenergetic collisionality, nu/v in units of 1/m.
    p1 : str
        Stencil for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    gauge : bool
        Whether to impose the gauge constraint by fixing f at a single point.
    weight : array-like, optional
        Under-relaxation parameter, scalar. Defaults to 0.7

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    invsym: jax.Array
    weight: jax.Array

    # label used by the (verbose) multigrid smoothing loop
    axorder = "plane"

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        nuhat: Float[ArrayLike, ""],
        p1="2d",
        p2=2,
        gauge: Bool[ArrayLike, ""] = True,
        weight: jax.Array | None = None,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.weight = jnp.asarray(0.7 if weight is None else weight)

        op = MDKE(field, pitchgrid, erhohat, nuhat, p1, p2, "atz", gauge)

        na = pitchgrid.nalpha
        nt, nz = field.ntheta, field.nzeta
        # frozen winds: plane-average of the streaming/ExB coeffs, one per pitch node
        # (matching the atz plane layout, pitch outermost)
        cbar_t = op._opt._w.mean(axis=(1, 2))  # (na,)
        cbar_z = op._opz._w.mean(axis=(1, 2))  # (na,)
        # per-pitch plane mean of the full operator diagonal (collisions enter here)
        fulldiag = op.diagonal().reshape(na, nt, nz).mean(axis=(1, 2))  # (na,)
        # circulant symbols of the forward/backward upwind stencils (first column
        # generates it); pick the upwind stencil per block by the frozen wind's sign
        et_fd = jnp.fft.fft(op._opt._fd[:, 0])
        et_bd = jnp.fft.fft(op._opt._bd[:, 0])
        ez_fd = jnp.fft.fft(op._opz._fd[:, 0])
        ez_bd = jnp.fft.fft(op._opz._bd[:, 0])
        et = jnp.where((cbar_t > 0)[:, None], et_bd[None, :], et_fd[None, :])  # na,nt
        ez = jnp.where((cbar_z > 0)[:, None], ez_bd[None, :], ez_fd[None, :])  # na,nz
        # remove the frozen stencil's own diagonal so d matches the exact block mean
        # diagonal without double counting the theta/zeta stencil diagonal
        Dt00 = jnp.where(cbar_t > 0, op._opt._bd[0, 0], op._opt._fd[0, 0])
        Dz00 = jnp.where(cbar_z > 0, op._opz._bd[0, 0], op._opz._fd[0, 0])
        dbar = fulldiag - (cbar_t * Dt00 + cbar_z * Dz00)
        lam = (
            cbar_t[:, None, None] * et[:, :, None]
            + cbar_z[:, None, None] * ez[:, None, :]
            + dbar[:, None, None]
        )
        self.invsym = jnp.where(
            jnp.abs(lam) > jnp.finfo(lam.real.dtype).eps, 1.0 / lam, 0.0
        )

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        with jax.named_scope("MDKEFrozenPlaneSmoother.mv"):
            na, nt, nz = self.invsym.shape
            # native atz flatten -> (na, nt, nz); theta, zeta are inner axes
            x = vector.reshape(na, nt, nz)
            y = jnp.fft.ifft2(
                jnp.fft.fft2(x, axes=(1, 2)) * self.invsym, axes=(1, 2)
            ).real
            return (self.weight * y).reshape(-1)

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nalpha,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return self.in_structure()

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


@lx.is_symmetric.register(DKELaplacian)
@lx.is_diagonal.register(DKELaplacian)
@lx.is_tridiagonal.register(DKELaplacian)
@lx.is_symmetric.register(DKEJacobiSmoother)
@lx.is_diagonal.register(DKEJacobiSmoother)
@lx.is_tridiagonal.register(DKEJacobiSmoother)
@lx.is_symmetric.register(MDKEJacobiSmoother)
@lx.is_diagonal.register(MDKEJacobiSmoother)
@lx.is_tridiagonal.register(MDKEJacobiSmoother)
@lx.is_symmetric.register(MDKEFrozenPlaneSmoother)
@lx.is_diagonal.register(MDKEFrozenPlaneSmoother)
@lx.is_tridiagonal.register(MDKEFrozenPlaneSmoother)
@lx.is_symmetric.register(DKEFrozenPlaneSmoother)
@lx.is_diagonal.register(DKEFrozenPlaneSmoother)
@lx.is_tridiagonal.register(DKEFrozenPlaneSmoother)
def _(operator):
    return False


def optimal_smoothing_parameter_3d(p1, p2, nuhat, ax):
    """Approximate best relaxation parameter for block jacobi smoother for MDKE."""
    method = p1  # smoothing seems to be the same for any p2 so ignore that
    nus = jnp.array([-6, -4, -2, 0, 2])
    nu = jnp.log10(nuhat)
    if method not in OPTIMAL_SMOOTHING_COEFFS_3D:
        warnings.warn(
            f"No optimal smoothing parameter for stencil={method}, using "
            "conservative default of w=0.1"
        )
        return jnp.array(0.1)  # conservative guess
    if ax not in OPTIMAL_SMOOTHING_COEFFS_3D[method]:
        warnings.warn(
            f"No optimal smoothing parameter for ax={ax}, using "
            "conservative default of w=0.1"
        )
        return jnp.array(0.1)  # conservative guess
    c = OPTIMAL_SMOOTHING_COEFFS_3D[method][ax]
    w = interpax.interp1d(nu, nus, c, method="linear", extrap=(c[0], c[-1]))
    return jnp.clip(w, 0.1, 1.0)


def optimal_smoothing_parameter_4d(p1, p2, nustar, ax):
    """Approximate best relaxation parameter for block jacobi smoother for DKE."""
    method = p1  # smoothing seems to be the same for any p2 so ignore that
    nus = jnp.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
    nu = jnp.log10(nustar)
    if method not in OPTIMAL_SMOOTHING_COEFFS_4D:
        warnings.warn(
            f"No optimal smoothing parameter for stencil={method}, using "
            "conservative default of w=0.01"
        )
        return jnp.array(0.01)  # conservative guess
    if ax not in OPTIMAL_SMOOTHING_COEFFS_4D[method]:
        warnings.warn(
            f"No optimal smoothing parameter for ax={ax}, using "
            "conservative default of w=0.01"
        )
        return jnp.array(0.01)  # conservative guess
    c = OPTIMAL_SMOOTHING_COEFFS_4D[method][ax]
    w = interpax.interp1d(nu, nus, c, method="linear", extrap=(c[0], c[-1]))
    return jnp.clip(w, 0.01, 1.0)
