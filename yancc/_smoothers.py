"""Smoothing operators for multigrid."""

import warnings
from typing import Any

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from jax import config
from jaxtyping import ArrayLike, Bool, Float

from ._collisions import RosenbluthPotentials
from ._finite_diff import fd2, fd_coeffs
from ._linalg import (
    AbstractYanccOperator,
    banded_to_dense,
    cr_banded_factor,
    cr_banded_periodic_factor,
    cr_banded_periodic_solve,
    cr_banded_solve,
    lu_factor_banded,
    lu_factor_banded_periodic,
    lu_solve_banded,
    lu_solve_banded_periodic,
)
from ._trajectories import DKE, MDKE, _parse_axorder_shape_3d, _parse_axorder_shape_4d
from .field import Field
from .species import LocalMaxwellian, _nustar_species
from .velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid, _AbstractSpeedGrid

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
    speedgrid: _AbstractSpeedGrid,
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
    speedgrid: _AbstractSpeedGrid,
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


class MDKEJacobiSmoother(AbstractYanccOperator):
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
        sizes = {
            "a": self.pitchgrid.nalpha,
            "t": self.field.ntheta,
            "z": self.field.nzeta,
        }
        # the band can't be wider than the axis it lies along
        self.bandwidth = min(
            max(fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2),
            sizes[self.axorder[-1]] // 2,
        )
        assert smooth_solver in {None, "banded", "cr", "dense"}
        if smooth_solver is None:
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

        # "cr" consumes the same banded storage as "banded"
        bd_fmt = "banded" if self.smooth_solver in ("banded", "cr") else "dense"
        mats = MDKE(
            field, pitchgrid, erhohat, nuhat, p1, p2, axorder, gauge
        ).block_diagonal(bd_fmt, self.bandwidth)

        # The pitch line smoother (convolved axis "a") is the only case with a
        # non-periodic band, so it uses the standard (non-periodic) banded/CR
        # factor/solve rather than the periodic variant used for theta/zeta.
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
            self.mats = lu_factor_banded(
                self.bandwidth,
                self.bandwidth,
                mats,
                equilibrate=True,
                pivot_tol=pivot_tol,
                unroll=4,
            )
        elif self.smooth_solver == "cr" and not pitch:
            self.mats = cr_banded_periodic_factor(mats, equilibrate=True)
        elif self.smooth_solver == "cr":
            self.mats = cr_banded_factor(mats, equilibrate=True)
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
                M = {
                    "a": self.pitchgrid.nalpha,
                    "t": self.field.ntheta,
                    "z": self.field.nzeta,
                }[self.axorder[-1]]
                x = x.reshape(-1, M)
                # pitch ("...a") is non-periodic; theta/zeta are periodic line smoothers
                if self.axorder[-1] == "a":
                    b = cr_banded_solve(self.mats, x)
                else:
                    b = cr_banded_periodic_solve(self.mats, x)
            else:
                size, N, M = self.mats.shape
                x = x.reshape(size, M)
                b = jnp.einsum("ijk,ik -> ij", self.mats, x[:, :])

            b = permute_f_3d(b.flatten(), self.field, self.pitchgrid, self.axorder)
            return self.weight * b

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nalpha,),
            dtype=self.field.Bmag.dtype,
        )


class DKEJacobiSmoother(AbstractYanccOperator):
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
        sizes = {
            "s": len(self.species),
            "x": self.speedgrid.nx,
            "a": self.pitchgrid.nalpha,
            "t": self.field.ntheta,
            "z": self.field.nzeta,
        }
        if self.axorder[-1] in "sx":
            self.bandwidth = sizes[self.axorder[-1]] // 2
        else:
            # the band can't be wider than the axis it lies along
            self.bandwidth = min(
                max(fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2),
                sizes[self.axorder[-1]] // 2,
            )
        assert smooth_solver in {None, "banded", "cr", "dense"}
        if smooth_solver is None:
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
            # the weight of each species depends on its thermal collisionality (at
            # x=1) only, so it is the same at every speed
            nus = _nustar_species(species, field, 1.0, background, lnlambda=coulomb_log)
            _fun = lambda y: optimal_smoothing_parameter_4d(p1, p2, y, axorder[-1])
            _weight = jnp.vectorize(_fun)(nus)[:, None, None, None, None]
            _weight = _weight * jnp.ones(
                (1, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta)
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
        # non-periodic band, so it uses the standard (non-periodic) banded/CR
        # factor/solve rather than the periodic variant used for theta/zeta.
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
            self.mats = lu_factor_banded(
                self.bandwidth,
                self.bandwidth,
                mats,
                equilibrate=True,
                pivot_tol=pivot_tol,
                unroll=4,
            )
        elif self.smooth_solver == "cr" and not pitch:
            self.mats = cr_banded_periodic_factor(mats, equilibrate=True)
        elif self.smooth_solver == "cr":
            self.mats = cr_banded_factor(mats, equilibrate=True)
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
                    b = cr_banded_solve(self.mats, x)
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


class DKEFrozenPlaneSmoother(AbstractYanccOperator):
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
    mean. It does not couple different pitch, speed or species nodes, so it is meant to
    be composed with smoothers that do. The exact theta and zeta lines can additionally
    damp the frozen-approximation error the plane discards.

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


class DKELaplacian(AbstractYanccOperator):
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


class MDKEFrozenPlaneSmoother(AbstractYanccOperator):
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
    It does not couple different pitch nodes, so it is meant to be composed with
    smoothers that do. The exact theta and zeta lines can additionally damp the
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

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nalpha,),
            dtype=self.field.Bmag.dtype,
        )


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


class DKEL01LineSmoother(AbstractYanccOperator):
    """Angle line smoother on the l = 0, 1 Legendre subspace in pitch.

    The pitch dependence of ``f`` is projected onto the first two Legendre
    polynomials and back,

        c_l = (2l + 1)/2 sum_a w_a P_l(xi_a) f_a,    f_a = sum_l P_l(xi_a) c_l,

    written ``c = W f`` and ``f = Q c`` with ``W Q = I``. For each species, speed and
    node of the other angle, the operator restricted to one angle line and to this
    subspace is the ``(2n, 2n)`` block, with ``n`` the number of points on the line,

        B[(i, l), (j, m)] = sum_a W[l, a] D_a[i, j] Q[a, m]
                          + delta_ij sum_{a, b} W[l, a] A'[a, b] Q[b, m],

    where ``D_a`` is the DKE operator coupling points ``i, j`` along the line at pitch
    node ``a``, and ``A'`` is the operator's pitch coupling at point ``i`` with its
    diagonal removed. The smoother applies ``M r = weight * Q B^-1 W r`` line by line.

    It is a subspace correction: its range is the span of ``P_0`` and ``P_1``, so it is
    meant to be composed with smoothers that act on the full pitch dependence.

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
        Background species included in the collision operator without solving for df.
    p1 : str
        Order/type of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    line : {"t", "z"}
        Which angle line to solve.
    weight : array-like, optional
        Under-relaxation parameter.
    operator_weights : array-like, optional
        Per-term weights of the DKE operator.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    background: list[LocalMaxwellian]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    line: str = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    weight: jax.Array
    _inv: jax.Array
    _W: jax.Array
    _Q: jax.Array

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
        line: str = "t",
        gauge: Bool[ArrayLike, ""] = True,
        weight: jax.Array | None = None,
        operator_weights: jax.Array | None = None,
        coulomb_log=None,
    ):
        if line not in ("t", "z"):
            raise ValueError(f"line must be 't' or 'z', got {line}")
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        self.p1 = p1
        self.p2 = p2
        self.line = line
        # exposed for the multigrid verbose trace, which prints Mi.axorder
        self.axorder = f"l01{line}"
        if operator_weights is None:
            operator_weights = jnp.ones(8).at[-1].set(0)
        self.weight = jnp.asarray(1.0 if weight is None else weight)

        ns, nx = len(species), speedgrid.nx
        na, nt, nz = pitchgrid.nalpha, field.ntheta, field.nzeta
        n = nt if line == "t" else nz
        nother = nz if line == "t" else nt

        xi = jnp.asarray(pitchgrid.xi)
        wxi = jnp.asarray(pitchgrid.wxi)
        # Q: reconstruction P_l(xi); W: projection (2l+1)/2 w_xi P_l  ->  W @ Q = I
        Q = jnp.stack([jnp.ones_like(xi), xi], axis=1)
        W = jnp.stack([0.5 * wxi, 1.5 * wxi * xi], axis=0)
        self._W, self._Q = W, Q

        def _op(axorder: str) -> DKE:
            return DKE(
                field=field,
                pitchgrid=pitchgrid,
                speedgrid=speedgrid,
                species=species,
                Erho=Erho,
                background=background,
                potentials=potentials,
                p1=p1,
                p2=p2,
                axorder=axorder,
                gauge=gauge,
                operator_weights=operator_weights,
                coulomb_log=coulomb_log,
            )

        # line blocks, pitch a spectator: leading axes (s, x, other, a). The line
        # operator is a finite difference stencil, so the blocks are banded and are
        # projected in banded storage, only expanding the small projected blocks.
        bw = min(max(fd_coeffs[1][p1].size // 2, fd_coeffs[2][p2].size // 2), n // 2)
        axD = "sxzat" if line == "t" else "sxtaz"
        D = _op(axD).block_diagonal("banded", bw=bw)
        D = D.reshape(ns, nx, nother, na, 2 * bw + 1, n)
        t1 = jnp.einsum("la,sxoahj,am->sxolmhj", W, D, Q)
        del D
        t1 = banded_to_dense(bw, bw, t1)

        # projected pitch blocks with the pitch diagonal removed, since the line
        # blocks already carry the full pointwise diagonal: leading axes (s, x, t, z).
        # Only the pitch, pitch angle scattering and field particle terms couple
        # different pitch nodes, the rest only add to the diagonal. Rather than
        # forming the (na, na) blocks, which are dense due to the field particle
        # term, W A Q is found by applying each term to the columns of Q.
        op = _op("sxatz")
        ow = operator_weights
        shape = (ns, nx, na, nt, nz)

        def pitch_mv(v):
            return ow[1] * op._opa.mv(v) + ow[4] * op._C.CL.mv(v)

        # pitch and pitch angle scattering terms are block diagonal in (s, x, t, z),
        # so a single probe per column of Q covers every block at once
        k2 = jnp.stack(
            [
                jnp.einsum(
                    "la,sxatz->sxtzl",
                    W,
                    pitch_mv(
                        jnp.broadcast_to(
                            Q[None, None, :, m, None, None], shape
                        ).reshape(-1)
                    ).reshape(shape),
                )
                for m in range(2)
            ],
            axis=-1,
        )

        # field particle term couples all speeds and species, so each (s, x) needs
        # its own probe
        def field_particle_probe(idx):
            s0, x0, m = idx
            q = jnp.broadcast_to(Q[:, m][:, None, None], (na, nt, nz))
            v = jnp.zeros(shape).at[s0, x0].set(q)
            y = op._C.CF.mv(v.reshape(-1)).reshape(shape)
            y = jax.lax.dynamic_index_in_dim(y, s0, 0, keepdims=False)
            y = jax.lax.dynamic_index_in_dim(y, x0, 0, keepdims=False)
            return jnp.einsum("la,atz->tzl", W, y)

        idx = jnp.stack(
            jnp.meshgrid(jnp.arange(ns), jnp.arange(nx), jnp.arange(2), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        kf = jax.lax.map(field_particle_probe, idx).reshape(ns, nx, 2, nt, nz, 2)
        k2 = k2 + ow[6] * jnp.moveaxis(kf, 2, -1)
        diag = (
            ow[1] * op._opa.diagonal()
            + ow[4] * op._C.CL.diagonal()
            + ow[6] * op._C.CF.diagonal()
        ).reshape(shape)
        k2 = k2 - jnp.einsum("la,sxatz,am->sxtzlm", W, diag, Q)
        # reorder to (s, x, other, line) so the line index is the block-diagonal one
        if line == "t":
            k2 = jnp.moveaxis(k2, 2, 3)

        nblk = ns * nx * nother
        t1 = t1.reshape(nblk, 2, 2, n, n)
        k2 = k2.reshape(nblk, n, 2, 2)
        # interleave as (i, l) so the band stays narrow and the block stays local
        B = jnp.transpose(t1, (0, 3, 1, 4, 2)) + jnp.einsum(
            "bilm,ij->biljm", k2, jnp.eye(n)
        )
        self._inv = jnp.linalg.inv(B.reshape(nblk, 2 * n, 2 * n))

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        with jax.named_scope(f"DKEL01LineSmoother.mv, line={self.line}"):
            ns, nx = len(self.species), self.speedgrid.nx
            na = self.pitchgrid.nalpha
            nt, nz = self.field.ntheta, self.field.nzeta
            n = nt if self.line == "t" else nz
            nother = nz if self.line == "t" else nt
            nblk = ns * nx * nother

            f = vector.reshape(ns, nx, na, nt, nz)
            # project onto {P_0, P_1} directly in block layout (s, x, other, line, l)
            lay = "sxztl" if self.line == "t" else "sxtzl"
            v = jnp.einsum(f"la,sxatz->{lay}", self._W, f).reshape(nblk, 2 * n)
            y = jnp.einsum("bij,bj->bi", self._inv, v)
            y = y.reshape(ns, nx, nother, n, 2)
            # reconstruct onto the pitch grid straight from the block layout
            out = jnp.einsum(f"al,{lay}->sxatz", self._Q, y)
            return (self.weight * out).reshape(-1)

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


class MDKEL01LineSmoother(AbstractYanccOperator):
    """Angle line smoother for MDKE on the l = 0, 1 Legendre subspace in pitch.

    The monoenergetic analog of :class:`DKEL01LineSmoother`. For each node of the
    other angle, the operator restricted to one angle line and to the span of
    ``P_0, P_1`` in pitch is the ``(2n, 2n)`` block

        B[(i, l), (j, m)] = sum_a W[l, a] D_a[i, j] Q[a, m]
                          + delta_ij sum_{a, b} W[l, a] A'[a, b] Q[b, m],

    with ``D_a`` the operator along the line at pitch node ``a`` and ``A'`` the pitch
    coupling at point ``i`` with its diagonal removed. The smoother applies
    ``M r = weight * Q B^-1 W r``. Its range is the span of ``P_0`` and ``P_1``, so it
    is meant to be composed with smoothers that act on the full pitch dependence.

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
    line : {"t", "z"}
        Which angle line to solve.
    gauge : bool
        Whether to impose the gauge constraint by fixing f at a single point.
    weight : array-like, optional
        Under-relaxation parameter.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    line: str = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    weight: jax.Array
    _inv: jax.Array
    _W: jax.Array
    _Q: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        nuhat: Float[ArrayLike, ""],
        p1="2d",
        p2=2,
        line: str = "t",
        gauge: Bool[ArrayLike, ""] = True,
        weight: jax.Array | None = None,
    ):
        if line not in ("t", "z"):
            raise ValueError(f"line must be 't' or 'z', got {line}")
        self.field = field
        self.pitchgrid = pitchgrid
        self.line = line
        # exposed for the multigrid verbose trace, which prints Mi.axorder
        self.axorder = f"l01{line}"
        self.weight = jnp.asarray(1.0 if weight is None else weight)

        na, nt, nz = pitchgrid.nalpha, field.ntheta, field.nzeta
        n = nt if line == "t" else nz
        nother = nz if line == "t" else nt

        xi = jnp.asarray(pitchgrid.xi)
        wxi = jnp.asarray(pitchgrid.wxi)
        # Q: reconstruction P_l(xi); W: projection (2l+1)/2 w_xi P_l  ->  W @ Q = I
        Q = jnp.stack([jnp.ones_like(xi), xi], axis=1)
        W = jnp.stack([0.5 * wxi, 1.5 * wxi * xi], axis=0)
        self._W, self._Q = W, Q

        # line blocks with pitch a spectator, leading axes (other, a), projected in
        # banded storage and only then expanded
        bw = min(max(fd_coeffs[1][p1].size // 2, fd_coeffs[2][p2].size // 2), n // 2)
        axD = "zat" if line == "t" else "taz"
        D = MDKE(field, pitchgrid, erhohat, nuhat, p1, p2, axD, gauge)
        D = D.block_diagonal("banded", bw=bw).reshape(nother, na, 2 * bw + 1, n)
        t1 = jnp.einsum("la,oahj,am->olmhj", W, D, Q)
        del D
        t1 = banded_to_dense(bw, bw, t1)

        # projected pitch blocks with the pitch diagonal removed, since the line
        # blocks already carry the full pointwise diagonal. Only the pitch and pitch
        # angle scattering terms couple different pitch nodes, and both are local in
        # (theta, zeta), so one probe per column of Q gives W A Q at every point.
        op = MDKE(field, pitchgrid, erhohat, nuhat, p1, p2, "atz", gauge)

        def pitch_mv(v):
            return op._opa.mv(v) + op._opp.mv(v)

        k2 = jnp.stack(
            [
                jnp.einsum(
                    "la,atz->tzl",
                    W,
                    pitch_mv(
                        jnp.broadcast_to(Q[:, m, None, None], (na, nt, nz)).reshape(-1)
                    ).reshape(na, nt, nz),
                )
                for m in range(2)
            ],
            axis=-1,
        )
        diag = (op._opa.diagonal() + op._opp.diagonal()).reshape(na, nt, nz)
        k2 = k2 - jnp.einsum("la,atz,am->tzlm", W, diag, Q)
        # reorder to (other, line) so the line index is the block-diagonal one
        if line == "t":
            k2 = jnp.moveaxis(k2, 0, 1)

        # interleave as (i, l) so the band stays narrow and the block stays local
        B = jnp.transpose(t1, (0, 3, 1, 4, 2)) + jnp.einsum(
            "oilm,ij->oiljm", k2, jnp.eye(n)
        )
        self._inv = jnp.linalg.inv(B.reshape(nother, 2 * n, 2 * n))

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        with jax.named_scope(f"MDKEL01LineSmoother.mv, line={self.line}"):
            na = self.pitchgrid.nalpha
            nt, nz = self.field.ntheta, self.field.nzeta
            n = nt if self.line == "t" else nz
            nother = nz if self.line == "t" else nt

            f = vector.reshape(na, nt, nz)
            # project onto {P_0, P_1} directly in block layout (other, line, l)
            lay = "ztl" if self.line == "t" else "tzl"
            v = jnp.einsum(f"la,atz->{lay}", self._W, f).reshape(nother, 2 * n)
            y = jnp.einsum("bij,bj->bi", self._inv, v).reshape(nother, n, 2)
            out = jnp.einsum(f"al,{lay}->atz", self._Q, y)
            return (self.weight * out).reshape(-1)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nalpha,),
            dtype=self.field.Bmag.dtype,
        )
