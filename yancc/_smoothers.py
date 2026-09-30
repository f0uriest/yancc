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
from .velocity_grids import (
    AbstractPitchAngleGrid,
    MaxwellSpeedGrid,
    UniformPitchAngleGrid,
    _AbstractSpeedGrid,
)

# need this here as well so that consts use 64 bit
config.update("jax_enable_x64", True)

# axorder string convention: s=species, x=speed, a=pitch, t=theta, z=zeta
# Shape annotations like (ns, nx, na, nt, nz) follow the same axis-letter mapping;
# field.ntheta / field.nzeta are the underlying attribute names for nt / nz.


OPTIMAL_SMOOTHING_COEFFS_3D = {
    "1a": {
        "z": jnp.array([0.5789, 0.5789, 0.5789, 0.5789, 0.6316, 0.6316, 0.6316]),
        "t": jnp.array([0.5263, 0.5263, 0.5263, 0.6316, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.5263, 0.5263, 0.5263, 0.5263, 0.6842, 0.9474, 1.0000]),
    },
    "1b": {
        "z": jnp.array([0.0526, 0.0526, 0.0526, 0.1053, 0.4737, 0.6316, 0.6316]),
        "t": jnp.array([0.0526, 0.0526, 0.0526, 0.1579, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.1579, 0.1579, 0.1579, 0.1579, 0.1579, 0.2105, 0.2105]),
    },
    "2a": {
        "z": jnp.array([0.4211, 0.4211, 0.4211, 0.4211, 0.4737, 0.6316, 0.6316]),
        "t": jnp.array([0.4211, 0.4211, 0.4211, 0.4737, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.4211, 0.4211, 0.4211, 0.4211, 0.4737, 0.6842, 0.6842]),
    },
    "2b": {
        "z": jnp.array([0.1579, 0.1579, 0.1579, 0.2105, 0.4737, 0.6316, 0.6316]),
        "t": jnp.array([0.1579, 0.1579, 0.1579, 0.2632, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.2632, 0.2632, 0.2632, 0.3158, 0.3158, 0.3158, 0.3158]),
    },
    "2c": {
        "z": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.3158, 0.6316, 0.6316]),
        "t": jnp.array([0.0526, 0.0526, 0.0526, 0.1053, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.1053, 0.1053, 0.0526, 0.1053, 0.1579, 0.1579, 0.1579]),
    },
    "2d": {
        "z": jnp.array([0.5263, 0.5263, 0.5263, 0.5263, 0.6316, 0.6316, 0.6316]),
        "t": jnp.array([0.5263, 0.5263, 0.5263, 0.5789, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.5263, 0.5263, 0.5263, 0.5263, 0.6316, 0.8947, 0.8947]),
    },
    "2e": {
        "z": jnp.array([0.5263, 0.5263, 0.0526, 0.0526, 0.0526, 0.5789, 0.6316]),
        "t": jnp.array([0.5263, 0.5263, 0.0526, 0.0526, 0.0526, 0.5789, 0.6316]),
        "a": jnp.array([0.5263, 0.5263, 0.2105, 0.0526, 0.0526, 0.0526, 0.7895]),
    },
    "2f": {
        "z": jnp.array([0.4211, 0.4211, 0.3684, 0.0526, 0.0526, 0.5789, 0.6316]),
        "t": jnp.array([0.4737, 0.4737, 0.4211, 0.0526, 0.0526, 0.5789, 0.6316]),
        "a": jnp.array([0.4737, 0.4737, 0.4737, 0.0526, 0.0526, 0.0526, 0.5263]),
    },
    "2g": {
        "z": jnp.array([0.5263, 0.5263, 0.5263, 0.5263, 0.5789, 0.6316, 0.6316]),
        "t": jnp.array([0.5263, 0.5263, 0.5263, 0.5789, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.5263, 0.5263, 0.4737, 0.4737, 0.6316, 0.7895, 0.7895]),
    },
    "3a": {
        "z": jnp.array([0.1579, 0.1579, 0.1579, 0.1579, 0.3684, 0.6316, 0.6316]),
        "t": jnp.array([0.1579, 0.1579, 0.1579, 0.1579, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.2632, 0.2632, 0.2632, 0.2632, 0.3158, 0.3158, 0.3158]),
    },
    "3b": {
        "z": jnp.array([0.0526, 0.0526, 0.0526, 0.1053, 0.3684, 0.6316, 0.6316]),
        "t": jnp.array([0.0526, 0.0526, 0.0526, 0.1579, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.1579, 0.1579, 0.1579, 0.1579, 0.2105, 0.2105, 0.2105]),
    },
    "3c": {
        "z": jnp.array([0.4737, 0.4737, 0.4737, 0.4737, 0.5263, 0.6316, 0.6316]),
        "t": jnp.array([0.4211, 0.4211, 0.4211, 0.5263, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.4211, 0.4211, 0.4211, 0.4211, 0.5789, 0.7368, 0.7368]),
    },
    "3d": {
        "z": jnp.array([0.3684, 0.3684, 0.3684, 0.4737, 0.5789, 0.6316, 0.6316]),
        "t": jnp.array([0.4211, 0.4211, 0.4211, 0.5789, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.4737, 0.4737, 0.4737, 0.4737, 0.5263, 0.5263, 0.5263]),
    },
    "3e": {
        "z": jnp.array([0.5263, 0.5263, 0.5263, 0.0526, 0.0526, 0.5789, 0.6316]),
        "t": jnp.array([0.5263, 0.5263, 0.5263, 0.0526, 0.0526, 0.5789, 0.6316]),
        "a": jnp.array([0.5263, 0.5263, 0.5263, 0.0526, 0.0526, 0.0526, 0.7895]),
    },
    "3f": {
        "z": jnp.array([0.5263, 0.5263, 0.5263, 0.5263, 0.6316, 0.6316, 0.6316]),
        "t": jnp.array([0.5263, 0.5263, 0.5263, 0.5789, 0.6316, 0.6316, 0.6316]),
        "a": jnp.array([0.5263, 0.5263, 0.5263, 0.5263, 0.6316, 0.6842, 0.6842]),
    },
    "4a": {
        "z": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.1579, 0.5789, 0.5789]),
        "t": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.4737, 0.5789, 0.5789]),
        "a": jnp.array([0.0526, 0.0526, 0.0526, 0.1053, 0.1053, 0.1579, 0.1053]),
    },
    "4b": {
        "z": jnp.array([0.1053, 0.1053, 0.1053, 0.1579, 0.4211, 0.5789, 0.5789]),
        "t": jnp.array([0.1053, 0.1053, 0.1053, 0.2105, 0.5789, 0.5789, 0.5789]),
        "a": jnp.array([0.2632, 0.2632, 0.2632, 0.2632, 0.2632, 0.2632, 0.2632]),
    },
    "4d": {
        "z": jnp.array([0.4211, 0.4211, 0.4211, 0.4737, 0.5789, 0.5789, 0.5789]),
        "t": jnp.array([0.3158, 0.3158, 0.3158, 0.4737, 0.5789, 0.5789, 0.5789]),
        "a": jnp.array([0.3684, 0.3684, 0.4737, 0.4737, 0.5789, 0.5263, 0.5263]),
    },
    "4e": {
        "z": jnp.array([0.2632, 0.2632, 0.0526, 0.0526, 0.0526, 0.5789, 0.5789]),
        "t": jnp.array([0.2105, 0.2105, 0.0526, 0.0526, 0.0526, 0.5789, 0.5789]),
        "a": jnp.array([0.3158, 0.3158, 0.2105, 0.0526, 0.0526, 0.0526, 0.4211]),
    },
    "5a": {
        "z": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.5789, 0.5789]),
        "t": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.1053, 0.5789, 0.5789]),
        "a": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526]),
    },
    "5b": {
        "z": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.2105, 0.5789, 0.5789]),
        "t": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.5789, 0.5789, 0.5789]),
        "a": jnp.array([0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526]),
    },
    "5c": {
        "z": jnp.array([0.1053, 0.1053, 0.1053, 0.1579, 0.4737, 0.5789, 0.5789]),
        "t": jnp.array([0.1579, 0.1579, 0.1579, 0.2105, 0.5789, 0.5789, 0.5789]),
        "a": jnp.array([0.2632, 0.2632, 0.2632, 0.2105, 0.2632, 0.2632, 0.2632]),
    },
    "5d": {
        "z": jnp.array([0.2105, 0.2105, 0.2105, 0.3158, 0.5789, 0.5789, 0.5789]),
        "t": jnp.array([0.1579, 0.1579, 0.1579, 0.2632, 0.5789, 0.5789, 0.5789]),
        "a": jnp.array([0.3158, 0.3158, 0.3158, 0.2105, 0.3684, 0.4211, 0.3684]),
    },
    "5e": {
        "z": jnp.array([0.2143, 0.2143, 0.2143, 0.2857, 0.5000, 0.5714, 0.5714]),
        "t": jnp.array([0.2143, 0.2143, 0.2143, 0.2857, 0.5714, 0.5714, 0.5714]),
        "a": jnp.array([0.2857, 0.2857, 0.2857, 0.2143, 0.3571, 0.4286, 0.4286]),
    },
    "6a": {
        "z": jnp.array([0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.4286, 0.5714]),
        "t": jnp.array([0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.5000, 0.5714]),
        "a": jnp.array([0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714]),
    },
    "6b": {
        "z": jnp.array([0.2143, 0.2143, 0.2143, 0.0714, 0.0714, 0.5714, 0.5714]),
        "t": jnp.array([0.2143, 0.2143, 0.2143, 0.0714, 0.0714, 0.5714, 0.5714]),
        "a": jnp.array([0.2857, 0.2857, 0.0714, 0.0714, 0.0714, 0.0714, 0.3571]),
    },
    "6c": {
        "z": jnp.array([0.1429, 0.1429, 0.1429, 0.0714, 0.0714, 0.5714, 0.5714]),
        "t": jnp.array([0.1429, 0.1429, 0.1429, 0.0714, 0.0714, 0.5714, 0.5714]),
        "a": jnp.array([0.2143, 0.2143, 0.0714, 0.0714, 0.0714, 0.0714, 0.3571]),
    },
    "7a": {
        "z": jnp.array([0.1429, 0.1429, 0.1429, 0.2143, 0.4286, 0.5714, 0.5714]),
        "t": jnp.array([0.1429, 0.1429, 0.1429, 0.2143, 0.5714, 0.5714, 0.5714]),
        "a": jnp.array([0.2857, 0.2857, 0.2857, 0.2857, 0.2857, 0.2857, 0.2857]),
    },
    "7b": {
        "z": jnp.array([0.1429, 0.1429, 0.1429, 0.1429, 0.5000, 0.5714, 0.5714]),
        "t": jnp.array([0.1429, 0.1429, 0.1429, 0.2143, 0.5714, 0.5714, 0.5714]),
        "a": jnp.array([0.2143, 0.2143, 0.2143, 0.2143, 0.2857, 0.2857, 0.2857]),
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
    f: jax.Array, field: Field, pitchgrid: AbstractPitchAngleGrid, axorder: str
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
    pitchgrid: AbstractPitchAngleGrid,
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
    f: jax.Array, field: Field, pitchgrid: AbstractPitchAngleGrid, axorder: str
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
    pitchgrid: AbstractPitchAngleGrid,
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
    pitchgrid: AbstractPitchAngleGrid
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
        pitchgrid: AbstractPitchAngleGrid,
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
    pitchgrid: AbstractPitchAngleGrid
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
        pitchgrid: AbstractPitchAngleGrid,
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
            nus = _nustar_species(
                species, field, speedgrid.x, background, lnlambda=coulomb_log
            )
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


class DKELaplacian(AbstractYanccOperator):
    """Normalized Laplacian operator on 4d phase space."""

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    norm: jax.Array

    def __init__(self, field, pitchgrid, speedgrid, species, normalize=True):
        assert isinstance(pitchgrid, UniformPitchAngleGrid), (
            "DKELaplacian smoother requires a uniform pitch grid "
            "(use smooth_method='standard' for non-uniform grids)."
        )
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


def optimal_smoothing_parameter_3d(p1, p2, nuhat, ax):
    """Approximate best relaxation parameter for block jacobi smoother for MDKE."""
    method = p1  # smoothing seems to be the same for any p2 so ignore that
    nus = jnp.array([-8, -6, -4, -2, 0, 2, 4])
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
