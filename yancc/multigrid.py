"""Stuff for multigrid cycles."""

import functools
from typing import Optional, Union

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Int

from .field import Field
from .linalg import InverseLinearOperator, TransposedLinearOperator
from .smoothers import (
    DKEJacobi2Smoother,
    DKEJacobiSmoother,
    DKELaplacian,
    MDKEJacobiSmoother,
)
from .trajectories import DKE, MDKE
from .velocity_grids import AbstractPitchAngleGrid


@functools.partial(jax.jit, static_argnames=["p1", "p2"])
@jax.named_call
def get_mdke_operators(fields, pitchgrids, erhohat, nuhat, p1, p2, gauge, **options):
    """Get multigrid operators for each field, pitchgrid."""
    operators = []
    for field, pitchgrid in zip(fields, pitchgrids):
        op = MDKE(
            field, pitchgrid, erhohat, nuhat, p1=p1, p2=p2, gauge=gauge, **options
        )
        operators.append(op)
    return operators


@functools.partial(jax.jit, static_argnames=["p1", "p2"])
@jax.named_call
def get_dke_operators(
    fields,
    pitchgrids,
    speedgrid,
    species,
    Erho,
    background,
    potentials,
    p1,
    p2,
    gauge,
    coulomb_log=None,
    **options,
):
    """Get multigrid operators for each field, pitchgrid."""
    operators = []
    for field, pitchgrid in zip(fields, pitchgrids):
        op = DKE(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho,
            background,
            potentials,
            p1=p1,
            p2=p2,
            gauge=gauge,
            coulomb_log=coulomb_log,
            **options,
        )
        operators.append(op)
    return operators


@eqx.filter_jit
@jax.named_call
def get_mdke_jacobi_smoothers(
    fields,
    pitchgrids,
    erhohat,
    nuhat,
    p1,
    p2,
    gauge,
    smooth_solver,
    weight,
    **options,
):
    """Get multigrid smoothers for each field, pitchgrid."""
    smoothers = []
    for field, pitchgrid in zip(fields, pitchgrids):
        smooth = [
            MDKEJacobiSmoother(
                field,
                pitchgrid,
                erhohat,
                nuhat,
                axorder=order,
                p1=p1,
                p2=p2,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=weight,
                **options,
            )
            for order in ["atz", "zat", "tza"]
        ]
        smoothers.append(smooth)
    return smoothers


@eqx.filter_jit
@jax.named_call
def get_dke_jacobi_smoothers(
    fields,
    pitchgrids,
    speedgrid,
    species,
    Erho,
    background,
    potentials,
    p1,
    p2,
    gauge,
    smooth_solver,
    weight,
    coulomb_log=None,
    **options,
):
    """Get multigrid smoothers for each field, pitchgrid."""
    smoothers = []
    for field, pitchgrid in zip(fields, pitchgrids):
        smooth = [
            DKEJacobiSmoother(
                field,
                pitchgrid,
                speedgrid,
                species,
                Erho,
                background,
                potentials,
                p1=p1,
                p2=p2,
                axorder=order,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=weight,
                coulomb_log=coulomb_log,
                **options,
            )
            for order in ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"]
        ]
        smoothers.append(smooth)
    return smoothers


@eqx.filter_jit
@jax.named_call
def get_dke_jacobi2_smoothers(
    fields,
    pitchgrids,
    speedgrid,
    species,
    Erho,
    background,
    potentials,
    p1,
    p2,
    gauge,
    smooth_solver,
    weight,
    coulomb_log=None,
    **options,
):
    """Get multigrid smoothers for each field, pitchgrid."""
    smoothers = []
    for field, pitchgrid in zip(fields, pitchgrids):
        smooth = [
            DKEJacobi2Smoother(
                field,
                pitchgrid,
                speedgrid,
                species,
                Erho,
                background,
                potentials,
                p1=p1,
                p2=p2,
                axorder=order,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=weight,
                coulomb_log=coulomb_log,
                **options,
            )
            for order in ["atzsx", "tzasx", "zatsx"]
        ]
        smoothers.append(smooth)
    return smoothers


def _half_next_odd(k: int, m: Union[int, float] = 2):
    if int(k // m) == 0:
        return 1
    elif int(k // m) % 2 == 0:
        return int(k // m + 1)
    else:
        return int(k // m)


def get_grid_resolutions(
    ns: int,
    nx: int,
    na: int,
    nt: int,
    nz: int,
    coarse_N: int = 8000,
    min_na: int = 5,
    min_nt: int = 5,
    min_nz: int = 5,
    max_grids: Optional[int] = None,
    coarsening_factor: Optional[Union[int, float]] = None,
) -> list[tuple]:
    """Determine resolutions for multigrid scheme.

    Parameters
    ----------
    ns, nx, na, nt, nz: int
        Resolutions in each coordinate of the finest grid.
    coarse_N : int
        Approximate desired size of the coarsest grid.
    min_na, min_nt, min_nz : int
        Minimum resolution in each coordinate.
    max_grids : int
        Maximum number of grids in the multigrid scheme.
    coarsening_factor : int, float
        How much to coarsen the grid in each coordinate at each level. Defaults to 2.

    Returns
    -------
    resolutions : list of tuple of int
        Each list element is a tuple of resolutions at a given grid level, each
        tuple is the resolution (ns, nx, na, nt, nz)
    """
    coarse_N = max(coarse_N, ns * nx * min_na * min_nt * min_nz)
    N = ns * nx * na * nt * nz
    dim = 2 if nz == 1 else 3  # tokamak vs stellarator

    if coarsening_factor is not None and max_grids is not None:
        raise ValueError("Cannot specify both coarsening_factor and max_grids")
    elif coarsening_factor is None and max_grids is None:
        coarsening_factor = 2.5
        max_grids = int(
            np.ceil(np.log(N / coarse_N) / np.log(coarsening_factor**dim) + 1)
        )
    elif coarsening_factor is None:
        assert isinstance(max_grids, int)
        coarsening_factor = float((N / coarse_N) ** (1 / (dim * (max_grids - 1))))
        coarsening_factor = max(2, coarsening_factor)
    elif max_grids is None:
        max_grids = int(
            np.ceil(np.log(N / coarse_N) / np.log(coarsening_factor**dim) + 1)
        )

    resolutions = [(ns, nx, na, nt, nz)]
    na = max(_half_next_odd(na, coarsening_factor), min_na)
    nt = max(_half_next_odd(nt, coarsening_factor), min_nt)
    nz = max(_half_next_odd(nz, coarsening_factor), min_nz)
    N = ns * nx * na * nt * nz
    while N > coarse_N and len(resolutions) < max_grids - 1:
        resolutions.append((ns, nx, na, nt, nz))
        na = max(_half_next_odd(na, coarsening_factor), min_na)
        nt = max(_half_next_odd(nt, coarsening_factor), min_nt)
        nz = max(_half_next_odd(nz, coarsening_factor), min_nz)
        N = ns * nx * na * nt * nz
    resolutions.append((ns, nx, na, nt, nz))
    return resolutions[::-1]


@eqx.filter_jit
@jax.named_call
def get_fields_grids(
    field,
    pitchgrid,
    resolutions,
):
    """Get fields and grids for multigrid problem.

    Parameters
    ----------
    field : Field
        Field at sufficient resolution to represent B.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    resolutions : array-like, shape(num_grid, 5)
        Resolutions at each grid level in (ns, nx, na, nt, nz)

    Returns
    -------
    fields : list[Field]
        fields at each resolution level
    pitchgrids : list[PitchAngleGrid]
        grids at each resolution level

    """
    fields = []
    grids = []
    for res in resolutions:
        _, _, na, nt, nz = res
        fields.append(field.resample(nt, nz))
        grids.append(pitchgrid.resample(na))
    return fields, grids


@functools.partial(jax.jit, static_argnames=["prefix_size", "method"])
@jax.named_call
def get_prolongations(fields, pitchgrids, prefix_size=1, method="linear"):
    """Build coarse->fine prolongation operators between adjacent grid levels.

    Parameters
    ----------
    fields : list[Field]
        Fields at each level, ordered coarse to fine.
    pitchgrids : list[AbstractPitchAngleGrid]
        Pitch angle grids at each level, ordered coarse to fine.
    prefix_size : int
        Product of leading axes that don't change between levels (e.g.
        ``len(species) * speedgrid.nx``).
    method : str
        Interpolation method.

    Returns
    -------
    prolongations : list[Prolongation]
        ``prolongations[k]`` maps level ``k`` (coarse) to level ``k+1`` (fine),
        for ``k`` in ``0 .. len(fields) - 2``.
    """
    return [
        Prolongation(
            field_coarse=fields[k],
            field_fine=fields[k + 1],
            pitchgrid_coarse=pitchgrids[k],
            pitchgrid_fine=pitchgrids[k + 1],
            prefix_size=prefix_size,
            method=method,
        )
        for k in range(len(fields) - 1)
    ]


@functools.partial(jax.jit, static_argnames=["prefix_size", "method"])
@jax.named_call
def get_restrictions(fields, pitchgrids, prefix_size=1, method="linear"):
    """Build fine->coarse restriction operators between adjacent grid levels.

    Parameters
    ----------
    fields : list[Field]
        Fields at each level, ordered coarse to fine.
    pitchgrids : list[AbstractPitchAngleGrid]
        Pitch angle grids at each level, ordered coarse to fine.
    prefix_size : int
        Product of leading axes that don't change between levels (e.g.
        ``len(species) * speedgrid.nx``).
    method : str
        Interpolation method that defines the underlying prolongation.

    Returns
    -------
    restrictions : list[Restriction]
        ``restrictions[k]`` maps level ``k+1`` (fine) to level ``k`` (coarse),
        for ``k`` in ``0 .. len(fields) - 2``.
    """
    return [
        Restriction(
            field_coarse=fields[k],
            field_fine=fields[k + 1],
            pitchgrid_coarse=pitchgrids[k],
            pitchgrid_fine=pitchgrids[k + 1],
            prefix_size=prefix_size,
            method=method,
        )
        for k in range(len(fields) - 1)
    ]


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def standard_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False, r0=None):
    """Apply smoothing operators to operator @ x = rhs.

    Returns (x, r) with r = rhs - operator.mv(x). The residual is maintained as
    part of the loop carry so callers can avoid a separate `rhs - operator.mv(x)`
    mv after smoothing. Pass r0 if the initial residual is known cheaply (e.g.,
    r0=rhs when x is zero) to skip the initial residual mv.
    """
    if r0 is None:
        r0 = rhs - operator.mv(x)

    def body(k, state):
        x, r = state
        for i, Mi in enumerate(smoothers):
            dx = Mi.mv(r)
            x = x + dx
            r = rhs - operator.mv(x)
            if verbose:
                err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
                jax.debug.print(
                    "v={k} after {a} err: {err:.3e}",
                    err=err,
                    k=k,
                    a=Mi.axorder,
                    ordered=True,
                )
        return x, r

    x, r = jax.lax.fori_loop(0, nsteps, body, (x, r0))
    return x, r


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def adpative_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False, r0=None):
    """Apply smoothing operators to operator @ x = rhs.

    Returns (x, r) with r = rhs - operator.mv(x). Pass r0 to skip the initial
    residual mv (e.g., r0=rhs when x is zero).
    """
    if r0 is None:
        r0 = rhs - operator.mv(x)
    res0 = res1 = jnp.linalg.norm(r0)

    def cond(state):
        k, x, r, res0, res1 = state
        # do at least 1 step but may stop early if residuals are increasing
        # note that this is just a heuristic. Residuals may increase even though error
        # decreases, but increasing residual can cause problems when used as a
        # preconditioner with GMRES.
        return (k < jnp.abs(nsteps)) & (res1 <= res0)

    def body(state):
        k, x, r, res0, res1 = state
        for i, Mi in enumerate(smoothers):
            dx = Mi.mv(r)
            x = x + dx
            r = rhs - operator.mv(x)
            if verbose:
                err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
                jax.debug.print(
                    "v={k} after {a} err: {err:.3e}",
                    err=err,
                    k=k,
                    a=Mi.axorder,
                    ordered=True,
                )
        res0 = res1
        res1 = jnp.linalg.norm(r)
        return k + 1, x, r, res0, res1

    _, x, r, _, _ = jax.lax.while_loop(cond, body, (0, x, r0, res0, res1))
    return x, r


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov1_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False, r0=None):
    """Apply smoothing operators to operator @ x = rhs.

    Returns (x, r) with r = rhs - operator.mv(x). Pass r0 to skip the initial
    residual mv (e.g., r0=rhs when x is zero).
    """
    if r0 is None:
        r0 = rhs - operator.mv(x)

    def body(k, state):
        x0, r = state
        rs = jnp.empty((len(smoothers) + 1, rhs.size))
        dxs = jnp.empty((len(smoothers), rhs.size))
        x = x0

        for i, Mi in enumerate(smoothers):
            rs = rs.at[i].set(r)
            dx = Mi.mv(r)
            dxs = dxs.at[i].set(dx)
            x = x + dx
            r = rhs - operator.mv(x)

        rs = rs.at[-1].set(r)

        rb = rs[0]
        dr = -jnp.diff(rs, axis=0)

        alpha = jnp.linalg.lstsq(dr.T, rb)[0]
        x = x0 + dxs.T @ alpha
        # r_new = rhs - A x = rs[0] - dr.T @ alpha (free, since dr[i] = A dx_i)
        r = rb - dr.T @ alpha

        if verbose:
            err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "v={k} err: {err:.3e} alpha: {alpha}",
                err=err,
                k=k,
                alpha=alpha,
                ordered=True,
            )

        return x, r

    x, r = jax.lax.fori_loop(0, nsteps, body, (x, r0))
    return x, r


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov1s_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False, r0=None):
    """Apply smoothing operators to operator @ x = rhs.

    Returns (x, r) with r = rhs - operator.mv(x). Pass r0 to skip the initial
    residual mv (e.g., r0=rhs when x is zero).
    """
    L = DKELaplacian(
        operator.field, operator.pitchgrid, operator.speedgrid, operator.species, True
    )

    if r0 is None:
        r0 = rhs - operator.mv(x)

    def body(k, state):
        x0, r = state
        rs = jnp.empty((len(smoothers), rhs.size))
        dxs = jnp.empty((len(smoothers), rhs.size))
        x = x0
        rs = rs.at[0].set(r)

        for i, Mi in enumerate(smoothers):
            dx = Mi.mv(r)
            dxs = dxs.at[i].set(dx)
            x = x + dx
            if i + 1 < len(smoothers):
                r = rhs - operator.mv(x)
                rs = rs.at[i + 1].set(r)

        Ldxs = jax.vmap(L.mv)(dxs)
        dxs = jnp.concatenate([dxs, Ldxs])
        Adxs = jax.vmap(operator.mv)(dxs)
        alpha = jnp.linalg.lstsq(Adxs.T, rs[0])[0]
        x = x0 + dxs.T @ alpha
        # r_new = rhs - A x = rs[0] - Adxs.T @ alpha (free)
        r = rs[0] - Adxs.T @ alpha

        if verbose:
            err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "v={k} err: {err:.3e} alpha: {alpha}",
                err=err,
                k=k,
                alpha=alpha,
                ordered=True,
            )

        return x, r

    x, r = jax.lax.fori_loop(0, nsteps, body, (x, r0))
    return x, r


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov2_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False, r0=None):
    """Apply smoothing operators to operator @ x = rhs.

    Returns (x, r) with r = rhs - operator.mv(x). Pass r0 to skip the initial
    residual mv (e.g., r0=rhs when x is zero).
    """
    if r0 is None:
        r0 = rhs - operator.mv(x)

    def body(k, state):
        x0, r = state
        dxs = jnp.empty((len(smoothers), rhs.size))
        Adxs = jnp.empty((len(smoothers), rhs.size))
        x = x0

        for i, Mi in enumerate(smoothers):
            dx = Mi.mv(r)
            dxs = dxs.at[i].set(dx)
            Adxs = Adxs.at[i].set(operator.mv(dx))

        alpha = jnp.linalg.lstsq(Adxs.T, r)[0]
        x = x0 + dxs.T @ alpha
        # r_new = rhs - A x = r - Adxs.T @ alpha (free, since Adxs[i] = A dx_i)
        r = r - Adxs.T @ alpha

        if verbose:
            err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "v={k} err: {err:.3e} alpha: {alpha}",
                err=err,
                k=k,
                alpha=alpha,
                ordered=True,
            )

        return x, r

    x, r = jax.lax.fori_loop(0, nsteps, body, (x, r0))
    return x, r


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov2s_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False, r0=None):
    """Apply smoothing operators to operator @ x = rhs.

    Returns (x, r) with r = rhs - operator.mv(x). Pass r0 to skip the initial
    residual mv (e.g., r0=rhs when x is zero).
    """
    L = DKELaplacian(
        operator.field, operator.pitchgrid, operator.speedgrid, operator.species, True
    )

    if r0 is None:
        r0 = rhs - operator.mv(x)

    def body(k, state):
        x0, r = state
        dxs = jnp.empty((len(smoothers), rhs.size))
        x = x0

        for i, Mi in enumerate(smoothers):
            dx = Mi.mv(r)
            dxs = dxs.at[i].set(dx)

        Ldxs = jax.vmap(L.mv)(dxs)
        dxs = jnp.concatenate([dxs, Ldxs])
        Adxs = jax.vmap(operator.mv)(dxs)
        alpha = jnp.linalg.lstsq(Adxs.T, r)[0]
        x = x0 + dxs.T @ alpha
        # r_new = rhs - A x = r - Adxs.T @ alpha (free, since Adxs[i] = A dxs[i])
        r = r - Adxs.T @ alpha

        if verbose:
            err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "v={k} err: {err:.3e} alpha: {alpha}",
                err=err,
                k=k,
                alpha=alpha,
                ordered=True,
            )

        return x, r

    x, r = jax.lax.fori_loop(0, nsteps, body, (x, r0))
    return x, r


def _build_interp_matrix(x_src, x_query, method, period):
    """Dense 1-D interpolation matrix.

    Returns ``P`` of shape ``(len(x_query), len(x_src))`` such that
    ``P @ f`` is the interpolant of ``f`` (defined on ``x_src``) evaluated at
    ``x_query``. Built by applying ``interpax.interp1d`` to each column of the
    identity matrix; valid for any linear interpolation method since
    ``interp1d`` is linear in its data argument.
    """
    eye = jnp.eye(len(x_src))
    P_T = jax.vmap(
        lambda col: interpax.interp1d(
            x_query, x_src, col, method=method, period=period, extrap=True
        )
    )(eye)
    return P_T.T


class Prolongation(lx.AbstractLinearOperator):
    """Coarse-to-fine grid prolongation as a linear operator.

    Interpolates a flattened ``(prefix_size, na, ntheta, nzeta)`` array from a
    coarse ``(field, pitchgrid)`` up to a fine one via 1-D interpolation along
    each coordinate axis. ``prefix_size`` absorbs any leading dimensions that
    don't change between levels (e.g. species, speed).

    Parameters
    ----------
    field_coarse, field_fine : Field
        Magnetic field data at the coarse and fine theta/zeta resolutions.
    pitchgrid_coarse, pitchgrid_fine : AbstractPitchAngleGrid
        Pitch angle grids at the coarse and fine na resolutions.
    prefix_size : int
        Product of leading axes that don't change between levels (e.g.
        ``len(species) * speedgrid.nx``). Defaults to 1.
    method : str
        Interpolation method. Passed to ``interpax.interp1d``.
    """

    field_coarse: Field
    field_fine: Field
    pitchgrid_coarse: AbstractPitchAngleGrid
    pitchgrid_fine: AbstractPitchAngleGrid
    prefix_size: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    P_xi: jax.Array
    P_theta: jax.Array
    P_zeta: jax.Array

    def __init__(
        self,
        field_coarse: Field,
        field_fine: Field,
        pitchgrid_coarse: AbstractPitchAngleGrid,
        pitchgrid_fine: AbstractPitchAngleGrid,
        prefix_size: int = 1,
        method: str = "linear",
    ):
        self.field_coarse = field_coarse
        self.field_fine = field_fine
        self.pitchgrid_coarse = pitchgrid_coarse
        self.pitchgrid_fine = pitchgrid_fine
        self.prefix_size = prefix_size
        self.method = method
        self.P_xi = _build_interp_matrix(
            pitchgrid_coarse.xi, pitchgrid_fine.xi, method=method, period=None
        )
        self.P_theta = _build_interp_matrix(
            field_coarse.theta, field_fine.theta, method=method, period=2 * jnp.pi
        )
        self.P_zeta = _build_interp_matrix(
            field_coarse.zeta,
            field_fine.zeta,
            method=method,
            period=2 * jnp.pi / field_coarse.NFP,
        )

    @eqx.filter_jit
    @jax.named_scope("Prolongation.mv")
    def mv(self, vector):
        """Matrix-vector product (coarse -> fine)."""
        nt_c = self.field_coarse.ntheta
        nz_c = self.field_coarse.nzeta
        na_c = self.pitchgrid_coarse.na
        f = vector.reshape((self.prefix_size, na_c, nt_c, nz_c))
        # axes after reshape: (0:prefix, 1:na, 2:nt, 3:nz)
        # Each tensordot brings its axis to 0 and leaves it there; we restore
        # the original axis order at the end.
        f = jnp.moveaxis(f, 1, 0)
        f = jnp.tensordot(self.P_xi, f, axes=1)
        f = jnp.moveaxis(f, 2, 0)
        f = jnp.tensordot(self.P_theta, f, axes=1)
        f = jnp.moveaxis(f, 3, 0)
        f = jnp.tensordot(self.P_zeta, f, axes=1)
        f = jnp.transpose(f, (3, 2, 1, 0))
        return f.flatten()

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        n = (
            self.prefix_size
            * self.pitchgrid_coarse.na
            * self.field_coarse.ntheta
            * self.field_coarse.nzeta
        )
        return jax.ShapeDtypeStruct((n,), dtype=self.field_coarse.Bmag.dtype)

    def out_structure(self):
        """Pytree structure of expected output."""
        n = (
            self.prefix_size
            * self.pitchgrid_fine.na
            * self.field_fine.ntheta
            * self.field_fine.nzeta
        )
        return jax.ShapeDtypeStruct((n,), dtype=self.field_fine.Bmag.dtype)

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class Restriction(lx.AbstractLinearOperator):
    """Fine-to-coarse grid restriction as a linear operator.

    Applies the volume-weighted transpose of piecewise-linear (or other)
    interpolation, i.e. ``R = (N_coarse / N_fine) P^T`` per axis, where ``P``
    is the prolongation matrix between adjacent grids.

    Parameters
    ----------
    field_coarse, field_fine : Field
        Magnetic field data at the coarse and fine theta/zeta resolutions.
    pitchgrid_coarse, pitchgrid_fine : AbstractPitchAngleGrid
        Pitch angle grids at the coarse and fine na resolutions.
    prefix_size : int
        Product of leading axes that don't change between levels (e.g.
        ``len(species) * speedgrid.nx``). Defaults to 1.
    method : str
        Interpolation method that defines the underlying prolongation.
    """

    field_coarse: Field
    field_fine: Field
    pitchgrid_coarse: AbstractPitchAngleGrid
    pitchgrid_fine: AbstractPitchAngleGrid
    prefix_size: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    P_xi: jax.Array
    P_theta: jax.Array
    P_zeta: jax.Array
    volume_scale: jax.Array

    def __init__(
        self,
        field_coarse: Field,
        field_fine: Field,
        pitchgrid_coarse: AbstractPitchAngleGrid,
        pitchgrid_fine: AbstractPitchAngleGrid,
        prefix_size: int = 1,
        method: str = "linear",
    ):
        self.field_coarse = field_coarse
        self.field_fine = field_fine
        self.pitchgrid_coarse = pitchgrid_coarse
        self.pitchgrid_fine = pitchgrid_fine
        self.prefix_size = prefix_size
        self.method = method
        # Store the coarse->fine prolongation matrix; restriction applies its
        # volume-weighted transpose.
        self.P_xi = _build_interp_matrix(
            pitchgrid_coarse.xi, pitchgrid_fine.xi, method=method, period=None
        )
        self.P_theta = _build_interp_matrix(
            field_coarse.theta, field_fine.theta, method=method, period=2 * jnp.pi
        )
        self.P_zeta = _build_interp_matrix(
            field_coarse.zeta,
            field_fine.zeta,
            method=method,
            period=2 * jnp.pi / field_coarse.NFP,
        )
        self.volume_scale = jnp.asarray(
            (pitchgrid_coarse.na / pitchgrid_fine.na)
            * (field_coarse.ntheta / field_fine.ntheta)
            * (field_coarse.nzeta / field_fine.nzeta)
        )

    @eqx.filter_jit
    @jax.named_scope("Restriction.mv")
    def mv(self, vector):
        """Matrix-vector product (fine -> coarse)."""
        nt_f = self.field_fine.ntheta
        nz_f = self.field_fine.nzeta
        na_f = self.pitchgrid_fine.na
        f = vector.reshape((self.prefix_size, na_f, nt_f, nz_f))
        # Apply the volume-weighted transpose of each per-axis prolongation.
        f = jnp.moveaxis(f, 1, 0)
        f = jnp.tensordot(self.P_xi, f, axes=([0], [0]))
        f = jnp.moveaxis(f, 2, 0)
        f = jnp.tensordot(self.P_theta, f, axes=([0], [0]))
        f = jnp.moveaxis(f, 3, 0)
        f = jnp.tensordot(self.P_zeta, f, axes=([0], [0]))
        f = f * self.volume_scale
        f = jnp.transpose(f, (3, 2, 1, 0))
        return f.flatten()

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        n = (
            self.prefix_size
            * self.pitchgrid_fine.na
            * self.field_fine.ntheta
            * self.field_fine.nzeta
        )
        return jax.ShapeDtypeStruct((n,), dtype=self.field_fine.Bmag.dtype)

    def out_structure(self):
        """Pytree structure of expected output."""
        n = (
            self.prefix_size
            * self.pitchgrid_coarse.na
            * self.field_coarse.ntheta
            * self.field_coarse.nzeta
        )
        return jax.ShapeDtypeStruct((n,), dtype=self.field_coarse.Bmag.dtype)

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


@lx.is_symmetric.register(Prolongation)
@lx.is_diagonal.register(Prolongation)
@lx.is_tridiagonal.register(Prolongation)
@lx.is_symmetric.register(Restriction)
@lx.is_diagonal.register(Restriction)
@lx.is_tridiagonal.register(Restriction)
def _(operator):
    return False


def _multigrid_cycle_recursive(
    cycle_index,
    k,
    x,
    operators,
    rhs,
    smoothers,
    prolongations,
    restrictions,
    v1,
    v2,
    smooth_method,
    coarse_opinv,
    coarse_method,
    verbose,
):
    """Apply multigrid cycle for solving operator @ x = rhs

    Parameters
    ----------
    cycle_index : int
        Type of cycle / number of sub-cycles. cycle_index=1 corresponds to a "V" cycle,
        cycle_index=2 is a "W" cycle etc.
    k : int
        Index of starting grid. Generally len(operators) - 1
    x : jax.Array, optional
        Starting guess for solution.
    operators : list[lx.AbstractLinearOperator]
        Operators for each level of discretization, from fine to coarse.
    rhs : jax.Array
        Right hand side vector on finest grid.
    smoothers: list[list[lx.AbstractLinearOperator]]
        Smoothers to apply at each level. Note the smoothing operation is
        smoother @ x, not inv(smoother) @ x.
    prolongations : list[lx.AbstractLinearOperator]
        Coarse->fine prolongation operators between adjacent levels.
        ``prolongations[k]`` maps level ``k`` to level ``k+1``.
    restrictions : list[lx.AbstractLinearOperator]
        Fine->coarse restriction operators between adjacent levels.
        ``restrictions[k]`` maps level ``k+1`` to level ``k``.
    n : int
        Number of cycles to perform.
    v1, v2 : int
        Number of pre- and post- smoothing iterations.
    smooth_method : {"standard", "krylov1", "krylov2", "krylov1s", "krylov2s"}
        Method to use for smoothing.
    coarse_opinv: lx.AbstractLinearOperator
        Precomputed inverse of coarsest grid operator.
    coarse_method : {"standard", "krylov1", "krylov2", "krylov1s", "krylov2s"}
        Method to use for coarse grid corrections.
    verbose : int
        Level of verbosity:
          - 0: no info printed.
          - 1: also print residuals at each multigrid level before and after smoothing.
          - 2: also print residuals within smoothing iterations.

    Returns
    -------
    x : jax.Array
        Updated estimate for solution x.
    """
    Ak = operators[k]
    Mk = smoothers[k]

    smooth = {
        "standard": standard_smooth,
        "adaptive": adpative_smooth,
        "krylov1": krylov1_smooth,
        "krylov1s": krylov1s_smooth,
        "krylov2": krylov2_smooth,
        "krylov2s": krylov2s_smooth,
    }[smooth_method]
    coarse_correction = {
        "standard": standard_coarse_correction,
        "krylov1": krylov1_coarse_correction,
        "krylov1s": krylov1s_coarse_correction,
        "krylov2": krylov2_coarse_correction,
        "krylov2s": krylov2s_coarse_correction,
    }[coarse_method]

    if verbose:
        rk = rhs - Ak.mv(x)
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print(
            "level={k} before presmooth err: {err:.3e}", err=err, k=k, ordered=True
        )

    # Pre-smooth: x is always zero on entry (top-level uses zeros_like(vector);
    # recursive calls pass x=jnp.zeros_like(rkm1)), so r0 = rhs - A.mv(0) = rhs.
    # The smoother returns the up-to-date residual, eliminating a separate mv.
    vv = jnp.where(v1 > 0, v1, len(operators) - k + jnp.abs(v1))
    with jax.named_scope(f"pre-smooth, level={k}"):
        x, rk = smooth(x, Ak, rhs, Mk, nsteps=vv, verbose=max(verbose - 1, 0), r0=rhs)

    if verbose:
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print(
            "level={k} after presmooth err: {err:.3e}", err=err, k=k, ordered=True
        )

    def body(i, state):
        rk, x = state

        with jax.named_scope(f"restriction level={k}"):
            rkm1 = restrictions[k - 1].mv(rk)
        if k == 1:
            with jax.named_scope("coarse grid solve, level=0"):
                ykm1 = coarse_opinv.mv(rkm1)
        else:
            ykm1 = _multigrid_cycle_recursive(
                cycle_index=cycle_index,
                k=k - 1,
                x=jnp.zeros_like(rkm1),
                operators=operators,
                rhs=rkm1,
                smoothers=smoothers,
                prolongations=prolongations,
                restrictions=restrictions,
                v1=v1,
                v2=v2,
                smooth_method=smooth_method,
                coarse_opinv=coarse_opinv,
                coarse_method=coarse_method,
                verbose=verbose,
            )
        with jax.named_scope(f"prolongation level={k}"):
            yk = prolongations[k - 1].mv(ykm1)
        with jax.named_scope(f"coarse_correction level={k}"):
            x = coarse_correction(x, k, i, Ak, yk, rk, verbose=max(verbose - 1, 0))

        if verbose:
            rk = rhs - Ak.mv(x)
            err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "level={k}/{i} before postsmooth err: {err:.3e}",
                err=err,
                k=k,
                i=i,
                ordered=True,
            )

        # Post-smooth: x has been modified by coarse_correction so rk is stale;
        # let the smoother compute its initial residual internally (r0=None).
        # The returned rk is up-to-date, so we don't need a separate mv after.
        vv = jnp.where(v2 > 0, v2, len(operators) - k + jnp.abs(v2))
        with jax.named_scope(f"post-smooth, level={k}"):
            x, rk = smooth(x, Ak, rhs, Mk, nsteps=vv, verbose=max(verbose - 1, 0))
        if verbose:
            err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "level={k}/{i} after postsmooth err: {err:.3e}",
                err=err,
                k=k,
                i=i,
                ordered=True,
            )
        return rk, x

    _, x = jax.lax.fori_loop(0, cycle_index, body, (rk, x))

    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def standard_coarse_correction(x, k, i, operator, yk, rk, verbose):
    """Apply coarse grid correction with standard weighting."""
    alpha = 1.0
    dx = alpha * yk

    if verbose:
        err = jnp.linalg.norm(dx) / jnp.linalg.norm(x)
        jax.debug.print(
            "level={k}/{i} coarse_correction: {err:.3e}, alpha: {alpha}",
            err=err,
            k=k - 1,
            i=i,
            alpha=alpha,
            ordered=True,
        )

    x += dx
    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov1_coarse_correction(x, k, i, operator, yk, rk, verbose):
    """Apply coarse grid correction st coarse grid residual is minimized over yk."""
    Ayk = operator.mv(yk)
    alpha = jnp.linalg.lstsq(Ayk[:, None], rk)[0][0]
    dx = alpha * yk

    if verbose:
        err = jnp.linalg.norm(dx) / jnp.linalg.norm(x)
        jax.debug.print(
            "level={k}/{i} coarse_correction: {err:.3e}, alpha: {alpha}",
            err=err,
            k=k - 1,
            i=i,
            alpha=alpha,
            ordered=True,
        )

    x += dx
    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov1s_coarse_correction(x, k, i, operator, yk, rk, verbose):
    """Apply coarse grid correction st coarse grid residual is minimized
    over yk, Lyk.
    """
    L = DKELaplacian(
        operator.field, operator.pitchgrid, operator.speedgrid, operator.species, True
    )
    Lyk = L.mv(yk)
    dxs = jnp.array([yk, Lyk])
    Adxs = jax.vmap(operator.mv)(dxs)
    alpha = jnp.linalg.lstsq(Adxs.T, rk)[0]
    dx = dxs.T @ alpha

    if verbose:
        err = jnp.linalg.norm(dx) / jnp.linalg.norm(x)
        jax.debug.print(
            "level={k}/{i} coarse_correction: {err:.3e}, alpha: {alpha}",
            err=err,
            k=k - 1,
            i=i,
            alpha=alpha,
            ordered=True,
        )

    x += dx
    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov2_coarse_correction(x, k, i, operator, yk, rk, verbose):
    """Apply coarse grid correction st coarse grid residual is minimized
    over yk, rk.
    """
    dxs = jnp.array([yk, rk])
    Adxs = jax.vmap(operator.mv)(dxs)
    alpha = jnp.linalg.lstsq(Adxs.T, rk)[0]
    dx = dxs.T @ alpha

    if verbose:
        err = jnp.linalg.norm(dx) / jnp.linalg.norm(x)
        jax.debug.print(
            "level={k}/{i} coarse_correction: {err:.3e}, alpha: {alpha}",
            err=err,
            k=k - 1,
            i=i,
            alpha=alpha,
            ordered=True,
        )

    x += dx
    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
@jax.named_call
def krylov2s_coarse_correction(x, k, i, operator, yk, rk, verbose):
    """Apply coarse grid correction st coarse grid residual is minimized
    over yk, rk, Lyk, Lrk.
    """
    L = DKELaplacian(
        operator.field, operator.pitchgrid, operator.speedgrid, operator.species, True
    )
    dxs = jnp.array([yk, rk])
    Ldxs = jax.vmap(L.mv)(dxs)
    dxs = jnp.concatenate([dxs, Ldxs])
    Adxs = jax.vmap(operator.mv)(dxs)
    alpha = jnp.linalg.lstsq(Adxs.T, rk)[0]
    dx = dxs.T @ alpha

    if verbose:
        err = jnp.linalg.norm(dx) / jnp.linalg.norm(x)
        jax.debug.print(
            "level={k}/{i} coarse_correction: {err:.3e}, alpha: {alpha}",
            err=err,
            k=k - 1,
            i=i,
            alpha=alpha,
            ordered=True,
        )

    x += dx
    return x


class MultigridOperator(lx.AbstractLinearOperator):
    """Multigrid cycle as a linear operator.

    Parameters
    ----------
    operators : list[lx.AbstractLinearOperator]
        Operators for each level of discretization, from coarse to fine.
    smoothers: list[list[lx.AbstractLinearOperator]]
        Smoothers to apply at each level. Note the smoothing operation is
        smoother @ x, not inv(smoother) @ x.
    prolongations : list[lx.AbstractLinearOperator]
        Coarse->fine prolongation operators between adjacent levels, length
        ``len(operators) - 1``. ``prolongations[k]`` maps level ``k`` to
        level ``k+1``.
    restrictions : list[lx.AbstractLinearOperator]
        Fine->coarse restriction operators between adjacent levels, length
        ``len(operators) - 1``. ``restrictions[k]`` maps level ``k+1`` to
        level ``k``.
    x0 : jax.Array, optional
        Starting guess for solution. Default is all zero.
    cycle_index : int
        Type of cycle / number of sub-cycles. cycle_index=1 corresponds to a "V" cycle,
        cycle_index=2 is a "W" cycle etc.
    v1, v2 : int
        Number of pre- and post- smoothing iterations.
    smooth_method : {"standard", "krylov1", "krylov2", "krylov1s", "krylov2s"}
        Method to use for smoothing.
    coarse_opinv: lx.AbstractLinearOperator
        Precomputed inverse of coarse grid operator.
    coarse_method : {"standard", "krylov1", "krylov2", "krylov1s", "krylov2s"}
        Method to use for coarse grid corrections.
    verbose : int
        Level of verbosity:
          - 0: no into printed.
          - 1: print residuals at each multigrid level before and after smoothing.
          - 2: also print residuals within smoothing iterations.

    """

    operators: list[lx.AbstractLinearOperator]
    smoothers: list[list[lx.AbstractLinearOperator]]
    prolongations: list[lx.AbstractLinearOperator]
    restrictions: list[lx.AbstractLinearOperator]
    x0: Union[None, jax.Array]
    cycle_index: jax.Array
    v1: jax.Array
    v2: jax.Array
    smooth_method: str = eqx.field(static=True)
    coarse_opinv: lx.AbstractLinearOperator
    coarse_method: str = eqx.field(static=True)
    verbose: int = eqx.field(static=True)

    def __init__(
        self,
        operators: list[lx.AbstractLinearOperator],
        smoothers: list[list[lx.AbstractLinearOperator]],
        prolongations: list[lx.AbstractLinearOperator],
        restrictions: list[lx.AbstractLinearOperator],
        x0: Optional[jax.Array] = None,
        cycle_index: Union[int, Int[Array, ""]] = 1,
        v1: Union[int, Int[Array, ""]] = 1,
        v2: Union[int, Int[Array, ""]] = 1,
        smooth_method: str = "standard",
        coarse_opinv: Optional[lx.AbstractLinearOperator] = None,
        coarse_method: str = "standard",
        verbose: Union[bool, int] = False,
    ):
        assert len(prolongations) == len(operators) - 1
        assert len(restrictions) == len(operators) - 1

        self.operators = operators
        self.smoothers = smoothers
        self.prolongations = prolongations
        self.restrictions = restrictions
        self.x0 = x0
        self.cycle_index = jnp.asarray(cycle_index)
        self.v1 = jnp.asarray(v1)
        self.v2 = jnp.asarray(v2)
        self.smooth_method = smooth_method
        if coarse_opinv is None:
            coarse_opinv = InverseLinearOperator(operators[0], lx.LU(), throw=False)
        self.coarse_opinv = coarse_opinv
        self.coarse_method = coarse_method
        self.verbose = verbose

    @eqx.filter_jit
    @jax.named_scope("MultigridOperator.mv")
    def mv(self, vector):
        """Matrix vector product."""
        x0 = jnp.zeros_like(vector)
        x = _multigrid_cycle_recursive(
            cycle_index=self.cycle_index,
            k=len(self.operators) - 1,
            x=x0,
            operators=self.operators,
            rhs=vector,
            smoothers=self.smoothers,
            prolongations=self.prolongations,
            restrictions=self.restrictions,
            v1=self.v1,
            v2=self.v2,
            smooth_method=self.smooth_method,
            coarse_opinv=self.coarse_opinv,
            coarse_method=self.coarse_method,
            verbose=self.verbose,
        )
        return x

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return self.operators[-1].in_structure()

    def out_structure(self):
        """Pytree structure of expected output."""
        return self.operators[-1].out_structure()

    def transpose(self):
        """Transpose of the operator."""
        opt = [op.transpose() for op in self.operators]
        smt = [[sm.transpose() for sm in smo] for smo in self.smoothers]
        # In the transposed cycle, the coarse->fine direction is the transpose
        # of the original fine->coarse direction, and vice versa.
        prot = [r.transpose() for r in self.restrictions]
        rest = [p.transpose() for p in self.prolongations]
        opit = self.coarse_opinv.transpose()
        return MultigridOperator(
            opt,
            smt,
            prot,
            rest,
            self.x0,
            self.cycle_index,
            self.v1,
            self.v2,
            self.smooth_method,
            opit,
            self.coarse_method,
            self.verbose,
        )


@lx.is_symmetric.register(MultigridOperator)
@lx.is_diagonal.register(MultigridOperator)
@lx.is_tridiagonal.register(MultigridOperator)
def _(operator):
    return False
