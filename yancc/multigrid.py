"""Stuff for multigrid cycles."""

import functools
from typing import Optional, Union

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from .linalg import InverseLinearOperator
from .smoothers import DKEJacobi2Smoother, DKEJacobiSmoother, MDKEJacobiSmoother
from .trajectories import DKE, MDKE


@functools.partial(jax.jit, static_argnames=["p1", "p2"])
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
def get_dke_operators(
    fields,
    pitchgrids,
    speedgrid,
    species,
    Erho,
    potentials,
    p1,
    p2,
    gauge,
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
            potentials,
            p1=p1,
            p2=p2,
            gauge=gauge,
            **options,
        )
        operators.append(op)
    return operators


@eqx.filter_jit
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
def get_dke_jacobi_smoothers(
    fields,
    pitchgrids,
    speedgrid,
    species,
    Erho,
    potentials,
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
            DKEJacobiSmoother(
                field,
                pitchgrid,
                speedgrid,
                species,
                Erho,
                potentials,
                p1=p1,
                p2=p2,
                axorder=order,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=weight,
                **options,
            )
            for order in ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"]
        ]
        smoothers.append(smooth)
    return smoothers


@eqx.filter_jit
def get_dke_jacobi2_smoothers(
    fields,
    pitchgrids,
    speedgrid,
    species,
    Erho,
    potentials,
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
            DKEJacobi2Smoother(
                field,
                pitchgrid,
                speedgrid,
                species,
                Erho,
                potentials,
                p1=p1,
                p2=p2,
                axorder=order,
                gauge=gauge,
                smooth_solver=smooth_solver,
                weight=weight,
                **options,
            )
            for order in ["atzsx", "tzasx", "zatsx"]
        ]
        smoothers.append(smooth)
    return smoothers


def _half_next_even(k: int, m: Union[int, float] = 2):
    if int(k // m) == 0:
        return 2
    elif int(k // m) % 2 == 0:
        return int(k // m)
    else:
        return int(k // m + 1)


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
    ress : list of tuple of int
        Each list element is a tuple of resolutions at a given grid level, each
        tuple is the resolution (ns, nx, na, nt, nz)
    """
    coarse_N = max(coarse_N, ns * nx * min_na * min_nt * min_nz)
    N = ns * nx * na * nt * nz

    if coarsening_factor is not None and max_grids is not None:
        raise ValueError("Cannot specify both coarsening_factor and max_grids")
    elif coarsening_factor is None and max_grids is None:
        coarsening_factor = 2
        max_grids = int(
            np.ceil(np.log(N / coarse_N) / np.log(coarsening_factor**3) + 1)
        )
    elif coarsening_factor is None:
        assert isinstance(max_grids, int)
        coarsening_factor = float((N / coarse_N) ** (1 / (3 * (max_grids - 1))))
        coarsening_factor = max(2, coarsening_factor)
    elif max_grids is None:
        max_grids = int(
            np.ceil(np.log(N / coarse_N) / np.log(coarsening_factor**3) + 1)
        )

    ress = [(ns, nx, na, nt, nz)]
    na = max(_half_next_odd(na, coarsening_factor), min_na)
    nt = max(_half_next_odd(nt, coarsening_factor), min_nt)
    nz = max(_half_next_odd(nz, coarsening_factor), min_nz)
    N = ns * nx * na * nt * nz
    while N > coarse_N and len(ress) < max_grids - 1:
        ress.append((ns, nx, na, nt, nz))
        na = max(_half_next_odd(na, coarsening_factor), min_na)
        nt = max(_half_next_odd(nt, coarsening_factor), min_nt)
        nz = max(_half_next_odd(nz, coarsening_factor), min_nz)
        N = ns * nx * na * nt * nz
    ress.append((ns, nx, na, nt, nz))
    return ress[::-1]


@eqx.filter_jit
def get_fields_grids(
    field,
    pitchgrid,
    ress,
):
    """Get fields and grids for multigrid problem.

    Parameters
    ----------
    field : Field
        Field at sufficient resolution to represent B.
    pitchgrid : PitchAngleGrid
        Pitch angle grid data.
    ress : array-like, shape(num_grid, 5)
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
    for res in ress:
        _, _, na, nt, nz = res
        fields.append(field.resample(nt, nz))
        grids.append(pitchgrid.resample(na))
    return fields, grids


@functools.partial(jax.jit, static_argnames=["verbose"])
def standard_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False):
    """Apply smoothing operators to operator @ x = rhs"""
    if not isinstance(smoothers, (tuple, list)):
        smoothers = [smoothers]

    def body(k, x):
        for i, Mi in enumerate(smoothers):
            Ax = operator.mv(x)
            r = rhs - Ax
            dx = Mi.mv(r)
            x += dx
            if verbose:
                r = rhs - operator.mv(x)
                err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
                jax.debug.print(
                    "v={k} after {a} err: {err:.3e}",
                    err=err,
                    k=k,
                    a=Mi.axorder,
                    ordered=True,
                )
        return x

    x = jax.lax.fori_loop(0, nsteps, body, x)
    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
def krylov_smooth(x, operator, rhs, smoothers, nsteps=1, verbose=False):
    """Apply smoothing operators to operator @ x = rhs"""
    if not isinstance(smoothers, (tuple, list)):
        smoothers = [smoothers]

    def body(k, x0):
        rs = jnp.empty((len(smoothers) + 1, rhs.size))
        dxs = jnp.empty((len(smoothers), rhs.size))
        x = x0

        for i, Mi in enumerate(smoothers):
            Ax = operator.mv(x)
            r = rhs - Ax
            rs = rs.at[i].set(r)
            dx = Mi.mv(r)
            dxs = dxs.at[i].set(dx)
            x += dx
            if verbose:
                r = rhs - operator.mv(x)
                err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
                jax.debug.print(
                    "v={k} after {a} err: {err:.3e}",
                    err=err,
                    k=k,
                    a=Mi.axorder,
                    ordered=True,
                )

        Ax = operator.mv(x)
        r = rhs - Ax
        rs = rs.at[-1].set(r)

        rb = rs[0]
        dr = -jnp.diff(rs, axis=0)

        alpha = jnp.linalg.lstsq(dr.T, rb)[0]
        x = x0 + dxs.T @ alpha

        if verbose:
            Ax = operator.mv(x)
            r = rhs - Ax
            err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "v={k} err: {err:.3e} alpha: {alpha}",
                err=err,
                k=k,
                alpha=alpha,
                ordered=True,
            )

        return x

    x = jax.lax.fori_loop(0, nsteps, body, x)
    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
def krylov_smooth2(x, operator, rhs, smoothers, nsteps=1, verbose=False):
    """Apply smoothing operators to operator @ x = rhs"""
    if not isinstance(smoothers, (tuple, list)):
        smoothers = [smoothers]

    def body(k, x0):
        rs = jnp.empty((len(smoothers) + 1, rhs.size))
        dxs = jnp.empty((len(smoothers), rhs.size))
        x = x0
        Ax = operator.mv(x)
        r = rhs - Ax

        for i, Mi in enumerate(smoothers):
            rs = rs.at[i].set(r)
            dx = Mi.mv(r)
            dxs = dxs.at[i].set(dx)
            Adx = operator.mv(dx)
            c = jnp.linalg.lstsq(Adx[:, None], r)[0][0]
            x += c * dx
            r -= c * Adx
            if verbose:
                err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
                jax.debug.print(
                    "v={k} after {a} err: {err:.3e} c: {c:.3e}",
                    err=err,
                    k=k,
                    c=c,
                    a=Mi.axorder[-1],
                    ordered=True,
                )

        Ax = operator.mv(x)
        r = rhs - Ax
        rs = rs.at[-1].set(r)

        rb = rs[0]
        dr = -jnp.diff(rs, axis=0)

        alpha = jnp.linalg.lstsq(dr.T, rb)[0]
        x = x0 + dxs.T @ alpha

        if verbose:
            Ax = operator.mv(x)
            r = rhs - Ax
            err = jnp.linalg.norm(r) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "v={k} err: {err:.3e} alpha: {alpha}",
                err=err,
                k=k,
                alpha=alpha,
                ordered=True,
            )

        return x

    x = jax.lax.fori_loop(0, nsteps, body, x)
    return x


def _prolongate_a(f, a1, a2, method="linear"):
    assert len(a2) >= len(a1)
    f2 = jnp.moveaxis(
        interpax.interp1d(
            a2, a1, jnp.moveaxis(f, -3, 0), method=method, period=None, extrap=True
        ),
        0,
        -3,
    )
    return f2


def _prolongate_t(f, t1, t2, method="linear"):
    assert len(t2) >= len(t1)
    f2 = jnp.moveaxis(
        interpax.interp1d(
            t2,
            t1,
            jnp.moveaxis(f, -2, 0),
            method=method,
            period=2 * jnp.pi,
            extrap=True,
        ),
        0,
        -2,
    )
    return f2


def _prolongate_z(f, z1, z2, NFP=1, method="linear"):
    assert len(z2) >= len(z1)
    f2 = jnp.moveaxis(
        interpax.interp1d(
            z2,
            z1,
            jnp.moveaxis(f, -1, 0),
            method=method,
            period=2 * jnp.pi / NFP,
            extrap=True,
        ),
        0,
        -1,
    )
    return f2


def _restrict_a(f, a1, a2, method="linear"):
    interp = lambda f: jnp.moveaxis(
        interpax.interp1d(
            a1, a2, jnp.moveaxis(f, -3, 0), method=method, period=None, extrap=True
        ),
        0,
        -3,
    )
    shp = list(f.shape)
    shp[-3] = len(a2)
    g = jnp.zeros(shp)
    f2 = jax.linear_transpose(interp, g)(f)[0]
    return f2 * len(a2) / len(a1)


def _restrict_t(f, t1, t2, method="linear"):
    interp = lambda f: jnp.moveaxis(
        interpax.interp1d(
            t1,
            t2,
            jnp.moveaxis(f, -2, 0),
            method=method,
            period=2 * jnp.pi,
            extrap=True,
        ),
        0,
        -2,
    )
    shp = list(f.shape)
    shp[-2] = len(t2)
    g = jnp.zeros(shp)
    f2 = jax.linear_transpose(interp, g)(f)[0]
    return f2 * len(t2) / len(t1)


def _restrict_z(f, z1, z2, NFP=1, method="linear"):
    interp = lambda f: jnp.moveaxis(
        interpax.interp1d(
            z1,
            z2,
            jnp.moveaxis(f, -1, 0),
            method=method,
            period=2 * jnp.pi / NFP,
            extrap=True,
        ),
        0,
        -1,
    )
    shp = list(f.shape)
    shp[-1] = len(z2)
    g = jnp.zeros(shp)
    f2 = jax.linear_transpose(interp, g)(f)[0]
    return f2 * len(z2) / len(z1)


@eqx.filter_jit
def interpolate(f, field1, field2, pitchgrid1, pitchgrid2, method="linear"):
    """Prolongation/restriction between grids via (transposed) interpolation."""
    nt1, nz1, na1 = field1.ntheta, field1.nzeta, pitchgrid1.nxi
    nt2, nz2, na2 = field2.ntheta, field2.nzeta, pitchgrid2.nxi
    t1, t2 = field1.theta, field2.theta
    z1, z2 = field1.zeta, field2.zeta
    a1, a2 = pitchgrid1.xi, pitchgrid2.xi

    N1 = nt1 * nz1 * na1
    nx = f.size // N1
    f = f.reshape((nx, na1, nt1, nz1))

    if na2 >= na1:
        f = _prolongate_a(f, a1, a2, method)
    else:
        f = _restrict_a(f, a1, a2, method)

    if nt2 >= nt1:
        f = _prolongate_t(f, t1, t2, method)
    else:
        f = _restrict_t(f, t1, t2, method)

    if nz2 >= nz1:
        f = _prolongate_z(f, z1, z2, field1.NFP, method)
    else:
        f = _restrict_z(f, z1, z2, field1.NFP, method)

    return f.flatten()


def _multigrid_cycle_recursive(
    cycle_index,
    k,
    x,
    operators,
    rhs,
    smoothers,
    v1,
    v2,
    interp_method,
    smooth_method,
    coarse_opinv,
    coarse_overweight,
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
    n : int
        Number of cycles to perform.
    v1, v2 : int
        Number of pre- and post- smoothing iterations.
    interp_method : str
        Method of interpolation, passed to interpax.interp3d
    smooth_method : {"standard", "krylov"}
        Method to use for smoothing.
    coarse_opinv: lx.AbstractLinearOperator
        Precomputed inverse of coarse grid operator.
    coarse_overweight : {"auto1", "auto2"} or float
        Factor to weight coarse grid residuals by, to improve coarse grid correction
        for closed characteristics. If str, use an automatically determined value to
        ensure strict decrease of the coarse grid residuals.
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

    assert smooth_method in {"standard", "krylov", "krylov2"}
    smooth = {
        "standard": standard_smooth,
        "krylov": krylov_smooth,
        "krylov2": krylov_smooth2,
    }[smooth_method]

    if verbose:
        rk = rhs - Ak.mv(x)
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print(
            "level={k} before presmooth err: {err:.3e}", err=err, k=k, ordered=True
        )

    vv = jnp.where(v1 > 0, v1, len(operators) - k + jnp.abs(v1))
    x = smooth(x, Ak, rhs, Mk, nsteps=vv, verbose=max(verbose - 1, 0))
    rk = rhs - Ak.mv(x)

    if verbose:
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print(
            "level={k} after presmooth err: {err:.3e}", err=err, k=k, ordered=True
        )

    def body(i, state):
        rk, x, idx = state

        rkm1 = interpolate(
            rk,
            operators[k].field,
            operators[k - 1].field,
            operators[k].pitchgrid,
            operators[k - 1].pitchgrid,
            interp_method,
        )
        if k == 1:
            ykm1 = coarse_opinv.mv(rkm1)
        else:
            ykm1 = _multigrid_cycle_recursive(
                cycle_index=idx,
                k=k - 1,
                x=jnp.zeros_like(rkm1),
                operators=operators,
                rhs=rkm1,
                smoothers=smoothers,
                v1=v1,
                v2=v2,
                interp_method=interp_method,
                smooth_method=smooth_method,
                coarse_opinv=coarse_opinv,
                coarse_overweight=coarse_overweight,
                verbose=verbose,
            )
        yk = interpolate(
            ykm1,
            operators[k - 1].field,
            operators[k].field,
            operators[k - 1].pitchgrid,
            operators[k].pitchgrid,
            interp_method,
        )
        if isinstance(coarse_overweight, str):
            if coarse_overweight == "auto1":
                Aykm1 = operators[k - 1].mv(ykm1)
                alpha = jnp.linalg.lstsq(Aykm1[:, None], rkm1)[0][0]
            elif coarse_overweight == "auto2":
                Ayk = operators[k].mv(yk)
                alpha = jnp.linalg.lstsq(Ayk[:, None], rk)[0][0]
            else:
                raise ValueError
        else:
            alpha = coarse_overweight

        if verbose:
            err = jnp.linalg.norm(alpha * yk) / jnp.linalg.norm(x)
            jax.debug.print(
                "level={k}/{i} coarse_correction: {err:.3e}, alpha: {alpha:.3e}",
                err=err,
                k=k - 1,
                i=i,
                alpha=alpha,
                ordered=True,
            )

        x += alpha * yk

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

        vv = jnp.where(v2 > 0, v2, len(operators) - k + jnp.abs(v2))
        x = smooth(x, Ak, rhs, Mk, nsteps=vv, verbose=max(verbose - 1, 0))
        rk = rhs - Ak.mv(x)
        if verbose:
            err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
            jax.debug.print(
                "level={k}/{i} after postsmooth err: {err:.3e}",
                err=err,
                k=k,
                i=i,
                ordered=True,
            )
        return rk, x, idx

    def normal_cycle(rk, x, idx):
        _, x, _ = jax.lax.fori_loop(0, cycle_index, body, (rk, x, idx))
        return x

    def f_cycle(rk, x, idx):
        rk, x, idx = body(0, (rk, x, idx))
        rk, x, idx = body(0, (rk, x, 1))
        return x

    x = jax.lax.cond(cycle_index == 0, f_cycle, normal_cycle, rk, x, cycle_index)

    return x


class MultigridOperator(lx.AbstractLinearOperator):
    """Multigrid cycle as a linear operator.

    Parameters
    ----------
    operators : list[lx.AbstractLinearOperator]
        Operators for each level of discretization, from coarse to fine.
    rhs : jax.Array
        Right hand side vector on finest grid.
    smoothers: list[list[lx.AbstractLinearOperator]]
        Smoothers to apply at each level. Note the smoothing operation is
        smoother @ x, not inv(smoother) @ x.
    x0 : jax.Array, optional
        Starting guess for solution. Default is all zero.
    n : int
        Number of cycles to perform.
    cycle_index : int
        Type of cycle / number of sub-cycles. cycle_index=1 corresponds to a "V" cycle,
        cycle_index=2 is a "W" cycle etc.
    v1, v2 : int
        Number of pre- and post- smoothing iterations.
    interp_method : str
        Method of interpolation, passed to interpax.interp3d
    smooth_method : {"standard", "krylov"}
        Method to use for smoothing.
    coarse_opinv: lx.AbstractLinearOperator
        Precomputed inverse of coarse grid operator.
    coarse_overweight : float
        Factor to weight coarse grid residuals by, to improve coarse grid correction
        for closed characteristics.
    verbose : int
        Level of verbosity:
          - 0: no into printed.
          - 1: print residuals at each multigrid level before and after smoothing.
          - 2: also print residuals within smoothing iterations.

    """

    operators: list[lx.AbstractLinearOperator]
    smoothers: list[list[lx.AbstractLinearOperator]]
    x0: Union[None, jax.Array]
    cycle_index: jax.Array
    v1: jax.Array
    v2: jax.Array
    interp_method: str = eqx.field(static=True)
    smooth_method: str = eqx.field(static=True)
    coarse_opinv: lx.AbstractLinearOperator
    coarse_overweight: Union[str, jax.Array]
    verbose: int = eqx.field(static=True)

    def __init__(
        self,
        operators: list[lx.AbstractLinearOperator],
        smoothers: list[list[lx.AbstractLinearOperator]],
        x0: Optional[jax.Array] = None,
        cycle_index: int = 1,
        v1: int = 1,
        v2: int = 1,
        interp_method: str = "linear",
        smooth_method: str = "standard",
        coarse_opinv: Optional[lx.AbstractLinearOperator] = None,
        coarse_overweight: float = 1.0,
        verbose: Union[bool, int] = False,
    ):

        self.operators = operators
        self.smoothers = smoothers
        self.x0 = x0
        self.cycle_index = jnp.asarray(cycle_index)
        self.v1 = jnp.asarray(v1)
        self.v2 = jnp.asarray(v2)
        self.interp_method = interp_method
        self.smooth_method = smooth_method
        if coarse_opinv is None:
            coarse_opinv = InverseLinearOperator(operators[0], lx.LU(), throw=False)
        self.coarse_opinv = coarse_opinv
        if not isinstance(coarse_overweight, str):
            self.coarse_overweight = jnp.asarray(coarse_overweight)
        else:
            self.coarse_overweight = coarse_overweight
        self.verbose = verbose

    @eqx.filter_jit
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
            v1=self.v1,
            v2=self.v2,
            interp_method=self.interp_method,
            smooth_method=self.smooth_method,
            coarse_opinv=self.coarse_opinv,
            coarse_overweight=self.coarse_overweight,
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
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(MultigridOperator)
@lx.is_diagonal.register(MultigridOperator)
@lx.is_tridiagonal.register(MultigridOperator)
def _(operator):
    return False
