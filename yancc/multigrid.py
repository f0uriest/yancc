"""Stuff for multigrid cycles."""

import functools
from typing import Union

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import lineax as lx

from .linalg import InverseLinearOperator
from .smoothers import MDKEJacobiSmoother, optimal_smoothing_parameter
from .trajectories import MDKE
from .velocity_grids import UniformPitchAngleGrid


def high_freq_err_l2(r, operator):
    """Compute the high frequency component of a residual."""
    shape = (operator.pitchgrid.nxi, operator.field.ntheta, operator.field.nzeta)
    r = r.reshape(shape)
    r = jnp.fft.fftn(r, norm="ortho")
    nxi, nt, nz = r.shape
    ka = jnp.fft.fftfreq(nxi, 1 / nxi)
    kt = jnp.fft.fftfreq(nt, 1 / nt)
    kz = jnp.fft.fftfreq(nz, 1 / nz)
    ka, kt, kz = jnp.meshgrid(ka, kt, kz, indexing="ij")
    low_freq = (
        (jnp.abs(ka) <= jnp.abs(ka).max() // 2)
        & (jnp.abs(kt) <= jnp.abs(kt).max() // 2)
        & (jnp.abs(kz) <= jnp.abs(kz).max() // 2)
    )
    mask = ~low_freq
    return jnp.linalg.norm((r * mask).flatten())


@functools.partial(jax.jit, static_argnames=["p1", "p2"])
def get_operators(fields, pitchgrids, E_psi, nu, p1, p2, gauge):
    """Get multigrid operators for each field, pitchgrid."""
    operators = []
    for field, pitchgrid in zip(fields, pitchgrids):
        op = MDKE(field, pitchgrid, E_psi, nu, p1=p1, p2=p2, gauge=gauge)
        operators.append(op)
    return operators


@functools.partial(jax.jit, static_argnames=["p1", "p2", "smooth_solver"])
def get_jacobi_smoothers(fields, pitchgrids, E_psi, nu, p1, p2, gauge, smooth_solver):
    """Get multigrid smoothers for each field, pitchgrid."""
    smoothers = []
    for field, pitchgrid in zip(fields, pitchgrids):
        smooth = [
            MDKEJacobiSmoother(
                field,
                pitchgrid,
                E_psi,
                nu,
                axorder=order,
                p1=p1,
                p2=p2,
                gauge=gauge,
                smooth_solver=smooth_solver,
            )
            for order in ["atz", "zat", "tza"]
        ]
        smoothers.append(smooth)
    return smoothers


def _half_next_even(k, m=2):
    if int(k // m) == 0:
        return 2
    elif int(k // m) % 2 == 0:
        return int(k // m)
    else:
        return int(k // m + 1)


def _half_next_odd(k, m=2):
    if int(k // m) == 0:
        return 1
    elif int(k // m) % 2 == 0:
        return int(k // m + 1)
    else:
        return int(k // m)


@eqx.filter_jit
def get_fields_grids(
    field, nt, nz, nx, coarsening_factor=2, min_N=1000, min_nt=7, min_nz=7, min_nx=7
):
    """Get fields and grids for multigrid problem.

    Parameters
    ----------
    field : Field
        Field at sufficient resolution to represent B.
    nt, nz, nx : int
        Desired resolution of finest grid in theta, zeta, xi.
    nlevels : int, optional
        Number of levels. Defaults to log2(max(nt,nz,nx))

    Returns
    -------
    fields : list[Field]
        fields at each resolution level
    pitchgrids : list[PitchAngleGrid]
        grids at each resolution level

    """
    min_N = max(min_N, min_nt * min_nz * min_nx)
    fields = []
    grids = []
    fields.append(field.resample(nt, nz))
    grids.append(UniformPitchAngleGrid(nx))
    nt = max(_half_next_odd(nt, coarsening_factor), min_nt)
    nz = max(_half_next_odd(nz, coarsening_factor), min_nz)
    nx = max(_half_next_odd(nx, coarsening_factor), min_nx)
    N = nt * nz * nx
    while N > min_N:
        fields.append(field.resample(nt, nz))
        grids.append(UniformPitchAngleGrid(nx))
        nt = max(_half_next_odd(nt, coarsening_factor), min_nt)
        nz = max(_half_next_odd(nz, coarsening_factor), min_nz)
        nx = max(_half_next_odd(nx, coarsening_factor), min_nx)
        N = nt * nz * nx
    fields.append(field.resample(nt, nz))
    grids.append(UniformPitchAngleGrid(nx))
    return fields, grids


@functools.partial(jax.jit, static_argnames=["verbose"])
def standard_smooth(x, operator, rhs, smoothers, nsteps=1, weights=None, verbose=False):
    """Apply smoothing operators to operator @ x = rhs"""
    if not isinstance(smoothers, (tuple, list)):
        smoothers = [smoothers]
    if weights is None:
        weights = [
            optimal_smoothing_parameter(
                operator.p1, operator.p2, operator.nu, s.axorder
            )
            for s in smoothers
        ]
    weights = jnp.atleast_1d(jnp.array(weights))
    weights = jnp.broadcast_to(weights, (len(smoothers),))

    def body(k, x):
        for Mi, wi in zip(smoothers, weights):
            Ax = operator.mv(x)
            r = rhs - Ax
            dx = wi * Mi.mv(r)
            x += dx
            if verbose:
                Sx = x - dx - wi * Mi.mv(Ax)
                err = high_freq_err_l2(Sx, operator) / high_freq_err_l2(x, operator)
                jax.debug.print("high freq l2 err: {err:.3e}", err=err)
        return x

    x = jax.lax.fori_loop(0, nsteps, body, x)
    return x


@functools.partial(jax.jit, static_argnames=["verbose"])
def krylov_smooth(x, operator, rhs, smoothers, nsteps=1, weights=None, verbose=False):
    """Apply smoothing operators to operator @ x = rhs"""
    if not isinstance(smoothers, (tuple, list)):
        smoothers = [smoothers]
    if weights is None:
        weights = [
            optimal_smoothing_parameter(
                operator.p1, operator.p2, operator.nu, s.axorder
            )
            for s in smoothers
        ]
    weights = jnp.atleast_1d(jnp.array(weights))
    weights = jnp.broadcast_to(weights, (len(smoothers),))

    def body(k, x0):
        rs = jnp.empty((4, rhs.size))
        dxs = jnp.empty((3, rhs.size))
        x = x0

        for i, (Mi, wi) in enumerate(zip(smoothers, weights)):
            Ax = operator.mv(x)
            r = rhs - Ax
            rs = rs.at[i].set(r)
            dx = wi * Mi.mv(r)
            dxs = dxs.at[i].set(dx)
            x += dx
            if verbose:
                Sx = x - dx - wi * Mi.mv(Ax)
                err = high_freq_err_l2(Sx, operator) / high_freq_err_l2(x, operator)
                jax.debug.print("high freq l2 err: {err:.3e}", err=err)

        Ax = operator.mv(x)
        r = rhs - Ax
        rs = rs.at[-1].set(r)

        rb = rs[0]
        dr = -jnp.diff(rs, axis=0)

        alpha = jnp.linalg.lstsq(dr.T, rb)[0]
        return x0 + dxs.T @ alpha

    x = jax.lax.fori_loop(0, nsteps, body, x)
    return x


@eqx.filter_jit
def interpolate(f, field1, field2, pitchgrid1, pitchgrid2, method="linear"):
    """Prolongation/restriction between grids via (transposed) interpolation."""
    nt1, nz1, nx1 = field1.ntheta, field1.nzeta, pitchgrid1.nxi
    nt2, nz2, nx2 = field2.ntheta, field2.nzeta, pitchgrid2.nxi
    t1, t2 = field1.theta, field2.theta
    z1, z2 = field1.zeta, field2.zeta
    x1, x2 = pitchgrid1.xi, pitchgrid2.xi

    N1 = nt1 * nz1 * nx1
    N2 = nt2 * nz2 * nx2

    if N2 > N1:
        xq, tq, zq = jnp.meshgrid(x2, t2, z2, indexing="ij")
        xq = xq.flatten()
        tq = tq.flatten()
        zq = zq.flatten()
        f = f.reshape((nx1, nt1, nz1))
        interp = lambda g: interpax.interp3d(
            xq,
            tq,
            zq,
            x1,
            t1,
            z1,
            g,
            method,
            extrap=True,
            period=(None, 2 * jnp.pi, 2 * jnp.pi / field1.NFP),
        )
        f2 = interp(f)
        return f2
    else:
        xq, tq, zq = jnp.meshgrid(x1, t1, z1, indexing="ij")
        xq = xq.flatten()
        tq = tq.flatten()
        zq = zq.flatten()
        interp = lambda g: interpax.interp3d(
            xq,
            tq,
            zq,
            x2,
            t2,
            z2,
            g.reshape((nx2, nt2, nz2)),
            method,
            extrap=True,
            period=(None, 2 * jnp.pi, 2 * jnp.pi / field1.NFP),
        )
        g = jnp.zeros(N2)
        f2 = jax.linear_transpose(interp, g)(f)[0]
        return f2 * N2 / N1


@functools.partial(
    jax.jit, static_argnames=["interp_method", "smooth_method", "verbose"]
)
def multigrid_cycle(
    operators,
    rhs,
    smoothers,
    x0=None,
    n=1,
    cycle_index=1,
    v1=1,
    v2=1,
    smooth_weights=None,
    interp_method="linear",
    smooth_method="standard",
    coarse_opinv=None,
    coarse_overweight=1.0,
    verbose=False,
):
    """Apply multigrid cycle for solving operator @ x = rhs

    Parameters
    ----------
    operators : list[lx.AbstractLinearOperator]
        Operators for each level of discretization, from fine to coarse.
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
    smooth_weights : jax.Array
        Damping factors for smoothing.
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
          - 1: print residuals after each cycle.
          - 2: also print residuals at each multigrid level before and after smoothing.
          - 3: also print residuals within smoothing iterations.

    """
    kk = len(operators) - 1
    if x0 is None:
        x0 = jnp.zeros_like(rhs)
    if coarse_opinv is None:
        coarse_opinv = InverseLinearOperator(operators[0], lx.LU(), throw=False)

    def body(i, x):
        x = _multigrid_cycle_recursive(
            cycle_index=cycle_index,
            k=kk,
            x=x,
            operators=operators,
            rhs=rhs,
            smoothers=smoothers,
            v1=v1,
            v2=v2,
            smooth_weights=smooth_weights,
            interp_method=interp_method,
            smooth_method=smooth_method,
            coarse_opinv=coarse_opinv,
            coarse_overweight=coarse_overweight,
            verbose=max(verbose - 1, 0),
        )
        if verbose:
            err = jnp.linalg.norm(rhs - operators[-1].mv(x)) / jnp.linalg.norm(rhs)
            jax.debug.print("(iter={i}) err: {err:.3e}", err=err, i=i)
        return x

    x = jax.lax.fori_loop(0, n, body, x0)
    return x


def _multigrid_cycle_recursive(
    cycle_index,
    k,
    x,
    operators,
    rhs,
    smoothers,
    v1,
    v2,
    smooth_weights,
    interp_method,
    smooth_method,
    coarse_opinv,
    coarse_overweight,
    verbose,
):
    Ak = operators[k]
    Mk = smoothers[k]

    assert smooth_method in {"standard", "krylov"}
    smooth = {"standard": standard_smooth, "krylov": krylov_smooth}[smooth_method]

    if verbose:
        rk = rhs - Ak.mv(x)
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print("level=({k}) before presmooth err: {err:.3e}", err=err, k=k)

    vv = jnp.where(v1 > 0, v1, len(operators) - k + jnp.abs(v1))
    x = smooth(
        x, Ak, rhs, Mk, nsteps=vv, weights=smooth_weights, verbose=max(verbose - 1, 0)
    )
    rk = rhs - Ak.mv(x)

    if verbose:
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print("(level={k}) after presmooth err: {err:.3e}", err=err, k=k)
    rkm1 = coarse_overweight * interpolate(
        rk,
        operators[k].field,
        operators[k - 1].field,
        operators[k].pitchgrid,
        operators[k - 1].pitchgrid,
        interp_method,
    )
    if k == 1:
        ykm1 = coarse_opinv.mv(rkm1)
        if verbose:
            err = jnp.linalg.norm(ykm1)
            jax.debug.print("(level=0) coarse_correction: {err:.3e}", err=err)
    else:
        ykm1 = jnp.zeros_like(rkm1)
        body = lambda _, yidx: (
            yidx[0],
            _multigrid_cycle_recursive(
                cycle_index=yidx[0],
                k=k - 1,
                x=yidx[1],
                operators=operators,
                rhs=rkm1,
                smoothers=smoothers,
                v1=v1,
                v2=v2,
                smooth_weights=smooth_weights,
                interp_method=interp_method,
                smooth_method=smooth_method,
                coarse_opinv=coarse_opinv,
                coarse_overweight=coarse_overweight,
                verbose=verbose,
            ),
        )

        def normal_cycle(y):
            idx, y = jax.lax.fori_loop(0, cycle_index, body, (cycle_index, y))
            return y

        def f_cycle(y):
            idx, y = body(0, (cycle_index, y))
            idx, y = body(0, (1, y))
            return y

        ykm1 = jax.lax.cond(cycle_index == 0, f_cycle, normal_cycle, rkm1)

    yk = interpolate(
        ykm1,
        operators[k - 1].field,
        operators[k].field,
        operators[k - 1].pitchgrid,
        operators[k].pitchgrid,
        interp_method,
    )
    x += yk
    if verbose:
        rk = rhs - Ak.mv(x)
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print("level=({k}) before postsmooth err: {err:.3e}", err=err, k=k)

    vv = jnp.where(v2 > 0, v2, len(operators) - k + jnp.abs(v2))
    x = smooth(
        x, Ak, rhs, Mk, nsteps=vv, weights=smooth_weights, verbose=max(verbose - 1, 0)
    )
    if verbose:
        rk = rhs - Ak.mv(x)
        err = jnp.linalg.norm(rk) / jnp.linalg.norm(rhs)
        jax.debug.print("level=({k}) after postsmooth err: {err:.3e}", err=err, k=k)

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
    smooth_weights : jax.Array
        Damping factors for smoothing.
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
          - 1: print residuals after each cycle.
          - 2: also print residuals at each multigrid level before and after smoothing.
          - 3: also print residuals within smoothing iterations.

    """

    operators: list[lx.AbstractLinearOperator]
    smoothers: list[list[lx.AbstractLinearOperator]]
    x0: Union[None, jax.Array]
    cycle_index: jax.Array
    v1: jax.Array
    v2: jax.Array
    smooth_weights: Union[None, jax.Array]
    interp_method: str = eqx.field(static=True)
    smooth_method: str = eqx.field(static=True)
    coarse_opinv: InverseLinearOperator
    coarse_overweight: jax.Array
    verbose: int = eqx.field(static=True)

    def __init__(
        self,
        operators,
        smoothers,
        x0=None,
        cycle_index=1,
        v1=1,
        v2=1,
        smooth_weights=None,
        interp_method="linear",
        smooth_method="standard",
        coarse_opinv=None,
        coarse_overweight=1.0,
        verbose=False,
    ):

        self.operators = operators
        self.smoothers = smoothers
        self.x0 = x0
        self.cycle_index = jnp.asarray(cycle_index)
        self.v1 = jnp.asarray(v1)
        self.v2 = jnp.asarray(v2)
        self.smooth_weights = smooth_weights
        self.interp_method = interp_method
        self.smooth_method = smooth_method
        if coarse_opinv is None:
            coarse_opinv = InverseLinearOperator(operators[0], lx.LU(), throw=False)
        self.coarse_opinv = coarse_opinv
        self.coarse_overweight = jnp.asarray(coarse_overweight)
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
            smooth_weights=self.smooth_weights,
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


def get_multigrid_preconditioner(
    field,
    E_psi,
    nu,
    nl,
    nz=None,
    nt=None,
    p1="4d",
    p2=4,
    cycle_index=0,
    v1=3,
    v2=3,
    smooth_weights=None,
    interp_method="linear",
    smooth_method="standard",
    smooth_solver="banded",
    coarse_overweight=1.0,
    coarse_N=200,
    coarsening=2,
    gauge=True,
    verbose=False,
):
    """Multigrid linear operator for preconditioning.

    Parameters
    ----------
    field : yancc.Field
        Magnetic field information.
    E_psi : float
        Normalized radial electric field.
    nu : float
        Normalized collisionality.
    nl : int
        Number of grid points in pitch angle.
    nz, nt : int
        Number of grid points in zeta/theta. Defaults to that from field.
    p1 : str
        Finite difference stencil for first derivatives.
    p2 : str
        Finite difference stencil for second derivatives.
    cycle_index : int
        Type of cycle. 0 = F cycle, 1 = V cycle, 2 = W cycle etc.
    v1, v2 : int
        Number of pre- and post- smoothing iterations.
    smooth_weights : jax.Array
        Damping factors for under relaxed smoothing.
    interp_method : str
        Method of interpolation, passed to interpax.interp3d
    smooth_method : {"standard", "krylov"}
        Method to use for smoothing.
    smooth_solver : {"banded", "dense"}
        Solver to use for inverting the smoother. "banded" is significantly faster in
        most cases but may be numerically unstable in some edge cases. "dense" is
        slower but more robust.
    coarse_overweight : float
        Factor to weight coarse grid residuals by, to improve coarse grid correction
        for closed characteristics.
    coarse_N : int
        Approximate max size of coarsest grid problem, which will be solve directly.
    coarsening : int
        How much to coarsen grids by in each direction at each level.
    gauge : bool
        Whether to include gauge freedom constraint.
    verbose : int
        Level of verbosity:
          - 0: no into printed.
          - 1: print residuals after each cycle.
          - 2: also print residuals at each multigrid level before and after smoothing.
          - 3: also print residuals within smoothing iterations.
    """
    if nt is None:
        nt = field.ntheta
    if nz is None:
        nz = field.nzeta

    fields, grids = get_fields_grids(
        field,
        nt,
        nz,
        nl,
        coarsening_factor=coarsening,
        min_N=coarse_N,
        min_nt=5,
        min_nz=5,
        min_nx=5,
    )
    operators = get_operators(fields, grids, E_psi, nu, p1, p2, gauge)
    smoothers = get_jacobi_smoothers(
        fields, grids, E_psi, nu, p1, p2, gauge, smooth_solver
    )
    Mlx = MultigridOperator(
        operators[::-1],
        smoothers[::-1],
        cycle_index=cycle_index,
        v1=v1,
        v2=v2,
        smooth_weights=smooth_weights,
        interp_method=interp_method,
        smooth_method=smooth_method,
        coarse_overweight=coarse_overweight,
        verbose=verbose,
    )
    return Mlx
