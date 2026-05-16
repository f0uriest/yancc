"""Tests for finite differences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yancc.finite_diff import (
    build_advection_matrix,
    build_lorentz_matrix,
    fd2,
    fd_coeffs,
    fd_kwargs,
    fdbwd,
    fdfwd,
)


@pytest.mark.parametrize("p", fd_coeffs[2].keys())
def test_fd2_periodic(p):
    """Validate centered finite differences w/ periodic bc."""
    Ns = (2 ** np.linspace(3, 6, 10)).astype(int)
    errs = []
    for N in Ns:
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        f = np.sin(x)
        ddf = -np.sin(x)
        h = 2 * np.pi / N
        ddfi = fd2(f, p, h, bc="periodic")
        err = np.max(np.abs(ddfi - ddf))
        errs.append(err)

    order, _ = np.polyfit(np.log(Ns), -np.log(errs), deg=1)
    assert order > p - 0.1


@pytest.mark.parametrize("p", fd_coeffs[1].keys())
def test_fdfwd_periodic(p):
    """Validate forward finite differences w/ periodic bc."""
    Ns = (2 ** np.linspace(3, 6, 10)).astype(int)
    errs = []
    for N in Ns:
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        f = np.sin(x)
        df = np.cos(x)
        h = 2 * np.pi / N
        dfi = fdfwd(f, p, h, bc="periodic")
        err = np.max(np.abs(dfi - df))
        errs.append(err)

    order, _ = np.polyfit(np.log(Ns), -np.log(errs), deg=1)
    assert order > int(p[0]) - 0.2


@pytest.mark.parametrize("p", fd_coeffs[1].keys())
def test_fdbwd_periodic(p):
    """Validate backward finite differences w/ periodic bc."""
    Ns = (2 ** np.linspace(3, 6, 10)).astype(int)
    errs = []
    for N in Ns:
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        f = np.sin(x)
        df = np.cos(x)
        h = 2 * np.pi / N
        dfi = fdbwd(f, p, h, bc="periodic")
        err = np.max(np.abs(dfi - df))
        errs.append(err)

    order, _ = np.polyfit(np.log(Ns), -np.log(errs), deg=1)
    assert order > int(p[0]) - 0.2


@pytest.mark.parametrize("p", fd_coeffs[2].keys())
def test_fd2_symmetric(p):
    """Validate centered finite differences w/ symmetric bc."""
    Ns = (2 ** np.linspace(2, 5, 10)).astype(int)
    errs = []
    for N in Ns:
        x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
        f = np.cos(x)
        ddf = -np.cos(x)
        h = np.pi / N
        ddfi = fd2(f, p, h, bc="symmetric")
        err = np.max(np.abs(ddfi - ddf))
        errs.append(err)

    order, _ = np.polyfit(np.log(Ns), -np.log(errs), deg=1)
    np.testing.assert_allclose(order, p, atol=0.1)


@pytest.mark.parametrize("p", fd_coeffs[1].keys())
def test_fdfwd_symmetric(p):
    """Validate forward finite differences w/ symmetric bc."""
    Ns = (2 ** np.linspace(3, 6, 10)).astype(int)
    errs = []
    for N in Ns:
        x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
        f = np.cos(x)
        df = -np.sin(x)
        h = np.pi / N
        dfi = fdfwd(f, p, h, bc="symmetric")
        err = np.max(np.abs(dfi - df))
        errs.append(err)

    order, _ = np.polyfit(np.log(Ns), -np.log(errs), deg=1)
    assert order > int(p[0]) - 0.2


@pytest.mark.parametrize("p", fd_coeffs[1].keys())
def test_fdbwd_symmetric(p):
    """Validate backward finite differences w/ symmetric bc."""
    Ns = (2 ** np.linspace(3, 6, 10)).astype(int)
    errs = []
    for N in Ns:
        x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
        f = np.cos(x)
        df = -np.sin(x)
        h = np.pi / N
        dfi = fdbwd(f, p, h, bc="symmetric")
        err = np.max(np.abs(dfi - df))
        errs.append(err)

    order, _ = np.polyfit(np.log(Ns), -np.log(errs), deg=1)
    assert order > int(p[0]) - 0.2


def _ad_lorentz(fun):
    # in xi = -cos(a)
    return jax.grad(lambda xi: (1 - xi**2) * jax.grad(fun)(xi))


@pytest.mark.parametrize("p", [2, 4, 6])
def test_lorentz(p):
    """Test finite difference Lorentz operator."""
    # test function that has lots of wiggles near xi=0
    fun = lambda x: jnp.sin(20 * x) / (1 + (10 * x) ** 2) + jnp.tanh(5 * x)
    dfun = jnp.vectorize(_ad_lorentz(fun))

    ns = (2 ** np.linspace(3, 10, 10)).astype(int)
    errs_uni = []
    errs_non = []

    c = 0.7

    # nonuniform mapping function to pack nodes near center
    def map1(x):
        x = 2 * (x / np.pi - 0.5)
        x = c * x**3 + (1 - c) * x
        x = (x + 1) / 2 * np.pi
        return x

    for n in ns:
        a_uni = np.linspace(0, np.pi, n, endpoint=False) + np.pi / (2 * n)
        a_non = map1(a_uni)
        xi_uni = -np.cos(a_uni)
        xi_non = -np.cos(a_non)
        f_uni = fun(xi_uni)
        f_non = fun(xi_non)
        D_uni = build_lorentz_matrix(a_uni, p)
        D_non = build_lorentz_matrix(a_non, p)
        Df1_uni = dfun(xi_uni)
        Df1_non = dfun(xi_non)
        Df2_uni = D_uni @ f_uni
        Df2_non = D_non @ f_non
        errs_uni.append(max(abs(Df1_uni - Df2_uni)))
        errs_non.append(max(abs(Df1_non - Df2_non)))

    # nonuniform grid that packs nodes near strong gradients should have
    # much lower error
    assert errs_non[-1] < 0.1 * errs_uni[-1]
    # but both should have the same asymptotic order of convergence
    order_uni, _ = np.polyfit(np.log(ns)[-4:], -np.log(errs_uni)[-4:], deg=1)
    order_non, _ = np.polyfit(np.log(ns)[-4:], -np.log(errs_non)[-4:], deg=1)
    assert order_uni > p - 0.2
    assert order_non > p - 0.2


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize("p", fd_kwargs.keys())
@pytest.mark.parametrize("bc", ["periodic", "symmetric"])
def test_advection_uniform(direction, p, bc):
    """Test finite difference advection operator with uniform spacing."""
    # we already test order of convergence for these stencils above,
    # so this is just to test that different ways of constructing the matrix is the same
    if direction == "fwd":
        fdfun = fdfwd
    else:
        fdfun = fdbwd

    n = 101
    if bc == "periodic":
        a_uni = np.linspace(0, 2 * np.pi, n, endpoint=False)
        domain = (0, 2 * np.pi)
    else:
        a_uni = np.linspace(0, np.pi, n, endpoint=False) + np.pi / (2 * n)
        domain = (0, np.pi)
    kwargs = fd_kwargs[p]
    D_uni = build_advection_matrix(
        a_uni, direction=direction, bc_type=bc, domain=domain, **kwargs
    )
    h = (domain[1] - domain[0]) / n
    D_old = jax.jacfwd(fdfun)(a_uni, p, h, bc)
    np.testing.assert_allclose(D_old, D_uni, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize("p", fd_kwargs.keys())
def test_advection_nonuniform(direction, p):
    """Test finite difference advection operator with nonuniform spacing."""
    # we already test order of convergence for these stencils above,
    # so this is just testing that nonuniform grid has lower error for local features.

    # test function that has lots of wiggles near xi=0
    def fun_xi(x):
        return 2 * jnp.sin(10 * x) / (1 + (10 * x) ** 2) + jnp.tanh(5 * x)

    def fun(a):
        x = -jnp.cos(a)
        return fun_xi(x)

    dfun = jnp.vectorize(jax.grad(fun))

    n = 257
    c = 0.7
    kwargs = fd_kwargs[p]

    # nonuniform mapping function to pack nodes near center
    def map1(x):
        x = 2 * (x / np.pi - 0.5)
        x = c * x**3 + (1 - c) * x
        x = (x + 1) / 2 * np.pi
        return x

    a_uni = np.linspace(0, np.pi, n, endpoint=False) + np.pi / (2 * n)
    a_non = map1(a_uni)
    f_uni = fun(a_uni)
    f_non = fun(a_non)
    D_uni = build_advection_matrix(
        a_uni, direction=direction, bc_type="symmetric", domain=(0, np.pi), **kwargs
    )
    D_non = build_advection_matrix(
        a_non, direction=direction, bc_type="symmetric", domain=(0, np.pi), **kwargs
    )
    Df1_uni = dfun(a_uni)
    Df1_non = dfun(a_non)
    Df2_uni = D_uni @ f_uni
    Df2_non = D_non @ f_non
    err_uni = max(abs(Df1_uni - Df2_uni))
    err_non = max(abs(Df1_non - Df2_non))

    # nonuniform grid that packs nodes near strong gradients should have
    # much lower error
    if kwargs["order"] > 1:
        fac = 0.1
    else:
        fac = 0.5
    assert err_non < fac * err_uni
