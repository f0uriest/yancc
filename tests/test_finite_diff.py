"""Tests for finite differences."""

import numpy as np
import pytest

from yancc.finite_diff import fd2, fd_coeffs, fdbwd, fdfwd


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
