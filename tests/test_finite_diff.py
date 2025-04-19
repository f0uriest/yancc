"""Tests for finite differences."""

import numpy as np

from yancc.finite_diff import fd2, fd_coeffs, fdbwd, fdfwd


def test_fd2_periodic():
    """Validate centered finite differences w/ periodic bc."""
    N = 21
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.sin(x)
    ddf = -np.sin(x)
    h = 2 * np.pi / N
    for order in [2, 4, 6]:
        ddfi = fd2(f, order, h, bc="periodic")
        np.testing.assert_allclose(
            ddf, ddfi, atol=h ** (order + 1), rtol=h ** (order + 1)
        )


def test_fdfwd_periodic():
    """Validate forward finite differences w/ periodic bc."""
    N = 21
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.sin(x)
    df = np.cos(x)
    h = 2 * np.pi / N
    for p in fd_coeffs[1].keys():
        order = int(p[0])
        dfi = fdfwd(f, p, h, bc="periodic")
        np.testing.assert_allclose(
            df, dfi, atol=h ** (order), rtol=h ** (order), err_msg=p
        )


def test_fdbwd_periodic():
    """Validate backward finite differences w/ periodic bc."""
    N = 21
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.sin(x)
    df = np.cos(x)
    h = 2 * np.pi / N
    for p in fd_coeffs[1].keys():
        order = int(p[0])
        dfi = fdbwd(f, p, h, bc="periodic")
        np.testing.assert_allclose(
            df, dfi, atol=h ** (order), rtol=h ** (order), err_msg=p
        )


def test_fd2_symmetric():
    """Validate centered finite differences w/ symmetric bc."""
    N = 21
    x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
    f = np.cos(x)
    ddf = -np.cos(x)
    h = np.pi / N
    for order in [2, 4, 6]:
        ddfi = fd2(f, order, h, bc="symmetric")
        np.testing.assert_allclose(
            ddf, ddfi, atol=h ** (order + 1), rtol=h ** (order + 1)
        )


def test_fdfwd_symmetric():
    """Validate forward finite differences w/ symmetric bc."""
    N = 21
    x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
    f = np.cos(x)
    df = -np.sin(x)
    h = np.pi / N
    for p in fd_coeffs[1].keys():
        order = int(p[0])
        dfi = fdfwd(f, p, h, bc="symmetric")
        np.testing.assert_allclose(
            df, dfi, atol=h ** (order), rtol=h ** (order), err_msg=p
        )


def test_fdbwd_symmetric():
    """Validate backward finite differences w/ symmetric bc."""
    N = 21
    x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
    f = np.cos(x)
    df = -np.sin(x)
    h = np.pi / N
    for p in fd_coeffs[1].keys():
        order = int(p[0])
        dfi = fdbwd(f, p, h, bc="symmetric")
        np.testing.assert_allclose(
            df, dfi, atol=h ** (order), rtol=h ** (order), err_msg=p
        )
