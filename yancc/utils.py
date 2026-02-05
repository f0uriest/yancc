"""Simple utility functions."""

import jax
import jax.numpy as jnp
import numpy as np


def _lgammastar(s, z, kmax=1000):
    k = jnp.arange(0, kmax)
    gammarg = s + k + 1
    gammasn = jax.scipy.special.gammasgn(gammarg)
    x = k * jnp.log(z) - jax.scipy.special.gammaln(gammarg)
    y, sgn = jax.scipy.special.logsumexp(x, b=gammasn, return_sign=True)
    t = -z + y
    return sgn, t


@jax.jit
@jnp.vectorize
def _exp1(x):
    """Exponential integral E1."""
    # adapted from
    # https://github.com/scipy/scipy/blob/b9b8b8171fd1453b42fc4492a279c71b54141c51/scipy/special/xsf/expint.h#L54 # noqa:E501

    def xlt2():
        e1 = 1.0
        r = 1.0

        def body(k, er):
            e1, r = er
            r = -r * k * x / (k + 1.0) ** 2
            e1 += r
            return e1, r

        e1, r = jax.lax.fori_loop(1, 100, body, (e1, r), unroll=True)
        return -jnp.euler_gamma - jnp.log(x) + x * e1

    def xgt2():
        m = 100
        t0 = 0.0

        def body(i, t0):
            k = m - i
            t0 = k / (1.0 + k / (x + t0))
            return t0

        t0 = jax.lax.fori_loop(0, m, body, t0, unroll=True)
        t = 1.0 / (x + t0)
        return jnp.exp(-x) * t

    e1 = jax.lax.cond(x <= 2, xlt2, xgt2)
    return jnp.where(x == 0, jnp.inf, e1)


@jnp.vectorize
def _lGammainc_large_x(s, x):
    # gamma(s,x) = Gamma(s) - Gamma(s,x)
    sgn1 = jax.scipy.special.gammasgn(s)
    G = jax.scipy.special.gammaln(s)
    sgn2, F = lGammaincc(s, x)
    x = jnp.array([G, F])
    sgn = jnp.array([sgn1, -sgn2])
    out, sgn = jax.scipy.special.logsumexp(x, b=sgn, return_sign=True)
    return sgn, out


@jax.jit
@jnp.vectorize
def lGammainc(s, x):
    """Log of lower incomplete gamma function.

    Valid for real x>0 and real s (any sign)

    Returns (sign, lGammainc) st Gammainc = sign * exp(lGammainc)
    """

    def small_x():
        sgn, t = _lgammastar(s, x)
        gammasn = jax.scipy.special.gammasgn(s)
        return (
            gammasn * sgn * jnp.sign(x),
            jax.scipy.special.gammaln(s) + s * jnp.log(x) + t,
        )

    def large_x():
        return _lGammainc_large_x(s, x)

    return jax.lax.cond(x > 100, large_x, small_x)


@jax.jit
@jnp.vectorize
def Gammainc(s, x):
    """Lower incomplete gamma function.

    Valid for real x>0 and real s (any sign)
    """
    sgn, gammarg = lGammainc(s, x)
    return sgn * jnp.exp(gammarg)


def _lGammaincc_large_x_correction(s, x, kmax=20):
    k = jnp.arange(0, kmax)
    arg = s - k
    arg = arg.at[0].set(1)
    arg = jnp.cumprod(arg)
    sgn = jnp.sign(arg) * jnp.sign(x ** (k % 2))
    f = -k * jnp.log(jnp.abs(x)) + jnp.log(jnp.abs(arg))
    f, sign = jax.scipy.special.logsumexp(f, b=sgn, return_sign=True)
    return sign, f


def _lGammaincc_large_x(s, x, kmax=20):
    """Asymptotic formula for x -> inf (practical for x >~ 10)"""
    sgn, t = _lGammaincc_large_x_correction(s, x, kmax)
    sgn = jnp.sign(x ** ((s - 1) % 2)) * sgn
    return sgn, ((s - 1) * jnp.log(x) - x) + t


@jax.jit
@jnp.vectorize
def lGammaincc(s, x):
    """Log of upper incomplete gamma function.

    Valid for real x>0 and real s (any sign)

    Returns (sign, lGammaincc) st Gammaincc = sign * exp(lGammaincc)
    """

    def spos():
        # using Gamma(s,x) = Gamma(s) - gamma(s,x) = Gamma(s)(1 - x^s gammastar(s,x))
        # lGamma(s,x) = lGamma(s) + log(1 - x^s gammastar(s,x))
        #             = lGamma(s) + logsumexp(0, log(x^s gammastar))
        #             = lGamma(s) + logsumexp(0, log(x^s gammastar))
        gammasn = jax.scipy.special.gammasgn(s)
        sgn1, t1 = _lgammastar(s, x)
        sgn2, t2 = jnp.sign(x) ** s, s * jnp.log(jnp.abs(x))
        sgn = jnp.array([1, -sgn1 * sgn2])
        arg = jnp.array([0, t1 + t2])
        y, sgn = jax.scipy.special.logsumexp(arg, b=sgn, return_sign=True)
        return gammasn * sgn, jax.scipy.special.gammaln(s) + y

    def sneg():
        # using recursive definition from wikipedia
        si = 0
        Gamma_sx = _exp1(x)
        expx = jnp.exp(-x)

        def body(state):
            si, Gamma_sx = state
            Gamma_sm1x = (Gamma_sx - x ** (si - 1) * expx) / (si - 1)
            return si - 1, Gamma_sm1x

        def cond(state):
            si, _ = state
            return si > s

        _, gincc = jax.lax.while_loop(cond, body, (si, Gamma_sx))

        sgn = jnp.sign(gincc)
        return sgn, jnp.log(jnp.abs(gincc))

    def large_x():
        return _lGammaincc_large_x(s, x)

    def small_x():
        return jax.lax.cond((s <= 0) & (s % 1 == 0), sneg, spos)

    return jax.lax.cond(x > 20, large_x, small_x)


@jax.jit
@jnp.vectorize
def Gammaincc(s, x):
    """Upper incomplete gamma function.

    Valid for real x>0 and real s (any sign)
    """
    sgn, gammarg = lGammaincc(s, x)
    return sgn * jnp.exp(gammarg)


def _parse_axorder_shape_3d(
    nt: int, nz: int, na: int, axorder: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = np.empty(3, dtype=int)
    shape[axorder.index("a")] = na
    shape[axorder.index("t")] = nt
    shape[axorder.index("z")] = nz
    caxorder = (axorder.index("a"), axorder.index("t"), axorder.index("z"))
    return tuple(shape), caxorder


def _parse_axorder_shape_4d(
    nt: int, nz: int, na: int, nx: int, ns: int, axorder: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = np.empty(5, dtype=int)
    shape[axorder.index("a")] = na
    shape[axorder.index("t")] = nt
    shape[axorder.index("z")] = nz
    shape[axorder.index("x")] = nx
    shape[axorder.index("s")] = ns
    caxorder = (
        axorder.index("s"),
        axorder.index("x"),
        axorder.index("a"),
        axorder.index("t"),
        axorder.index("z"),
    )
    return tuple(shape), caxorder


def _refold(a, k):
    N, M, _ = a.shape
    a = a.reshape((N // k, k, M, M))
    # TODO: make this better
    return jax.vmap(lambda x: jax.scipy.linalg.block_diag(*x))(a)
