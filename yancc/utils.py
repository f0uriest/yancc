"""Simple utility functions."""

import jax
import jax.numpy as jnp


def _lgammastar(s, z, kmax=60):
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

        e1, r = jax.lax.fori_loop(1, 30, body, (e1, r), unroll=20)
        return -jnp.euler_gamma - jnp.log(x) + x * e1

    def xgt2():
        m = 40
        t0 = 0.0

        def body(i, t0):
            k = m - i
            t0 = k / (1.0 + k / (x + t0))
            return t0

        t0 = jax.lax.fori_loop(0, m, body, t0, unroll=20)
        t = 1.0 / (x + t0)
        return jnp.exp(-x) * t

    e1 = jax.lax.cond(x <= 2, xlt2, xgt2)
    return jnp.where(x == 0, jnp.inf, e1)


@jax.jit
@jnp.vectorize
def lGammainc(s, x):
    """Log of lower incomplete gamma function.

    Returns (sign, lGammainc) st Gammainc = sign * exp(lGammainc)
    """
    sgn, t = _lgammastar(s, x)
    gammasn = jax.scipy.special.gammasgn(s)
    return (
        gammasn * sgn * jnp.sign(x),
        jax.scipy.special.gammaln(s) + s * jnp.log(x) + t,
    )


@jax.jit
@jnp.vectorize
def Gammainc(s, x):
    """Lower incomplete gamma function."""
    sgn, gammarg = lGammainc(s, x)
    return sgn * jnp.exp(gammarg)


@jax.jit
@jnp.vectorize
def lGammaincc(s, x):
    """Log of upper incomplete gamma function.

    Returns (sign, lGammaincc) st Gammaincc = sign * exp(lGammaincc)
    """

    def spos():
        gammasn = jax.scipy.special.gammasgn(s)
        sgn, t = _lgammastar(s, x)
        y = x**s * sgn * jnp.exp(t)
        # log(|y|) = s log(x) + t
        logy = s * jnp.log(jnp.abs(x)) + t
        sgny = jnp.sign(y)
        sgnlogy = jnp.sign(logy)
        # 1-y>0 when y<1 -> sign(y)<0 or log(|y|) < 0
        # sign(1-y) = -sign(log(|y|))
        sgn1my = jnp.where(sgny < 0, 1, -sgnlogy)
        # when |y| ~ 0 (ie log|y| << 0) we want to use log1p(-y)
        # otherwise just do regular log(|1-y|)
        log1my = jnp.where(logy < -3, jnp.log1p(-y), jnp.log(jnp.abs(1 - y)))
        return gammasn * sgn1my, jax.scipy.special.gammaln(s) + log1my

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

    return jax.lax.cond((s <= 0) & (s % 1 == 0), sneg, spos)


@jax.jit
@jnp.vectorize
def Gammaincc(s, x):
    """Upper incomplete gamma function."""
    sgn, gammarg = lGammaincc(s, x)
    return sgn * jnp.exp(gammarg)
