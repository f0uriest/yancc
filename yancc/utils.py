"""Simple utility functions."""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jnp.vectorize, signature="()->(n)", excluded=(1,))
def _onehot(i, N):
    return jnp.zeros(N).at[i].set(1.0)


def getslice(operator, rows, cols):
    """Get rows and cols from a given linear operator."""
    if isinstance(rows, slice):
        rows = _slice_to_range(rows, operator.shape[0])

    if isinstance(cols, slice):
        cols = _slice_to_range(cols, operator.shape[1])

    @jax.jit
    def bar(c):
        return operator.mv(_onehot(c, operator.shape[0]))[rows]

    return jax.lax.map(bar, cols, batch_size=100)


def _slice_to_range(s, N):
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else N
    step = s.step if s.step is not None else 1
    start = start % N
    stop = stop & N
    step = step % N
    return jnp.arange(start, stop, step)


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


@jax.jit
@jnp.vectorize
def Gammaincc(s, x):
    """Upper incomplete gamma function."""
    sgn, gammarg = lGammaincc(s, x)
    return sgn * jnp.exp(gammarg)
