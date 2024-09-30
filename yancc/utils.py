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


def _gammastar(s, z, kmax=40):
    k = jnp.arange(0, kmax)
    gammarg = s + k + 1
    gammasn = jax.scipy.special.gammasgn(gammarg)
    x = k * jnp.log(z) - jax.scipy.special.gammaln(gammarg)
    y, sgn = jax.scipy.special.logsumexp(x, b=gammasn, return_sign=True)
    t = -z + sgn * y
    return jnp.exp(t)


@jax.jit
@jnp.vectorize
def gammainc(s, x):
    """Lower incomplete gamma function."""
    return x**s * _gammastar(s, x)


@jax.jit
@jnp.vectorize
def gammaincc(s, x):
    """Upper incomplete gamma function."""
    return 1 - gammainc(s, x)
