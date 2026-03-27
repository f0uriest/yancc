import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .field import SplineField


def dkes_w_theta_pinn(
    a, t, z, field: SplineField, erhohat: Float[Array, ""]
) -> Float[Array, ""]:
    """Wind in theta direction for MDKE."""
    xi = -jnp.cos(a)
    w = (
        field.B_sup_t(t, z) / field.Bmag(t, z) * xi
        + field.B_sub_z(t, z) / field.B2mag_fsa / field.sqrtg(t, z) * erhohat
    )
    return w


def dkes_w_zeta_pinn(
    a, t, z, field: SplineField, erhohat: Float[Array, ""]
) -> Float[Array, ""]:
    """Wind in zeta direction for MDKE."""
    xi = -jnp.cos(a)
    w = (
        field.B_sup_z(t, z) / field.Bmag(t, z) * xi
        - field.B_sub_t(t, z) / field.B2mag_fsa / field.sqrtg(t, z) * erhohat
    )
    return w


def dkes_w_pitch_pinn(a, t, z, field: SplineField) -> Float[Array, ""]:
    """Wind in xi/pitch direction for MDKE."""
    xi = -jnp.cos(a)
    sina = jnp.sin(a)
    w = -field.bdotgradB(t, z) / (2 * field.Bmag(t, z)) * (1 - xi**2) / sina
    return w


def mdke_rhs_pinn(a, t, z, field: SplineField) -> jax.Array:
    """RHS of monoenergetic DKE.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.

    Returns
    -------
    f : jax.Array, shape(N,3)
        RHS of linear monoenergetic DKE.
    """
    xi = -jnp.cos(a)
    s1 = (1 + xi**2) / (2 * field.Bmag(t, z) ** 3) * field.BxgradrhodotgradB(t, z)
    s2 = s1
    s3 = xi * field.Bmag(t, z)
    rhs = jnp.array([s1, s2, s3]).reshape((3, -1)).T
    return rhs


def mdke_pinn(a, t, z, pinn, params, nuhat, erhohat, field: SplineField):
    wt = dkes_w_theta_pinn(a, t, z, field, erhohat)
    wz = dkes_w_zeta_pinn(a, t, z, field, erhohat)
    wa = dkes_w_pitch_pinn(a, t, z, field)

    dfdt = pinn.dfdt(a, t, z, params)
    dfdz = pinn.dfdz(a, t, z, params)
    dfda = pinn.dfda(a, t, z, params)
    ddfda = pinn.ddfda(a, t, z, params)

    sina = jnp.sin(a)
    cosa = jnp.cos(a)
    f1 = -(nuhat / 2 * cosa / sina) * dfda
    f2 = -nuhat / 2 * ddfda

    df = wt * dfdt + wz * dfdz + wa * dfda + f1 + f2
    return df


def mdke_bc_pinn(a, t, z, pinn, params, nuhat, erhohat, field: SplineField):
    f = pinn.dfdt(a, t, z, params)
    ft1 = pinn.dfdt(a, t + 2 * jnp.pi, z, params)
    ft2 = pinn.dfdt(a, t - 2 * jnp.pi, z, params)
    fz1 = pinn.dfdt(a, t, z + 2 * jnp.pi / field.NFP, params)
    fz2 = pinn.dfdt(a, t, z - 2 * jnp.pi / field.NFP, params)
    fa1 = pinn.dfdt(-a, t, z, params)
    fa2 = pinn.dfdt(2 * jnp.pi - a, t, z, params)
    return jnp.concatenate([ft1 - f, ft2 - f, fz1 - f, fz2 - f, fa1 - f, fa2 - f])


class MDKEPinn(eqx.Module):

    sigma: jax.Array
    N: int = eqx.field(static=True)
    W: jax.Array
    b: jax.Array
    c = jax.Array

    def __init__(self, N, sigma, W=None, b=None, c=None, key=jnp.array(123)):
        keyw, keyb, keyc = jax.random.split(key)
        if W is None:
            W = jax.random.normal(keyw, shape=(N, 3))
        assert W.shape[0] == N
        assert sigma.shape[0] == N
        self.W = sigma * W
        self.sigma = sigma
        if b is None:
            b = jax.random.uniform(keyb, shape=(N,)) * 2 * jnp.pi
        assert b.shape[0] == N
        self.b = b
        if c is None:
            c = jax.random.normal(keyb, shape=(N,))
        assert c.shape[0] == N
        self.c = c

    def __call__(self, a, t, z, c=None, W=None, b=None):
        if c is None:
            c = self.c
        if W is None:
            W = self.W
        if b is None:
            b = self.b

        return _pinn_call_pure(a, t, z, c, W, b)


@jax.jit
@functools.partial(jnp.vectorize, signature="(),(),(),(N),(N,k),(N)->(N)")
def _pinn_call_pure(a, t, z, c, W, b):
    x = jnp.array([a, t, z])
    y = jnp.dot(x, W) + b
    y = jnp.sin(y)
    return jnp.dot(c, y)
