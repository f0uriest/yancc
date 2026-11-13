"""Toroidal rotation of the plasma in axisymmetric fields."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .field import Field
from .species import _JOULE_PER_EV, LocalMaxwellian


class Rotation(eqx.Module):
    """Toroidal rotation of the plasma, common to all species.

    The plasma rotates rigidly on each flux surface with the angular velocity omega
    along the toroidal angle of the field. omega*R may be of the order of the thermal
    speeds (sonic rotation): the centrifugal and Coriolis forces, the variation of the
    potential and of the densities on the surface, and the drive by the rotation
    shear are kept. Velocities are then those in the rotating frame, and the densities
    and density gradients of the species are those at theta=0.

    The field must be axisymmetric (``nzeta=1``) and have ``R`` and ``dRdrho``.

    Parameters
    ----------
    omega : float
        Angular velocity of the plasma in the direction of increasing zeta, in rad/s.
    domega_drho : float
        Derivative of omega with respect to rho.

    """

    omega: Float[Array, ""]
    domega_drho: Float[Array, ""]

    def __init__(
        self, omega: Float[ArrayLike, ""], domega_drho: Float[ArrayLike, ""] = 0.0
    ):
        self.omega = jnp.asarray(omega, dtype=jnp.float64)
        self.domega_drho = jnp.asarray(domega_drho, dtype=jnp.float64)


class RotatingEquilibrium(eqx.Module):
    """Lowest order state of a rotating plasma on a flux surface.

    In the frame rotating with the plasma each species is a Maxwellian with density

    n_s(θ) = n_s(0) exp(-λ_s(θ)),  λ_s = [q_s Φ₀(θ) - m_s ω² (R(θ)² - R(0)²)/2] / T_s

    where the potential Φ₀, zero at θ=0, is such that the charge density is the same
    at every point of the surface as at θ=0.

    Parameters
    ----------
    field : Field
        Magnetic field information. Must be axisymmetric and have R and dRdrho.
    species : list[LocalMaxwellian]
        Species being considered. Densities and density gradients are those at θ=0.
    rotation : Rotation
        Toroidal rotation of the plasma.
    background : list[LocalMaxwellian], optional
        Static background species, which rotate with the plasma as well.

    Attributes
    ----------
    Phi : jax.Array, shape(ntheta, nzeta)
        Potential Φ₀ on the surface, in Volts.
    lam, dlamdt : jax.Array, shape(ns, ntheta, nzeta)
        Potential energy λ_s of each species in the rotating frame, normalized to its
        temperature, and its derivative with respect to theta.
    density : jax.Array, shape(ns, ntheta, nzeta)
        Density of each species normalized to its density at θ=0.
    background_density : jax.Array, shape(nb, ntheta, nzeta)
        Density of each background species normalized to its density at θ=0.
    dlnNdrho : jax.Array, shape(ns)
        -(m_s/T_s) (ω dω/dρ R² + ω² R dR/dρ) at θ=0, the part of the logarithmic
        density gradient at θ=0 that does not drive transport.

    """

    omega: Float[Array, ""]
    domega_drho: Float[Array, ""]
    Phi: Float[Array, "nt nz"]
    lam: Float[Array, "ns nt nz"]
    dlamdt: Float[Array, "ns nt nz"]
    density: Float[Array, "ns nt nz"]
    background_density: Float[Array, "nb nt nz"]
    dlnNdrho: Float[Array, " ns"]

    def __init__(
        self,
        field: Field,
        species: list[LocalMaxwellian],
        rotation: Rotation,
        background: list[LocalMaxwellian] | None = None,
    ):
        if field.nzeta != 1:
            raise ValueError("Toroidal rotation requires an axisymmetric field.")
        if field.R is None or field.dRdrho is None:
            raise ValueError("Toroidal rotation requires a Field with R and dRdrho.")
        if background is None:
            background = []
        ns = len(species)
        all_species = list(species) + list(background)

        # charge in units of e, temperature in eV, so that Z*Phi/T is dimensionless
        Z = jnp.array([sp.species.charge for sp in all_species]) / _JOULE_PER_EV
        n = jnp.array([sp.density for sp in all_species])
        T = jnp.array([sp.temperature for sp in all_species])
        m_T = jnp.array([sp.species.mass for sp in all_species]) / (T * _JOULE_PER_EV)
        Z, n, T, m_T = (a[:, None, None] for a in (Z, n, T, m_T))

        self.omega = rotation.omega
        self.domega_drho = rotation.domega_drho
        R = field.R
        dRdt = field._dfdt(R)
        R0 = R[0, 0]
        centrifugal = m_T * self.omega**2 * (R**2 - R0**2) / 2

        Phi = _quasineutral_potential(Z, n, T, centrifugal)
        lam = Z * Phi / T - centrifugal
        density = jnp.exp(-lam)
        # dΦ₀/dθ from the derivative of the quasineutrality equation
        dPhidt = (
            self.omega**2
            * R
            * dRdt
            * (Z * n * density * m_T).sum(axis=0)
            / (Z**2 * n * density / T).sum(axis=0)
        )
        dlamdt = Z * dPhidt / T - m_T * self.omega**2 * R * dRdt

        self.Phi = Phi
        self.lam = lam[:ns]
        self.dlamdt = dlamdt[:ns]
        self.density = density[:ns]
        self.background_density = density[ns:]
        self.dlnNdrho = -m_T[:ns, 0, 0] * (
            self.omega * self.domega_drho * R0**2
            + self.omega**2 * R0 * field.dRdrho[0, 0]
        )


def _quasineutral_potential(Z, n, T, centrifugal):
    """Potential at which the charge density equals that at the reference point.

    Solves sum_s Z_s n_s [exp(c_s - Z_s Φ/T_s) - 1] = 0 for Φ at each point of the
    surface by Newton iteration, where c_s is the centrifugal energy over T_s. The
    left hand side decreases monotonically with Φ.
    """

    def residual(phi, Z, n, T, centrifugal):
        ex = jnp.exp(centrifugal - Z * phi / T)
        f = (Z * n * (ex - 1)).sum(axis=0)
        df = -(Z**2 * n / T * ex).sum(axis=0)
        return f, df

    args = (Z, n, T, centrifugal)
    frozen = jax.lax.stop_gradient(args)
    # largest step: a change of one in every exponent
    cap = jnp.min(frozen[2] / jnp.abs(frozen[0]))

    def cond(state):
        _, step, k = state
        return (jnp.max(jnp.abs(step)) > 1e-12 * cap) & (k < 1000)

    def body(state):
        phi, _, k = state
        f, df = residual(phi, *frozen)
        step = jnp.clip(-f / df, -cap, cap)
        return phi + step, step, k + 1

    shape = jnp.broadcast_shapes(*(a.shape for a in args))[1:]
    dtype = jnp.result_type(*args)
    phi = jnp.zeros(shape, dtype=dtype)
    phi, _, _ = jax.lax.while_loop(
        cond, body, (phi, jnp.full(shape, jnp.inf, dtype=dtype), 0)
    )
    phi = jax.lax.stop_gradient(phi)
    # Newton steps from the converged potential carry its derivatives with respect
    # to the inputs: each step adds an order
    for _ in range(2):
        f, df = residual(phi, *args)
        phi = phi - f / df
    return phi
