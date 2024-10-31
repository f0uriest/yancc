"""Collision operators and methods for computing Rosenbluth potentials."""

import functools

import cola
import equinox as eqx
import jax
import jax.numpy as jnp
import monkes
import numpy as np
import orthax
import quadax

from .utils import Gammainc, Gammaincc
from .velocity_grids import PitchAngleGrid, SpeedGrid


class RosenbluthPotentials(eqx.Module):
    """Thing to calculate Rosenbluth Potentials.

    Parameters
    ----------
    xgrid : SpeedGrid
        Grid of coordinates in speed.
    xigrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    quad : bool
        Whether to compute potentials using quadrature or incomplete gamma functions
    """

    xgrid: SpeedGrid
    xigrid: PitchAngleGrid
    quad: bool = eqx.field(static=True)
    ddGxlk: jax.Array
    Hxlk: jax.Array
    dHxlk: jax.Array

    def __init__(self, xgrid, xigrid, species, quad=False):
        self.xgrid = xgrid
        self.xigrid = xigrid
        self.quad = quad

        ns = len(species)
        x = self.xgrid.x[:, None, None]
        l = jnp.arange(self.xigrid.nxi)[None, :, None]
        k = jnp.arange(self.xgrid.nx)[None, None, :]
        self.ddGxlk = jnp.zeros((ns, ns, self.xgrid.nx, self.xigrid.nxi, self.xgrid.nx))
        self.dHxlk = jnp.zeros((ns, ns, self.xgrid.nx, self.xigrid.nxi, self.xgrid.nx))
        self.Hxlk = jnp.zeros((ns, ns, self.xgrid.nx, self.xigrid.nxi, self.xgrid.nx))
        # arr[a,b] is potential operator from species b to species a
        for a, spa in enumerate(species):
            for b, spb in enumerate(species):
                va, vb = spa.v_thermal, spb.v_thermal
                v = x * va  # speed on a grid
                xb = v / vb  # on b grid
                ddG = jax.jit(jnp.vectorize(self._ddGlk))(xb, l, k)
                dH = jax.jit(jnp.vectorize(self._dHlk))(xb, l, k)
                H = jax.jit(jnp.vectorize(self._Hlk))(xb, l, k)
                self.ddGxlk = self.ddGxlk.at[a, b].set(ddG)
                self.dHxlk = self.dHxlk.at[a, b].set(dH)
                self.Hxlk = self.Hxlk.at[a, b].set(H)

    def rosenbluth_ddG(self, f, a, b):
        """Second derivative of G potential for species indices a, b"""
        # this only knows about a single species,
        # f assumed to be shape(xi, x, theta, zeta)
        # transform from nodal->modal for x, xi
        f = jnp.einsum(
            "kx,li,ixtz->lktz", self.xgrid.xvander_inv, self.xigrid.xivander_inv, f
        )
        # project onto greens fn of poisson eqn
        ddGl = jnp.einsum("lktz,xlk->lxtz", f, self.ddGxlk[a, b])
        # convert back to real space
        ddG = jnp.einsum("il,lxtz->ixtz", self.xigrid.xivander, ddGl)
        return ddG

    def rosenbluth_H(self, f, a, b):
        """H potential for species indices a, b"""
        # this only knows about a single species,
        # f assumed to be shape(xi, x, theta, zeta)
        # transform from nodal->modal for x, xi
        f = jnp.einsum(
            "kx,li,ixtz->lktz", self.xgrid.xvander_inv, self.xigrid.xivander_inv, f
        )
        # project onto greens fn of poisson eqn
        Hl = jnp.einsum("lktz,xlk->lxtz", f, self.Hxlk[a, b])
        # convert back to real space
        H = jnp.einsum("il,lxtz->ixtz", self.xigrid.xivander, Hl)
        return H

    def rosenbluth_dH(self, f, a, b):
        """Derivative of H potential for species indices a, b"""
        # this only knows about a single species,
        # f assumed to be shape(xi, x, theta, zeta)
        # transform from nodal->modal for x, xi
        f = jnp.einsum(
            "kx,li,ixtz->lktz", self.xgrid.xvander_inv, self.xigrid.xivander_inv, f
        )
        # project onto greens fn of poisson eqn
        dHl = jnp.einsum("lktz,xlk->lxtz", f, self.dHxlk[a, b])
        # convert back to real space
        dH = jnp.einsum("il,lxtz->ixtz", self.xigrid.xivander, dHl)
        return dH

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand1(self, z, l, k):
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        return (
            z ** (-l + 1)
            * orthax.orthval(z, c, self.xgrid.xrec)
            * self.xgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand2(self, z, l, k):
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        return (
            z ** (l + 2)
            * orthax.orthval(z, c, self.xgrid.xrec)
            * self.xgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand3(self, z, l, k):
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        return (
            z ** (-l + 3)
            * orthax.orthval(z, c, self.xgrid.xrec)
            * self.xgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand4(self, z, l, k):
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        return (
            z ** (l + 4)
            * orthax.orthval(z, c, self.xgrid.xrec)
            * self.xgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_1(self, x, l, k):
        if self.quad:
            return quadax.quadgk(self._integrand1, (x, np.inf), (l, k))[0]
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.xgrid.xrec)
        n = jnp.arange(self.xgrid.nx)
        g = Gammaincc(-l/2 + n / 2 + 1, x**2)
        return jnp.sum(p * g)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_2(self, x, l, k):
        if self.quad:
            return quadax.quadgk(self._integrand2, (0, x), (l, k))[0]
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.xgrid.xrec)
        n = jnp.arange(self.xgrid.nx)
        g = Gammainc(l/ 2 + n / 2 + 3/2, x**2)
        return jnp.sum(p * g)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_3(self, x, l, k):
        if self.quad:
            return quadax.quadgk(self._integrand3, (x, np.inf), (l, k))[0]
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.xgrid.xrec)
        n = jnp.arange(self.xgrid.nx)
        g = Gammaincc(-l/2 + n / 2 + 2, x**2)
        return jnp.sum(p * g)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_4(self, x, l, k):
        if self.quad:
            return quadax.quadgk(self._integrand4, (0, x), (l, k))[0]
        c = jnp.zeros(self.xgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.xgrid.xrec)
        n = jnp.arange(self.xgrid.nx)
        g = Gammainc(l / 2 + n / 2 + 5/2, x**2)
        return jnp.sum(p * g)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_1(self, x, l, k):
        return -self._integrand1(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_2(self, x, l, k):
        return self._integrand2(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_3(self, x, l, k):
        return -self._integrand3(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_4(self, x, l, k):
        return self._integrand4(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_1(self, x, l, k):
        return jax.grad(self._dI_1)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_2(self, x, l, k):
        return jax.grad(self._dI_2)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_3(self, x, l, k):
        return jax.grad(self._dI_3)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_4(self, x, l, k):
        return jax.grad(self._dI_4)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _Hlk(self, xb, l, k):
        term1 = 1 / xb ** (l + 1) * self._I_2(xb, l, k)
        term2 = xb**l * self._I_1(xb, l, k)
        return (4 * jnp.pi) / (2 * l + 1) * (term1 + term2)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dHlk(self, xb, l, k):
        term11 = -(l + 1) / xb ** (l + 2) * self._I_2(xb, l, k)
        term12 = 1 / xb ** (l + 1) * self._dI_2(xb, l, k)
        term21 = l * xb ** (l - 1) * self._I_1(xb, l, k)
        term22 = xb**l * self._dI_1(xb, l, k)
        return (4 * jnp.pi) / (2 * l + 1) * (term11 + term12 + term21 + term22)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _Glk(self, xb, l, k):
        term1 = xb**l * self._I_3(xb, l, k)
        term2 = -(2 * l - 1) / (2 * l + 3) * xb ** (l + 2) * self._I_1(xb, l, k)
        term3 = -(2 * l - 1) / (2 * l + 3) / xb ** (l + 1) * self._I_4(xb, l, k)
        term4 = 1 / xb ** (l - 1) * self._I_2(xb, l, k)
        return -(4 * jnp.pi) / (4 * l**2 - 1) * (term1 + term2 + term3 + term4)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dGlk(self, xb, l, k):
        term11 = l * xb ** (l - 1) * self._I_3(xb, l, k)
        term12 = xb**l * self._dI_3(xb, l, k)
        term21 = (
            -(2 * l - 1) / (2 * l + 3) * (l + 2) * xb ** (l + 1) * self._I_1(xb, l, k)
        )
        term22 = -(2 * l - 1) / (2 * l + 3) * xb ** (l + 2) * self._dI_1(xb, l, k)
        term31 = (
            (2 * l - 1) / (2 * l + 3) * (l + 1) / xb ** (l + 2) * self._I_4(xb, l, k)
        )
        term32 = -(2 * l - 1) / (2 * l + 3) / xb ** (l + 1) * self._dI_4(xb, l, k)
        term41 = -(l - 1) / xb ** (l) * self._I_2(xb, l, k)
        term42 = 1 / xb ** (l - 1) * self._dI_2(xb, l, k)
        return (
            -(4 * jnp.pi)
            / (4 * l**2 - 1)
            * (term11 + term12 + term21 + term22 + term31 + term32 + term41 + term42)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddGlk(self, xb, l, k):
        term111 = l * (l - 1) * xb ** (l - 2) * self._I_3(xb, l, k)
        term112 = l * xb ** (l - 1) * self._dI_3(xb, l, k)
        term121 = l * xb ** (l - 1) * self._dI_3(xb, l, k)
        term122 = xb**l * self._ddI_3(xb, l, k)
        term211 = (
            -(2 * l - 1)
            / (2 * l + 3)
            * (l + 2)
            * (l + 1)
            * xb ** (l)
            * self._I_1(xb, l, k)
        )
        term212 = (
            -(2 * l - 1) / (2 * l + 3) * (l + 2) * xb ** (l + 1) * self._dI_1(xb, l, k)
        )
        term221 = (
            -(2 * l - 1) / (2 * l + 3) * (l + 2) * xb ** (l + 1) * self._dI_1(xb, l, k)
        )
        term222 = -(2 * l - 1) / (2 * l + 3) * xb ** (l + 2) * self._ddI_1(xb, l, k)
        term311 = (
            -(2 * l - 1)
            / (2 * l + 3)
            * (l + 1)
            * (l + 2)
            / xb ** (l + 3)
            * self._I_4(xb, l, k)
        )
        term312 = (
            (2 * l - 1) / (2 * l + 3) * (l + 1) / xb ** (l + 2) * self._dI_4(xb, l, k)
        )
        term321 = (
            +(2 * l - 1) / (2 * l + 3) * (l + 1) / xb ** (l + 2) * self._dI_4(xb, l, k)
        )
        term322 = -(2 * l - 1) / (2 * l + 3) / xb ** (l + 1) * self._ddI_4(xb, l, k)
        term411 = l * (l - 1) / xb ** (l + 1) * self._I_2(xb, l, k)
        term412 = -(l - 1) / xb ** (l) * self._dI_2(xb, l, k)
        term421 = -(l - 1) / xb ** (l) * self._dI_2(xb, l, k)
        term422 = 1 / xb ** (l - 1) * self._ddI_2(xb, l, k)
        return (
            -(4 * jnp.pi)
            / (4 * l**2 - 1)
            * (
                term111
                + term121
                + term211
                + term221
                + term311
                + term321
                + term411
                + term421
                + term112
                + term122
                + term212
                + term222
                + term312
                + term322
                + term412
                + term422
            )
        )


class FokkerPlanckLandau(eqx.Module):
    """Fokker-Planck Landau collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    species : list[LocalMaxwellian]
        Species being considered
    xgrid : SpeedGrid
        Grid of coordinates in speed.
    xigrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    Er : float
        Radial electric field.

    """

    field: monkes.Field
    species: list[monkes._species.LocalMaxwellian]
    xgrid: SpeedGrid
    xigrid: PitchAngleGrid
    potentials: RosenbluthPotentials
    Er: float
    ntheta: int
    nzeta: int
    nxi: int
    nx: int
    ns: int

    def __init__(self, field, species, xgrid, xigrid, potentials, Er):

        self.field = field
        if not isinstance(species, (list, tuple)):
            species = [species]
        self.species = species
        self.Er = Er
        self.ntheta = field.ntheta
        self.nzeta = field.nzeta
        self.nxi = xigrid.nxi
        self.nx = xgrid.nx
        self.ns = len(species)
        self.xgrid = xgrid
        self.xigrid = xigrid
        self.potentials = potentials

    def mv(self, f):
        """Matrix vector product, action of the collision operator on f.

        Parameters
        ----------
        f : jax.Array
            Perturbed distribution function(s)

        Returns
        -------
        Cf : jax.Array
            Linearized Fokker-Planck-Landau collision operator acting on f.
        """
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._pitch_angle(f)
        out += self._energy_scattering(f)
        out += self._field_part(f)
        return out.reshape(shp)

    def _pitch_angle(self, f):
        """Pitch angle scattering part of the collision operator.

        Parameters
        ----------
        f : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Perturbed distribution function(s)

        Returns
        -------
        Cf : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Pitch angle scattering part of the collision operator acting on f.
        """
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))

        xi = self.xigrid.xi[:, None, None, None]
        x = self.xgrid.x[None, :, None, None]
        out = jnp.zeros((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        for i, spa in enumerate(self.species):
            df = self.xigrid._dfdxi(f[i])
            df *= 1 - xi**2
            ddf = self.xigrid._dfdxi(df)
            nu = 0.0
            for j, spb in enumerate(self.species):
                nu += monkes._species.nuD_ab(spa, spb, x * spa.v_thermal)
            out = out.at[i].add(nu / 2 * ddf)
        return out.reshape(shp)

    def _energy_scattering(self, f):
        """Energy scattering part of the collision operator.

        Parameters
        ----------
        f : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Perturbed distribution function(s)

        Returns
        -------
        Cf : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Energy scattering part of the collision operator acting on f.
        """
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))

        x = self.xgrid.x[None, :, None, None]
        out = jnp.zeros((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        for i, spa in enumerate(self.species):
            vta = spa.v_thermal
            v = x * vta
            dfdx = self.xgrid._dfdx(f[i])
            d2fdx2 = self.xgrid._dfdx(dfdx)
            for j, spb in enumerate(self.species):
                nupar = monkes._species.nupar_ab(spa, spb, v)
                nuD = monkes._species.nuD_ab(spa, spb, v)
                gamma = monkes._species.gamma_ab(spa, spb, v)
                ma, mb = spa.species.mass, spb.species.mass
                vtb = spb.v_thermal
                term1 = nupar * (
                    x**2 / 2 * d2fdx2 - (x * vta / vtb) ** 2 * (1 - ma / mb) * x * dfdx
                )
                term2 = nuD * x * dfdx
                term3 = 4 * jnp.pi * gamma * ma / mb * spb(v) * f[i]
                out = out.at[i].add(term1 + term2 + term3)
        return out.reshape(shp)

    def _field_part(self, f):
        """Field part of the collision operator.

        (ie, the part with the rosenbluth potentials)

        Parameters
        ----------
        f : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Perturbed distribution function(s)

        Returns
        -------
        Cf : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Field part of the collision operator acting on f.
        """
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))

        x = self.xgrid.x[None, :, None, None]
        out = jnp.zeros((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        for a, spa in enumerate(self.species):
            v = x * spa.v_thermal
            Fa = spa(v)
            for b, spb in enumerate(self.species):
                gamma = monkes._species.gamma_ab(spa, spb, v)
                fb1 = f[b]
                CG = self._CG(a, b, fb1)
                CH = self._CH(a, b, fb1)
                CD = self._CD(a, b, fb1)

                out = out.at[a].add(gamma * Fa * (CG + CH + CD))
        return out.reshape(shp)

    def _CG(self, a, b, fb1):
        x = self.xgrid.x[None, :, None, None]
        va, vb = self.species[a].v_thermal, self.species[b].v_thermal
        v = x * va
        ddG = vb**4 * self.potentials._rosenbluth_ddG(fb1, a, b)
        CG = 2 * v**2 / va**4 * ddG / vb**2
        return CG

    def _CH(self, a, b, fb1):
        x = self.xgrid.x[None, :, None, None]
        va, vb = self.species[a].v_thermal, self.species[b].v_thermal
        ma, mb = self.species[a].species.mass, self.species[b].species.mass
        v = x * va
        H = vb**2 * self.potentials._rosenbluth_H(fb1, a, b)
        dH = vb**2 * self.potentials._rosenbluth_dH(fb1, a, b)
        CH = -2 * v / va**2 * (1 - ma / mb) * dH / vb - 2 / va**2 * H
        return CH

    def _CD(self, a, b, fb1):
        va, vb = self.species[a].v_thermal, self.species[b].v_thermal
        ma, mb = self.species[a].species.mass, self.species[b].species.mass
        xa = self.xgrid.x * va / vb
        xb = self.xgrid.x
        fb1 = self.xgrid._interp(xb, fb1, xa)
        CD = 4 * jnp.pi * ma / mb * fb1
        return CD


class PitchAngleScattering(eqx.Module):
    """Pitch angle scattering collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    species : list[LocalMaxwellian]
        Species being considered
    x : jax.Array
        Values of coordinates in speed.
    xigrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    Er : float
        Radial electric field.

    """

    field: monkes.Field
    species: list[monkes._species.LocalMaxwellian]
    x: jax.Array
    xigrid: PitchAngleGrid
    ntheta: int
    nzeta: int
    nxi: int
    nx: int
    ns: int

    def __init__(self, field, species, x, xigrid):

        self.field = field
        if not isinstance(species, (list, tuple)):
            species = [species]
        self.species = species
        self.ntheta = field.ntheta
        self.nzeta = field.nzeta
        self.x = jnp.atleast_1d(x)
        self.xigrid = xigrid
        self.nxi = xigrid.nxi
        self.nx = self.x.size
        self.ns = len(species)

    def mv(self, f):
        """Matrix vector product, action of the collision operator on f.

        Parameters
        ----------
        f : jax.Array
            Perturbed distribution function(s)

        Returns
        -------
        Cf : jax.Array
            Linearized Fokker-Planck-Landau collision operator acting on f.
        """
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))
        out = self._pitch_angle(f)
        return out.reshape(shp)

    def _pitch_angle(self, f):
        """Pitch angle scattering part of the collision operator.

        Parameters
        ----------
        f : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Perturbed distribution function(s)

        Returns
        -------
        Cf : jax.Array, shape(ns, nxi, nx, ntheta, nzeta)
            Pitch angle scattering part of the collision operator acting on f.
        """
        shp = f.shape
        f = f.reshape((self.ns, self.nxi, self.nx, self.ntheta, self.nzeta))

        x = self.x[None, :, None, None]

        fk = jnp.einsum("ki,sixtz->skxtz", self.xigrid.xivander_inv, f)
        k = jnp.arange(self.xigrid.nxi)[:, None, None, None]
        for i, spa in enumerate(self.species):
            nu = 0.0
            for j, spb in enumerate(self.species):
                nu += monkes._species.nuD_ab(spa, spb, x * spa.v_thermal)
            fk = fk.at[i].multiply(nu / 2 * k * (k + 1))
        Lf = jnp.einsum("ik,skxtz->sixtz", self.xigrid.xivander, fk)
        return Lf.reshape(shp)


def pitch_angle_collisions(field, speedgrid, pitchgrid, species, approx_rdot=False):
    """Pitch angle collision operator."""
    Is = cola.ops.Identity((len(species), len(species)), pitchgrid.xi.dtype)
    Ix = cola.ops.Dense(speedgrid.xvander)
    It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
    Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
    x = speedgrid.x
    L = cola.ops.Dense(pitchgrid.L)

    nus = []
    for spa in species:
        nu = 0.0
        for spb in species:
            nu += monkes._species.nuD_ab(spa, spb, x * spa.v_thermal)
        nus.append(nu / 2)
    nus = cola.ops.Diagonal(jnp.array(nus).flatten()) @ cola.ops.Kronecker(Is, Ix)
    if approx_rdot:
        return cola.ops.Kronecker(nus, L, It, Iz)
    return cola.ops.Kronecker(nus, L, cola.ops.Kronecker(It, Iz))


def energy_scattering(field, speedgrid, pitchgrid, species, approx_rdot=False):
    """Energy scattering part of collision operator."""
    Ix = cola.ops.Dense(speedgrid.xvander)
    Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
    It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
    Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
    Dx = cola.ops.Dense(speedgrid.xvander @ speedgrid.Dx)
    D2x = cola.ops.Dense(speedgrid.xvander @ speedgrid.Dx @ speedgrid.Dx)
    x = speedgrid.x

    out = []
    for spa in species:
        vta = spa.v_thermal
        v = x * vta
        term1 = 0.0
        term2 = 0.0
        term3 = 0.0
        for spb in species:
            nupar = monkes._species.nupar_ab(spa, spb, v)
            nuD = monkes._species.nuD_ab(spa, spb, v)
            gamma = monkes._species.gamma_ab(spa, spb, v)
            ma, mb = spa.species.mass, spb.species.mass
            vtb = spb.v_thermal
            term1 += nupar * x**2 / 2
            term2 += nuD * x - nupar * (x * vta / vtb) ** 2 * (1 - ma / mb) * x
            term3 += 4 * jnp.pi * gamma * ma / mb * spb(v)
        out.append(
            cola.ops.Diagonal(term1) @ D2x
            + cola.ops.Diagonal(term2) @ Dx
            + cola.ops.Diagonal(term3) @ Ix
        )
    out = cola.ops.BlockDiag(*out)
    if approx_rdot:
        return cola.ops.Kronecker(out, Ixi, It, Iz)
    return cola.ops.Kronecker(out, Ixi, cola.ops.Kronecker(It, Iz))
