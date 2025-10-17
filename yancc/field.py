"""Magnetic field data structures."""

import functools
from typing import Optional

import equinox as eqx
import interpax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int


class Field(eqx.Module):
    """Magnetic field on a flux surface.

    Field is given as 2D arrays uniformly spaced in arbitrary
    poloidal and toroidal angles.

    Parameters
    ----------
    rho : float
        Flux surface label.
    B_sup_t : jax.Array, shape(ntheta, nzeta)
        B^theta, contravariant poloidal component of field.
    B_sup_z : jax.Array, shape(ntheta, nzeta)
        B^zeta, contravariant toroidal component of field.
    B_sub_t : jax.Array, shape(ntheta, nzeta)
        B_theta, covariant poloidal component of field.
    B_sub_z : jax.Array, shape(ntheta, nzeta)
        B_zeta, covariant toroidal component of field.
    Bmag : jax.Array, shape(ntheta, nzeta)
        Magnetic field magnitude.
    sqrtg : jax.Array, shape(ntheta, nzeta)
        Coordinate jacobian determinant from (psi, theta, zeta) to (R, phi, Z).
    psi_r : float
        Derivative of toroidal flux wrt minor radius (rho*a_minor)
    R_major : float
        Major radius.
    a_minor : float
        Minor radius.
    NFP : int
        Number of field periods.
    """

    # note: assumes (psi, theta, zeta) coordinates, not (rho, theta, zeta)
    rho: Float[Array, ""]
    theta: Float[Array, "ntheta "]
    zeta: Float[Array, "nzeta "]
    wtheta: Float[Array, "ntheta "]
    wzeta: Float[Array, "nzeta "]
    B_sup_t: Float[Array, "ntheta nzeta"]
    B_sup_z: Float[Array, "ntheta nzeta"]
    B_sub_t: Float[Array, "ntheta nzeta"]
    B_sub_z: Float[Array, "ntheta nzeta"]
    sqrtg: Float[Array, "ntheta nzeta"]
    Bmag: Float[Array, "ntheta nzeta"]
    bdotgradB: Float[Array, "ntheta nzeta"]
    BxgradpsidotgradB: Float[Array, "ntheta nzeta"]
    dBdt: Float[Array, "ntheta nzeta"]
    dBdz: Float[Array, "ntheta nzeta"]
    Bmag_fsa: Float[Array, ""]
    B2mag_fsa: Float[Array, ""]
    psi_r: Float[Array, ""]
    R_major: Float[Array, ""]
    a_minor: Float[Array, ""]
    iota: Float[Array, ""]
    B0: Float[Array, ""]
    ntheta: int = eqx.field(static=True)
    nzeta: int = eqx.field(static=True)
    NFP: Int[Array, ""]

    def __init__(
        self,
        rho: Float[ArrayLike, ""],
        B_sup_t: Float[ArrayLike, "ntheta nzeta"],
        B_sup_z: Float[ArrayLike, "ntheta nzeta"],
        B_sub_t: Float[ArrayLike, "ntheta nzeta"],
        B_sub_z: Float[ArrayLike, "ntheta nzeta"],
        Bmag: Float[ArrayLike, "ntheta nzeta"],
        sqrtg: Float[ArrayLike, "ntheta nzeta"],
        psi_r: Float[ArrayLike, ""],
        iota: Float[ArrayLike, ""],
        R_major: Float[ArrayLike, ""],
        a_minor: Float[ArrayLike, ""],
        NFP: Int[ArrayLike, ""] = 1,
        *,
        dBdt: Optional[Float[ArrayLike, "ntheta nzeta"]] = None,
        dBdz: Optional[Float[ArrayLike, "ntheta nzeta"]] = None,
        B0: Optional[Float[ArrayLike, ""]] = None,
    ):
        self.rho = jnp.asarray(rho)
        self.NFP = jnp.asarray(NFP)
        self.B_sup_t = jnp.asarray(B_sup_t)
        self.B_sup_z = jnp.asarray(B_sup_z)
        self.B_sub_t = jnp.asarray(B_sub_t)
        self.B_sub_z = jnp.asarray(B_sub_z)
        self.sqrtg = jnp.asarray(sqrtg)
        self.Bmag = jnp.asarray(Bmag)
        self.ntheta = self.sqrtg.shape[0]
        self.nzeta = self.sqrtg.shape[1]
        assert (self.ntheta % 2 == 1) and (
            self.nzeta % 2 == 1
        ), "ntheta and nzeta must be odd"
        if dBdt is None:
            dBdt = self._dfdt(self.Bmag)
        if dBdz is None:
            dBdz = self._dfdz(self.Bmag)
        self.dBdt = jnp.asarray(dBdt)
        self.dBdz = jnp.asarray(dBdz)
        if B0 is None:
            B0 = self.Bmag.mean()
        self.B0 = jnp.asarray(B0)
        self.bdotgradB = (
            self.B_sup_t * self.dBdt + self.B_sup_z * self.dBdz
        ) / self.Bmag
        self.BxgradpsidotgradB = (
            self.B_sub_z * self.dBdt - self.B_sub_t * self.dBdz
        ) / self.sqrtg
        self.Bmag_fsa = self.flux_surface_average(self.Bmag)
        self.B2mag_fsa = self.flux_surface_average(self.Bmag**2)
        self.psi_r = jnp.asarray(psi_r)
        self.iota = jnp.asarray(iota)
        self.R_major = jnp.asarray(R_major)
        self.a_minor = jnp.asarray(a_minor)
        self.theta = jnp.linspace(0, 2 * np.pi, self.ntheta, endpoint=False)
        self.zeta = jnp.linspace(0, 2 * np.pi / NFP, self.nzeta, endpoint=False)
        self.wtheta = jnp.diff(self.theta, append=jnp.array([2 * jnp.pi]))
        self.wzeta = jnp.diff(self.zeta, append=jnp.array([2 * jnp.pi / self.NFP]))

    @classmethod
    def from_desc(
        cls,
        eq,
        rho: Float[ArrayLike, ""],
        ntheta: int,
        nzeta: int,
    ) -> "Field":
        """Construct Field from DESC equilibrium.

        Parameters
        ----------
        eq : desc.equilibrium.Equilibrium
            DESC Equilibrium.
        rho : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
            Both must be odd.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"

        from desc.grid import LinearGrid  # pyright: ignore[reportMissingImports]

        grid = LinearGrid(rho=rho, theta=ntheta, zeta=nzeta, endpoint=False, NFP=eq.NFP)
        keys = [
            "B^theta",
            "B^zeta",
            "B_theta",
            "B_zeta",
            "|B|",
            "|B|_t",
            "|B|_z",
            "sqrt(g)",
            "psi_r",
            "iota",
            "a",
            "R0",
        ]
        desc_data = eq.compute(keys, grid=grid)

        data = {
            "B_sup_t": desc_data["B^theta"],
            "B_sup_z": desc_data["B^zeta"],
            "B_sub_t": desc_data["B_theta"],
            "B_sub_z": desc_data["B_zeta"],
            "Bmag": desc_data["|B|"],
            "dBdt": desc_data["|B|_t"],
            "dBdz": desc_data["|B|_z"],
            "sqrtg": desc_data["sqrt(g)"] / desc_data["psi_r"],
        }

        data = {
            key: val.reshape((grid.num_theta, grid.num_zeta), order="F")
            for key, val in data.items()
        }
        return cls(
            rho=rho,
            psi_r=desc_data["psi_r"][0] / desc_data["a"],
            iota=desc_data["iota"][0],
            R_major=desc_data["R0"],
            a_minor=desc_data["a"],
            **data,
            NFP=eq.NFP,
        )

    @classmethod
    def from_vmec(cls, wout, s: float, ntheta: int, nzeta: int):
        """Construct Field from VMEC equilibrium.

        Parameters
        ----------
        wout : path-like
            Path to vmec wout file.
        s : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        """
        raise NotImplementedError

    @classmethod
    def from_booz_xform(
        cls,
        booz,
        s: Float[ArrayLike, ""],
        ntheta: int,
        nzeta: int,
        cutoff: Float[ArrayLike, ""] = 0.0,
    ) -> "Field":
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        booz : path-like
            Path to booz_xform wout file.
        s : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        cutoff : float
            Modes with abs(b_mn) < cutoff * abs(b_00) will be excluded.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"
        from netCDF4 import Dataset

        file = Dataset(booz, mode="r")
        assert not bool(
            file.variables["lasym__logical__"][:].filled()
        ), "non-symmetric booz-xform not supported"

        ns = file.variables["ns_b"][:].filled()
        nfp = file.variables["nfp_b"][:].filled()

        theta = jnp.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta = jnp.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)

        s_full = np.linspace(0, 1, ns)
        hs = 1 / (ns - 1)
        s_half = s_full[:-1] + hs / 2

        aspect = file.variables["aspect_b"][:].filled()
        b_mnc = file.variables["bmnc_b"][:].filled()
        r_mnc = file.variables["rmnc_b"][:].filled()
        nfp = int(file.variables["nfp_b"][:].filled())
        iota = file.variables["iota_b"][:].filled()
        buco = file.variables["buco_b"][:].filled()  # (AKA Boozer I)
        bvco = file.variables["bvco_b"][:].filled()  # (AKA Boozer G)
        jlist = file.variables["jlist"][:].filled()
        Psi = file.variables["phi_b"][:].filled()[-1]

        R0 = r_mnc[1, 0]
        a_minor = R0 / aspect

        # jlist = 2 + indices of half grid where boozer transform was computed
        b_mnc = interpax.interp1d(s, s_half[jlist - 2], b_mnc)
        # profiles are on half grid, but with an extra 0 at the beginning bc
        # the world is awful.
        buco = -interpax.interp1d(s, s_half, buco[1:])  # sign flip LH -> RH
        bvco = interpax.interp1d(s, s_half, bvco[1:])
        iota = -interpax.interp1d(s, s_half, iota[1:])  # sign flip LH -> RH

        xm = file.variables["ixm_b"][:].filled()
        xn = file.variables["ixn_b"][:].filled()

        B0 = jnp.abs(b_mnc).max()
        mask = jnp.abs(b_mnc) > cutoff * B0

        # booz_xform uses (m*t - n*z) instead of vmecs (m*t + n*z)
        Bmag = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn)
        dBdt = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dt=1)
        dBdz = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dz=1)

        sign = jnp.sign(bvco + iota * buco)
        buco *= sign
        bvco *= sign
        sqrtg = (bvco + iota * buco) / Bmag**2
        data = {}
        data["sqrtg"] = sqrtg
        data["Bmag"] = Bmag
        data["dBdt"] = dBdt
        data["dBdz"] = dBdz
        data["B_sub_t"] = buco * jnp.ones((ntheta, nzeta))
        data["B_sub_z"] = bvco * jnp.ones((ntheta, nzeta))
        data["B_sup_t"] = iota / sqrtg
        data["B_sup_z"] = 1 / sqrtg
        # d psi/drho = d Psi rho^2 / drho = 2 Psi rho;  r = rho*a
        # d psi / dr = d psi / drho / a
        data["psi_r"] = 2 * Psi * jnp.sqrt(s) / a_minor
        data["iota"] = iota
        data["B0"] = B0
        data["R_major"] = R0
        data["a_minor"] = a_minor
        return cls(rho=jnp.sqrt(s), **data, NFP=nfp)

    @functools.partial(jnp.vectorize, signature="(m,n)->()", excluded=[0])
    def flux_surface_average(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, ""]:
        """Compute flux surface average of f."""
        f = f.reshape((self.ntheta, self.nzeta))
        g = f * self.sqrtg
        return g.mean() / self.sqrtg.mean()

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def bdotgrad(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        """𝐛 ⋅ ∇ f."""
        return (self.B_sup_t * self._dfdt(f) + self.B_sup_z * self._dfdz(f)) / self.Bmag

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def Bxgradpsidotgrad(
        self, f: Float[Array, "ntheta nzeta"]
    ) -> Float[Array, "ntheta nzeta"]:
        """𝐁 × ∇ ψ ⋅ ∇ f."""
        return (
            self.B_sub_z * self._dfdt(f) - self.B_sub_t * self._dfdz(f)
        ) / self.sqrtg

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def _dfdt(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        g = jnp.fft.fft(f, axis=0)
        k = jnp.fft.fftfreq(self.ntheta, 1 / self.ntheta)
        df = jnp.fft.ifft(1j * k[:, None] * g, axis=0)
        return df.real

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def _dfdz(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        g = jnp.fft.fft(f, axis=1)
        k = jnp.fft.fftfreq(self.nzeta, 1 / self.nzeta) * self.NFP
        df = jnp.fft.ifft(1j * k[None, :] * g, axis=1)
        return df.real

    def resample(self, ntheta: int, nzeta: int) -> "Field":
        """Resample field to a higher resolution."""
        if ntheta == self.ntheta and nzeta == self.nzeta:
            return self

        keys = [
            "B_sup_t",
            "B_sup_z",
            "B_sub_t",
            "B_sub_z",
            "sqrtg",
            "Bmag",
            "dBdt",
            "dBdz",
        ]
        out = {}
        for key in keys:
            out[key] = interpax.fft_interp2d(getattr(self, key), ntheta, nzeta)
        return Field(
            rho=self.rho,
            **out,
            psi_r=self.psi_r,
            B0=self.B0,
            iota=self.iota,
            R_major=self.R_major,
            a_minor=self.a_minor,
            NFP=self.NFP,
        )


def vmec_eval(t, z, xc, xs, m, n, dt=0, dz=0):
    """Evaluate a vmec style double-fourier series.

    eg sum_mn xc*cos(m*t-n*z) + xs*sin(m*t-n*z)

    Parameters
    ----------
    t, z : float, jax.Array
        theta, zeta coordinates to evaluate at.
    xc, xs : jax.Array
        Cosine, sine coefficients of double fourier series.
    m, n : jax.Array
        Poloidal and toroidal mode numbers.

    Returns
    -------
    x : float, jax.Array
        Evaluated quantity at t, z.
    """
    xc, xs, m, n, dt, dz = jnp.atleast_1d(xc, xs, m, n, dt, dz)
    xc, xs, m, n, dt, dz = jnp.broadcast_arrays(xc, xs, m, n, dt, dz)
    return _vmec_eval(t, z, xc, xs, m, n, dt, dz)


@functools.partial(jnp.vectorize, signature="(),(),(n),(n),(n),(n),(n),(n)->()")
def _vmec_eval(t, z, xc, xs, m, n, dt, dz):
    arg = m * t - n * z
    arg += dt * jnp.pi / 2
    arg -= dz * jnp.pi / 2
    xc *= m**dt
    xc *= n**dz
    xs *= m**dt
    xs *= n**dz
    c = (xc * jnp.cos(arg)).sum()
    s = (xs * jnp.sin(arg)).sum()
    return c + s
