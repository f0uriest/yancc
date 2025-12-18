"""Magnetic field data structures."""

import functools
from typing import Optional

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int
from scipy.constants import mu_0


class Field(eqx.Module):
    """Magnetic field on a flux surface.

    Field is given as 2D arrays uniformly spaced in arbitrary
    poloidal and toroidal angles. The radial coordinate is assumed to be
    rho = sqrt(normalized toroidal flux)

    Parameters
    ----------
    rho : float
        Normalized surface label = sqrt(s).
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
        Coordinate jacobian determinant from (rho, theta, zeta) to (R, phi, Z). Units
        of m^3
    Psi : float
        Total toroidal flux within the LCFS (in Webers, not divided by 2pi).
    iota : float
        Rotational transform.
    R_major : float
        Major radius.
    a_minor : float
        Minor radius.
    NFP : int
        Number of field periods.
    dBdt, dBdz : jax.Array, shape(ntheta, nzeta), optional
        Derivative of Bmag with respect to theta/zeta. Default is to compute with fft.
    B0 : float, optional
        Characteristic scale for magnetic field. Default is surface average of B.
    """

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
    BxgradrhodotgradB: Float[Array, "ntheta nzeta"]
    dBdt: Float[Array, "ntheta nzeta"]
    dBdz: Float[Array, "ntheta nzeta"]
    Bmag_fsa: Float[Array, ""]
    B2mag_fsa: Float[Array, ""]
    psi_r: Float[Array, ""]
    psi_rho: Float[Array, ""]
    Psi: Float[Array, ""]
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
        Psi: Float[ArrayLike, ""],
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
        self.BxgradrhodotgradB = (
            self.B_sub_z * self.dBdt - self.B_sub_t * self.dBdz
        ) / self.sqrtg
        self.Bmag_fsa = self.flux_surface_average(self.Bmag)
        self.B2mag_fsa = self.flux_surface_average(self.Bmag**2)
        self.Psi = jnp.asarray(Psi)
        self.iota = jnp.asarray(iota)
        self.R_major = jnp.asarray(R_major)
        self.a_minor = jnp.asarray(a_minor)
        self.psi_rho = jnp.asarray(self.Psi / np.pi * self.rho)
        self.psi_r = self.psi_rho / self.a_minor
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
            Normalized surface label = sqrt(s).
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
            "sqrtg": desc_data["sqrt(g)"],
        }

        data = {
            key: val.reshape((grid.num_theta, grid.num_zeta), order="F")
            for key, val in data.items()
        }

        data["Psi"] = eq.Psi
        data["a_minor"] = desc_data["a"]
        data["R_major"] = desc_data["R0"]
        data["iota"] = desc_data["iota"][0]

        return cls(
            rho=rho,
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
            Normalized surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"
        from netCDF4 import Dataset

        file = Dataset(wout, mode="r")

        ns = file.variables["ns"][:].filled()
        nfp = file.variables["nfp"][:].filled()
        theta = 2 * np.pi - jnp.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta = jnp.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
        assert "bmns" not in file.variables, "non-symmetric vmec not supported"

        s_full = jnp.linspace(0, 1, ns)
        hs = 1 / (ns - 1)
        s_half = s_full[0:-1] + hs / 2

        a_minor = file.variables["Aminor_p"][:].filled()
        R_major = file.variables["Rmajor_p"][:].filled()
        g_mnc = file.variables["gmnc"][:].filled()
        b_mnc = file.variables["bmnc"][:].filled()
        bsupu_mnc = file.variables["bsupumnc"][:].filled()
        bsupv_mnc = file.variables["bsupvmnc"][:].filled()
        bsubu_mnc = file.variables["bsubumnc"][:].filled()
        bsubv_mnc = file.variables["bsubvmnc"][:].filled()

        nfp = file.variables["nfp"][:].filled()
        iota = file.variables["iotaf"][:].filled()
        phi = file.variables["phi"][:].filled()[-1]  # total flux

        # assuming the field is only over a single flux surface s
        g_mnc = interpax.interp1d(s, s_half, g_mnc[1:, :])
        b_mnc = interpax.interp1d(s, s_half, b_mnc[1:, :])
        bsupu_mnc = interpax.interp1d(s, s_half, bsupu_mnc[1:, :])
        bsupv_mnc = interpax.interp1d(s, s_half, bsupv_mnc[1:, :])
        bsubu_mnc = interpax.interp1d(s, s_half, bsubu_mnc[1:, :])
        bsubv_mnc = interpax.interp1d(s, s_half, bsubv_mnc[1:, :])
        iota = interpax.interp1d(s, s_full, iota)

        xm = file.variables["xm_nyq"][:].filled()
        xn = file.variables["xn_nyq"][:].filled()

        sqrtg = vmec_eval(theta[:, None], zeta[None, :], g_mnc, 0, xm, xn)
        Bmag = vmec_eval(theta[:, None], zeta[None, :], b_mnc, 0, xm, xn)
        B_sub_t = vmec_eval(theta[:, None], zeta[None, :], bsubu_mnc, 0, xm, xn)
        B_sub_z = vmec_eval(theta[:, None], zeta[None, :], bsubv_mnc, 0, xm, xn)
        B_sup_t = vmec_eval(theta[:, None], zeta[None, :], bsupu_mnc, 0, xm, xn)
        B_sup_z = vmec_eval(theta[:, None], zeta[None, :], bsupv_mnc, 0, xm, xn)

        B0 = jnp.abs(b_mnc).max()
        dBdt = vmec_eval(theta[:, None], zeta[None, :], b_mnc, 0, xm, xn, dt=1)
        dBdz = vmec_eval(theta[:, None], zeta[None, :], b_mnc, 0, xm, xn, dz=1)

        sign = file.variables["signgs"][:].filled()
        sqrtg *= sign
        # theta is in opposite direction
        B_sub_t *= sign
        B_sup_t *= sign
        dBdt *= sign
        iota *= sign

        data = {}
        data["sqrtg"] = sqrtg * 2 * jnp.sqrt(s)
        data["Bmag"] = Bmag
        data["dBdt"] = dBdt
        data["dBdz"] = dBdz
        data["B_sub_t"] = B_sub_t
        data["B_sub_z"] = B_sub_z
        data["B_sup_t"] = B_sup_t
        data["B_sup_z"] = B_sup_z
        data["Psi"] = phi
        data["iota"] = iota
        data["B0"] = B0
        data["R_major"] = R_major
        data["a_minor"] = a_minor

        return cls(rho=jnp.sqrt(s), **data, NFP=nfp)

    @classmethod
    def from_booz_xform(
        cls,
        booz: str,
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
            Normalized surface label.
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
        g_mnc = file.variables["gmn_b"][:].filled()
        nfp = int(file.variables["nfp_b"][:].filled())
        iota = file.variables["iota_b"][:].filled()
        buco = file.variables["buco_b"][:].filled()  # (AKA Boozer I)
        bvco = file.variables["bvco_b"][:].filled()  # (AKA Boozer G)
        jlist = file.variables["jlist"][:].filled()
        Psi = file.variables["phi_b"][:].filled()[-1]

        xm = file.variables["ixm_b"][:].filled()
        xn = file.variables["ixn_b"][:].filled()

        # need to do volume integrals to get R0 to match desc/vmec
        R = jax.vmap(lambda x: vmec_eval(theta[:, None], zeta[None, :], x, 0, xm, -xn))(
            r_mnc
        )
        g = jax.vmap(lambda x: vmec_eval(theta[:, None], zeta[None, :], x, 0, xm, -xn))(
            g_mnc
        )
        h = (2 * np.pi / ntheta) * (2 * np.pi / nzeta) * (Psi / 2 / np.pi * hs)
        V = (g * h).sum()
        h = (2 * np.pi / ntheta) * (Psi / 2 / np.pi * hs)
        A = (g / R * h).sum((0, 1)).mean()

        R0 = V / (2 * np.pi * A)
        a_minor = R0 / aspect

        # jlist = 2 + indices of half grid where boozer transform was computed
        b_mnc = interpax.interp1d(s, s_half[jlist - 2], b_mnc)
        # profiles are on half grid, but with an extra 0 at the beginning bc
        # the world is awful.
        buco = -interpax.interp1d(s, s_half, buco[1:])  # sign flip LH -> RH
        bvco = interpax.interp1d(s, s_half, bvco[1:])
        iota = -interpax.interp1d(s, s_half, iota[1:])  # sign flip LH -> RH

        B0 = jnp.abs(b_mnc).max()
        mask = jnp.abs(b_mnc) > cutoff * B0

        # booz_xform uses (m*t - n*z) instead of vmecs (m*t + n*z)
        Bmag = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn)
        dBdt = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dt=1)
        dBdz = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dz=1)

        # make jacobian positive
        sign = jnp.sign(bvco + iota * buco)
        buco *= sign
        bvco *= sign

        return cls.from_boozer(
            rho=jnp.sqrt(s),
            Bmag=Bmag,
            I=buco,
            G=bvco,
            iota=iota,
            Psi=Psi,
            a_minor=a_minor,
            R_major=R0,
            NFP=nfp,
            dBdt=dBdt,
            dBdz=dBdz,
            B0=B0,
        )

    @classmethod
    def from_ipp_bc(
        cls,
        path: str,
        s: Float[ArrayLike, ""],
        ntheta: int,
        nzeta: int,
        cutoff: Float[ArrayLike, ""] = 0.0,
    ) -> "Field":
        """Construct Field from IPP format bc file.

        Parameters
        ----------
        path : path-like
            Path to input bc file.
        s : float
            Normalized surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        cutoff : float
            Modes with abs(b_mn) < cutoff * abs(b_00) will be excluded.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"

        data = read_bc(path)
        nfp = data["nfp"]
        theta = jnp.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta = jnp.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)

        s_grid = data["s"]

        R0 = data["Rvol"]
        a_minor = data["avol"]

        b_mnc = interpax.interp1d(s, s_grid, data["Bmn"])
        I = interpax.interp1d(s, s_grid, data["I"])
        G = interpax.interp1d(s, s_grid, data["G"])
        iota = interpax.interp1d(s, s_grid, data["iota"])

        xm, xn = np.meshgrid(data["m"], data["n"], indexing="ij")
        xm, xn = xm.flatten(), xn.flatten()

        B0 = jnp.abs(b_mnc).max()
        mask = jnp.abs(b_mnc) > cutoff * B0

        b_mnc = b_mnc.flatten()
        mask = mask.flatten()
        # booz_xform uses (m*t - n*z) instead of vmecs (m*t + n*z)
        Bmag = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn)
        dBdt = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dt=1)
        dBdz = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dz=1)

        # make jacobian positive
        sign = jnp.sign(G + iota * I)
        I *= sign
        G *= sign

        return cls.from_boozer(
            rho=jnp.sqrt(s),
            Bmag=Bmag,
            I=I,
            G=G,
            iota=iota,
            Psi=data["Psi"],
            a_minor=a_minor,
            R_major=R0,
            NFP=nfp,
            dBdt=dBdt,
            dBdz=dBdz,
            B0=B0,
        )

    @classmethod
    def from_boozer(
        cls,
        rho: Float[ArrayLike, ""],
        Bmag: Float[ArrayLike, "ntheta nzeta"],
        I: Float[ArrayLike, ""],
        G: Float[ArrayLike, ""],
        iota: Float[ArrayLike, ""],
        Psi: Float[ArrayLike, ""],
        R_major: Float[ArrayLike, ""],
        a_minor: Float[ArrayLike, ""],
        NFP: Int[ArrayLike, ""] = 1,
        *,
        dBdt: Optional[Float[ArrayLike, "ntheta nzeta"]] = None,
        dBdz: Optional[Float[ArrayLike, "ntheta nzeta"]] = None,
        B0: Optional[Float[ArrayLike, ""]] = None,
    ):
        """Construct a field in Boozer coordinates.

        Parameters
        ----------
        rho : float
            Normalized surface label = sqrt(s).
        Bmag : jax.Array, shape(ntheta, nzeta)
            Magnetic field magnitude in uniformly spaced Boozer angles.
        I, G : float
            Boozer toroidal and poloidal currents, in T*m.
        iota : float
            Rotational transform.
        Psi : float
            Total flux through LCFS in webers.
        R_major : float
            Major radius.
        a_minor : float
            Minor radius.
        NFP : int
            Number of field periods.
        dBdt, dBdz : jax.Array, shape(ntheta, nzeta), optional
            Derivative of Bmag with respect to theta/zeta. Default is to compute with
            fft.
        B0 : float, optional
            Characteristic scale for magnetic field. Default is surface average of B.
        """
        sqrtg = (G + iota * I) / Bmag**2
        data = {}
        data["sqrtg"] = sqrtg * Psi * rho / np.pi
        data["Bmag"] = Bmag
        data["dBdt"] = dBdt
        data["dBdz"] = dBdz
        data["B_sub_t"] = I * jnp.ones_like(Bmag)
        data["B_sub_z"] = G * jnp.ones_like(Bmag)
        data["B_sup_t"] = iota / sqrtg
        data["B_sup_z"] = 1 / sqrtg
        data["Psi"] = Psi
        data["iota"] = iota
        data["B0"] = B0
        data["R_major"] = R_major
        data["a_minor"] = a_minor
        return cls(rho=rho, **data, NFP=NFP)

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
    def Bxgradrhodotgrad(
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
            Psi=self.Psi,
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


def _strip_comments(lines):
    lines = lines.copy()
    idxs = []
    for i, line in enumerate(lines):
        if line.startswith("CC"):
            idxs.append(i)
    for i in idxs[::-1]:  # iterate backwards here so we don't mess up idxs as we delete
        del lines[i]
    return lines


def _read_globals(lines):
    assert (
        lines[0].strip().startswith("m0b")
    ), "Error in reading global parameters of bc file"
    lines = lines.copy()
    dat = np.genfromtxt(lines[:2], skip_header=1)
    data = {}
    data["M"] = int(dat[0])
    data["N"] = int(dat[1])
    data["ns"] = int(dat[2])
    data["nfp"] = int(dat[3])
    data["Psi"] = -float(dat[4])
    data["a"] = float(dat[5])
    data["R"] = float(dat[6])
    if len(dat) > 7:
        data["avol"] = float(dat[7])
        data["Rvol"] = float(dat[8])
    else:
        data["avol"] = data["a"]
        data["Rvol"] = data["R"]
    return data, lines[2:]


def _split_by_surface(lines):
    lines = lines.copy()
    assert lines[0].replace(" ", "").startswith("siota")

    blocks = []
    thisblock = []
    for line in lines:
        if line.replace(" ", "").startswith("siota"):
            # start of new block
            if len(thisblock):
                blocks.append(thisblock)
            thisblock = [line]
        else:
            thisblock.append(line)
    if len(thisblock):
        blocks.append(thisblock)
    for block in blocks:
        assert len(block) > 0
        assert block[0].replace(" ", "").startswith("siota")
    return blocks


def _parse_surface(block):
    surf_dat = np.genfromtxt(block[:3], skip_header=2)
    surf_data = {}
    surf_data["s"] = surf_dat[0]
    surf_data["iota"] = -surf_dat[1]
    surf_data["G"] = surf_dat[2]
    surf_data["I"] = -surf_dat[3]
    surf_data["pprime"] = surf_dat[4]
    surf_data["dV/ds"] = surf_dat[5]

    surf_dat = np.genfromtxt(block[3:], skip_header=1)
    surf_data["m"] = surf_dat[:, 0].astype(int)
    surf_data["n"] = surf_dat[:, 1].astype(int)
    surf_data["Rmn"] = surf_dat[:, 2]
    surf_data["Zmn"] = surf_dat[:, 3]
    surf_data["numn"] = surf_dat[:, 4]
    surf_data["Bmn"] = surf_dat[:, 5]
    return surf_data


def _combine_surf_data(surf_data, global_data):
    assert len(surf_data) == global_data["ns"]

    all_data = {
        "s": np.zeros(global_data["ns"]),
        "iota": np.zeros(global_data["ns"]),
        "I": np.zeros(global_data["ns"]),
        "G": np.zeros(global_data["ns"]),
        "pprime": np.zeros(global_data["ns"]),
        "dV/ds": np.zeros(global_data["ns"]),
        "m": np.arange(0, global_data["M"] + 1),
        "n": np.fft.ifftshift(np.arange(-global_data["N"], global_data["N"] + 1))
        * global_data["nfp"],
        "Bmn": np.zeros(
            (global_data["ns"], global_data["M"] + 1, 2 * global_data["N"] + 1)
        ),
        "Rmn": np.zeros(
            (global_data["ns"], global_data["M"] + 1, 2 * global_data["N"] + 1)
        ),
        "Zmn": np.zeros(
            (global_data["ns"], global_data["M"] + 1, 2 * global_data["N"] + 1)
        ),
        "numn": np.zeros(
            (global_data["ns"], global_data["M"] + 1, 2 * global_data["N"] + 1)
        ),
    }
    for i, surf in enumerate(surf_data):
        all_data["s"][i] = surf["s"]
        all_data["iota"][i] = surf["iota"]
        all_data["I"][i] = surf["I"]
        all_data["G"][i] = surf["G"]
        all_data["pprime"][i] = surf["pprime"]
        all_data["dV/ds"][i] = surf["dV/ds"]
        all_data["Bmn"][i][surf["m"], surf["n"]] = surf["Bmn"]
        all_data["Rmn"][i][surf["m"], surf["n"]] = surf["Rmn"]
        all_data["Zmn"][i][surf["m"], surf["n"]] = surf["Zmn"]
        all_data["numn"][i][surf["m"], surf["n"]] = surf["numn"]

    # ensure its sorted in case we screwed something up
    idx = np.argsort(all_data["s"])
    all_data["s"] = all_data["s"][idx]
    all_data["iota"] = all_data["iota"][idx]
    all_data["I"] = all_data["I"][idx] * mu_0 / (2 * np.pi)
    all_data["G"] = all_data["G"][idx] * mu_0 / (2 * np.pi) * global_data["nfp"]
    all_data["Bmn"] = all_data["Bmn"][idx]
    all_data["Rmn"] = all_data["Rmn"][idx]
    all_data["Zmn"] = all_data["Zmn"][idx]
    all_data["numn"] = all_data["numn"][idx]

    all_data.update(global_data)
    return all_data


def read_bc(path):
    """Read an IPP boozer.bc file as a dict of ndarray."""
    lines = open(path).readlines()
    lines = _strip_comments(lines)
    global_data, lines = _read_globals(lines)
    surf_data = _split_by_surface(lines)
    surf_data = [_parse_surface(block) for block in surf_data]
    all_data = _combine_surf_data(surf_data, global_data)
    return all_data
