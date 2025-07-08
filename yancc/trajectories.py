"""Drift Kinetic Operators without collisions."""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float

from .collisions import (
    FokkerPlanckLandau,
    MDKEPitchAngleScattering,
    RosenbluthPotentials,
)
from .field import Field
from .finite_diff import fd_coeffs, fdbwd, fdfwd
from .species import LocalMaxwellian
from .utils import _parse_axorder_shape_3d, _parse_axorder_shape_4d
from .velocity_grids import SpeedGrid, UniformPitchAngleGrid


def dkes_w_theta(
    field: Field, pitchgrid: UniformPitchAngleGrid, E_psi: Float[Array, ""]
) -> Float[Array, "nalpha ntheta nzeta"]:
    """Wind in theta direction for MDKE."""
    w = (
        field.B_sup_t / field.Bmag * pitchgrid.xi[:, None, None]
        + field.B_sub_z / field.B2mag_fsa / field.sqrtg * E_psi
    )
    return w


def dkes_w_zeta(
    field: Field, pitchgrid: UniformPitchAngleGrid, E_psi: Float[Array, ""]
) -> Float[Array, "nalpha ntheta nzeta"]:
    """Wind in zeta direction for MDKE."""
    w = (
        field.B_sup_z / field.Bmag * pitchgrid.xi[:, None, None]
        - field.B_sub_t / field.B2mag_fsa / field.sqrtg * E_psi
    )
    return w


def dkes_w_pitch(
    field: Field, pitchgrid: UniformPitchAngleGrid
) -> Float[Array, "nalpha ntheta nzeta"]:
    """Wind in xi/pitch direction for MDKE."""
    sina = jnp.sqrt(1 - pitchgrid.xi**2)
    w = (
        -field.bdotgradB
        / (2 * field.Bmag)
        * (1 - pitchgrid.xi[:, None, None] ** 2)
        / sina[:, None, None]
    )
    return w


class MDKETheta(lx.AbstractLinearOperator):
    """Advection operator in theta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        E_psi: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = True,
    ):
        assert field.ntheta > fd_coeffs[1][p1].size // 2
        assert field.ntheta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        w = dkes_w_theta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.ntheta

        fd = fdfwd(f, self.p1, h=h, bc="periodic", axis=1)
        bd = fdbwd(f, self.p1, h=h, bc="periodic", axis=1)
        # get only L or U by only taking forward or backward diff? + diagonal correction
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = dkes_w_theta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.ntheta
        fd = jnp.diag(jax.jacfwd(fdfwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = dkes_w_theta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.ntheta
        fd = (jax.jacfwd(fdfwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None, :
        ]
        bd = (jax.jacfwd(fdbwd)(f[0, :, 0], self.p1, h=h, bc="periodic"))[
            None, :, None, :
        ]
        w = w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, 0].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.field.ntheta, self.field.ntheta))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class MDKEZeta(lx.AbstractLinearOperator):
    """Advection operator in zeta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        E_psi: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = True,
    ):
        assert field.nzeta > fd_coeffs[1][p1].size // 2
        assert field.nzeta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        w = dkes_w_zeta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.nzeta / self.field.NFP

        fd = fdfwd(f, self.p1, h=h, bc="periodic", axis=2)
        bd = fdbwd(f, self.p1, h=h, bc="periodic", axis=2)
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = dkes_w_zeta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.nzeta / self.field.NFP
        fd = jnp.diag(jax.jacfwd(fdfwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = dkes_w_zeta(self.field, self.pitchgrid, self.E_psi)
        h = 2 * np.pi / self.field.nzeta / self.field.NFP
        fd = (jax.jacfwd(fdfwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :, :
        ]
        bd = (jax.jacfwd(fdbwd)(f[0, 0, :], self.p1, h=h, bc="periodic"))[
            None, None, :, :
        ]
        w = w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, 0].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.field.nzeta, self.field.nzeta))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class MDKEPitch(lx.AbstractLinearOperator):
    """Advection operator in pitch angle direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        E_psi: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = True,
    ):
        assert pitchgrid.nxi > fd_coeffs[1][p1].size // 2
        assert pitchgrid.nxi > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        w = dkes_w_pitch(self.field, self.pitchgrid)
        h = np.pi / self.pitchgrid.nxi

        fd = fdfwd(f, self.p1, h=h, bc="symmetric", axis=0)
        bd = fdbwd(f, self.p1, h=h, bc="symmetric", axis=0)
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = dkes_w_pitch(self.field, self.pitchgrid)
        h = np.pi / self.pitchgrid.nxi
        fd = jnp.diag(jax.jacfwd(fdfwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        w = dkes_w_pitch(self.field, self.pitchgrid)
        h = np.pi / self.pitchgrid.nxi
        fd = (jax.jacfwd(fdfwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None, :
        ]
        bd = (jax.jacfwd(fdbwd)(f[:, 0, 0], self.p1, h=h, bc="symmetric"))[
            :, None, None, :
        ]
        w = w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.nxi // 2
        scale = jnp.mean(jnp.abs(w)) / h
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, idx].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.pitchgrid.nxi, self.pitchgrid.nxi))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class MDKE(lx.AbstractLinearOperator):
    """Monoenergetic Drift Kinetic Equation operator.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    nu : float
        Normalized collisionality, nu/v
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    E_psi: Float[Array, ""]
    nu: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)
    _opa: MDKEPitch
    _opt: MDKETheta
    _opz: MDKEZeta
    _opp: MDKEPitchAngleScattering

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        E_psi: Float[ArrayLike, ""],
        nu: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = True,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.E_psi = jnp.array(E_psi)
        self.nu = jnp.array(nu)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        self._opa = MDKEPitch(field, pitchgrid, E_psi, p1, p2, axorder, gauge)
        self._opt = MDKETheta(field, pitchgrid, E_psi, p1, p2, axorder, gauge)
        self._opz = MDKEZeta(field, pitchgrid, E_psi, p1, p2, axorder, gauge)
        self._opp = MDKEPitchAngleScattering(
            field, pitchgrid, nu, p1, p2, axorder, gauge
        )

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f0 = self._opa.mv(vector)
        f1 = self._opt.mv(vector)
        f2 = self._opz.mv(vector)
        f3 = self._opp.mv(vector)
        return f0 + f1 + f2 + f3

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        d0 = self._opa.diagonal()
        d1 = self._opt.diagonal()
        d2 = self._opz.diagonal()
        d3 = self._opp.diagonal()
        return d0 + d1 + d2 + d3

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        d0 = self._opa.block_diagonal()
        d1 = self._opt.block_diagonal()
        d2 = self._opz.block_diagonal()
        d3 = self._opp.block_diagonal()
        return d0 + d1 + d2 + d3

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(MDKE)
@lx.is_diagonal.register(MDKE)
@lx.is_tridiagonal.register(MDKE)
@lx.is_symmetric.register(MDKETheta)
@lx.is_diagonal.register(MDKETheta)
@lx.is_tridiagonal.register(MDKETheta)
@lx.is_symmetric.register(MDKEZeta)
@lx.is_diagonal.register(MDKEZeta)
@lx.is_tridiagonal.register(MDKEZeta)
@lx.is_symmetric.register(MDKEPitch)
@lx.is_diagonal.register(MDKEPitch)
@lx.is_tridiagonal.register(MDKEPitch)
def _(operator):
    return False


#######
# SFINCS trajectories
#######


def sfincs_w_theta(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    E_psi: Float[Array, ""],
    v: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in theta direction for SFINCS DKE."""
    v = v[:, :, None, None, None]
    xi = pitchgrid.xi[None, None, :, None, None]
    vpar = v * xi
    w = (
        field.B_sup_t / field.Bmag * vpar
        + field.B_sub_z / field.Bmag**2 / field.sqrtg * E_psi
    )
    return w


def sfincs_w_zeta(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    E_psi: Float[Array, ""],
    v: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in zeta direction for SFINCS DKE."""
    v = v[:, :, None, None, None]
    xi = pitchgrid.xi[None, None, :, None, None]
    vpar = v * xi
    w = (
        field.B_sup_z / field.Bmag * vpar
        - field.B_sub_t / field.Bmag**2 / field.sqrtg * E_psi
    )
    return w


def sfincs_w_pitch(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    E_psi: Float[Array, ""],
    v: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in xi/pitch direction for SFINCS DKE."""
    xi = pitchgrid.xi[None, None, :, None, None]
    v = v[:, :, None, None, None]
    sina = jnp.sqrt(1 - xi**2)
    w1 = -field.bdotgradB / (2 * field.Bmag) * (1 - xi**2) / sina * v
    w2 = (
        xi * (1 - xi**2) / (2 * field.Bmag**3) * E_psi * field.BxgradpsidotgradB
    ) / sina
    return w1 + w2


def sfincs_w_speed(
    field: Field,
    pitchgrid: UniformPitchAngleGrid,
    E_psi: Float[Array, ""],
    x: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in speed/x direction for SFINCS DKE."""
    xi = pitchgrid.xi[None, None, :, None, None]
    x = x[:, :, None, None, None]
    w = (1 + xi**2) * x / (2 * field.Bmag**2) * field.BxgradpsidotgradB * E_psi
    return w


class DKETheta(lx.AbstractLinearOperator):
    """Advection operator in theta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Electric field, E_psi = -dPhi/dpsi in Volts/Webers
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        assert field.ntheta > fd_coeffs[1][p1].size // 2
        assert field.ntheta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.E_psi = jnp.array(E_psi)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_theta(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = 2 * np.pi / self.field.ntheta

        fd = fdfwd(f, self.p1, h=h, bc="periodic", axis=3)
        bd = fdbwd(f, self.p1, h=h, bc="periodic", axis=3)
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = jnp.ones(self.field.ntheta)
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_theta(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = 2 * np.pi / self.field.ntheta
        fd = jnp.diag(jax.jacfwd(fdfwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, :, None
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, :, None
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "s":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, len(self.species))))
        if self.axorder[-1] == "x":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.speedgrid.nx)))
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = jnp.ones(self.field.ntheta)
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_theta(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = 2 * np.pi / self.field.ntheta
        fd = (jax.jacfwd(fdfwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, :, None, :
        ]
        bd = (jax.jacfwd(fdbwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, :, None, :
        ]
        w = w[:, :, :, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, self.field.ntheta, self.field.ntheta))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class DKEZeta(lx.AbstractLinearOperator):
    """Advection operator in zeta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Electric field, E_psi = -dPhi/dpsi in Volts/Webers
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        assert field.nzeta > fd_coeffs[1][p1].size // 2
        assert field.nzeta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.E_psi = jnp.array(E_psi)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_zeta(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = 2 * np.pi / self.field.nzeta / self.field.NFP

        fd = fdfwd(f, self.p1, h=h, bc="periodic", axis=4)
        bd = fdbwd(f, self.p1, h=h, bc="periodic", axis=4)
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = jnp.ones(self.field.nzeta)
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_zeta(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = 2 * np.pi / self.field.nzeta / self.field.NFP
        fd = jnp.diag(jax.jacfwd(fdfwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, None, :
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, None, :
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "s":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, len(self.species))))
        if self.axorder[-1] == "x":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.speedgrid.nx)))
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = jnp.ones(self.field.nzeta)
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_zeta(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = 2 * np.pi / self.field.nzeta / self.field.NFP
        fd = (jax.jacfwd(fdfwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, None, :, :
        ]
        bd = (jax.jacfwd(fdbwd)(f, self.p1, h=h, bc="periodic"))[
            None, None, None, None, :, :
        ]
        w = w[:, :, :, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, self.field.nzeta, self.field.nzeta))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class DKEPitch(lx.AbstractLinearOperator):
    """Advection operator in pitch angle direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Electric field, E_psi = -dPhi/dpsi in Volts/Webers
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        assert pitchgrid.nxi > fd_coeffs[1][p1].size // 2
        assert pitchgrid.nxi > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.E_psi = jnp.array(E_psi)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_pitch(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = np.pi / self.pitchgrid.nxi

        fd = fdfwd(f, self.p1, h=h, bc="symmetric", axis=2)
        bd = fdbwd(f, self.p1, h=h, bc="symmetric", axis=2)
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = jnp.ones(self.pitchgrid.nxi)
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_pitch(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = np.pi / self.pitchgrid.nxi
        fd = jnp.diag(jax.jacfwd(fdfwd)(f, self.p1, h=h, bc="symmetric"))[
            None, None, :, None, None
        ]
        bd = jnp.diag(jax.jacfwd(fdbwd)(f, self.p1, h=h, bc="symmetric"))[
            None, None, :, None, None
        ]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "s":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, len(self.species))))
        if self.axorder[-1] == "x":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.speedgrid.nx)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = jnp.ones(self.pitchgrid.nxi)
        vth = jnp.array([s.v_thermal for s in self.species])
        w = sfincs_w_pitch(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * vth[:, None],
        )
        h = np.pi / self.pitchgrid.nxi
        fd = (jax.jacfwd(fdfwd)(f, self.p1, h=h, bc="symmetric"))[
            None, None, :, None, None, :
        ]
        bd = (jax.jacfwd(fdbwd)(f, self.p1, h=h, bc="symmetric"))[
            None, None, :, None, None, :
        ]
        w = w[:, :, :, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, self.pitchgrid.nxi, self.pitchgrid.nxi))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class DKESpeed(lx.AbstractLinearOperator):
    """Advection operator in speed direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Electric field, E_psi = -dPhi/dpsi in Volts/Webers
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    E_psi: Float[Array, ""]
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.E_psi = jnp.array(E_psi)
        self.axorder = axorder

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        w = sfincs_w_speed(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * jnp.ones(len(self.species))[:, None],
        )
        df = jnp.einsum("yx,sxatz->syatz", self.speedgrid.Dx_pseudospectral, f)
        df = w * df
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        w = sfincs_w_speed(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * jnp.ones(len(self.species))[:, None],
        )
        df = jnp.diag(self.speedgrid.Dx_pseudospectral)[None, :, None, None, None]
        df = w * df
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "s":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, len(self.species))))
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        w = sfincs_w_speed(
            self.field,
            self.pitchgrid,
            self.E_psi,
            self.speedgrid.x[None, :] * jnp.ones(len(self.species))[:, None],
        )
        df = self.speedgrid.Dx_pseudospectral[None, :, None, None, None, :]
        w = w[:, :, :, :, :, None]
        df = w * df
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, self.speedgrid.nx, self.speedgrid.nx))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class DKE(lx.AbstractLinearOperator):
    """Drift Kinetic Equation operator.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Normalized electric field, E_psi/v
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    E_psi: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    _opx: DKESpeed
    _opa: DKEPitch
    _opt: DKETheta
    _opz: DKEZeta
    _C: FokkerPlanckLandau

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        potentials: Optional[RosenbluthPotentials] = None,
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if potentials is None:
            potentials = RosenbluthPotentials(speedgrid, pitchgrid, species)
        self.potentials = potentials
        self.E_psi = jnp.array(E_psi)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder

        self._opx = DKESpeed(field, pitchgrid, speedgrid, species, E_psi, axorder)
        self._opa = DKEPitch(
            field, pitchgrid, speedgrid, species, E_psi, p1, p2, axorder
        )
        self._opt = DKETheta(
            field, pitchgrid, speedgrid, species, E_psi, p1, p2, axorder
        )
        self._opz = DKEZeta(
            field, pitchgrid, speedgrid, species, E_psi, p1, p2, axorder
        )
        self._C = FokkerPlanckLandau(
            field, pitchgrid, speedgrid, species, potentials, p2, axorder
        )

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f0 = self._opx.mv(vector)
        f1 = self._opa.mv(vector)
        f2 = self._opt.mv(vector)
        f3 = self._opz.mv(vector)
        f4 = self._C.mv(vector)
        return f0 + f1 + f2 + f3 + f4

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        d0 = self._opx.diagonal()
        d1 = self._opa.diagonal()
        d2 = self._opt.diagonal()
        d3 = self._opz.diagonal()
        d4 = self._C.diagonal()
        return d0 + d1 + d2 + d3 + d4

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        d0 = self._opx.block_diagonal()
        d1 = self._opa.block_diagonal()
        d2 = self._opt.block_diagonal()
        d3 = self._opz.block_diagonal()
        d4 = self._C.block_diagonal()
        return d0 + d1 + d2 + d3 + d4

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(DKE)
@lx.is_diagonal.register(DKE)
@lx.is_tridiagonal.register(DKE)
@lx.is_symmetric.register(DKESpeed)
@lx.is_diagonal.register(DKESpeed)
@lx.is_tridiagonal.register(DKESpeed)
@lx.is_symmetric.register(DKEPitch)
@lx.is_diagonal.register(DKEPitch)
@lx.is_tridiagonal.register(DKEPitch)
@lx.is_symmetric.register(DKEZeta)
@lx.is_diagonal.register(DKEZeta)
@lx.is_tridiagonal.register(DKEZeta)
@lx.is_symmetric.register(DKETheta)
@lx.is_diagonal.register(DKETheta)
@lx.is_tridiagonal.register(DKETheta)
def _(operator):
    return False
