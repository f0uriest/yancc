"""Drift Kinetic Operators without collisions."""

import functools
import itertools
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
from .finite_diff import build_advection_matrix, fd_coeffs, fd_kwargs
from .linalg import (
    TransposedLinearOperator,
    banded_mm,
    banded_to_dense,
    dense_to_banded,
)
from .species import LocalMaxwellian
from .utils import _parse_axorder_shape_3d, _parse_axorder_shape_4d, _refold
from .velocity_grids import (
    AbstractPitchAngleGrid,
    AbstractSpeedGrid,
    MaxwellSpeedGrid,
    MonoenergeticSpeedGrid,
)


def _advection_matrices(x, p, bc_type, domain):
    """Forward/backward advection matrices for stencil id ``p``.

    Builds the matrices from the actual node coordinates ``x`` (so non-uniform
    spacing is handled correctly), reproducing the uniform stencil ``p`` via the
    ``fd_kwargs`` lookup.
    """
    kwargs = fd_kwargs[p]
    fd = build_advection_matrix(
        x, direction="fwd", bc_type=bc_type, domain=domain, **kwargs
    )
    bd = build_advection_matrix(
        x, direction="bwd", bc_type=bc_type, domain=domain, **kwargs
    )
    return fd, bd


def dkes_w_theta(
    field: Field, pitchgrid: AbstractPitchAngleGrid, erhohat: Float[Array, ""]
) -> Float[Array, "nalpha ntheta nzeta"]:
    """Wind in theta direction for MDKE."""
    w = (
        field.B_sup_t / field.Bmag * pitchgrid.xi[:, None, None]
        + field.B_sub_z / field.B2mag_fsa / field.sqrtg * erhohat
    )
    return w


def dkes_w_zeta(
    field: Field, pitchgrid: AbstractPitchAngleGrid, erhohat: Float[Array, ""]
) -> Float[Array, "nalpha ntheta nzeta"]:
    """Wind in zeta direction for MDKE."""
    w = (
        field.B_sup_z / field.Bmag * pitchgrid.xi[:, None, None]
        - field.B_sub_t / field.B2mag_fsa / field.sqrtg * erhohat
    )
    return w


def dkes_w_pitch(
    field: Field, pitchgrid: AbstractPitchAngleGrid
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
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v in units of V*s/m
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
    pitchgrid: AbstractPitchAngleGrid
    erhohat: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)
    _fd: Float[Array, "nt nt"]
    _bd: Float[Array, "nt nt"]
    _w: Float[Array, "na nt nz"]
    _scale: Float[Array, ""]

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert field.ntheta > fd_coeffs[1][p1].size // 2
        assert field.ntheta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.erhohat = jnp.array(erhohat)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        h = 2 * np.pi / field.ntheta
        self._fd, self._bd = _advection_matrices(
            field.theta, p1, bc_type="periodic", domain=(0, 2 * np.pi)
        )
        self._w = dkes_w_theta(field, pitchgrid, self.erhohat)
        self._scale = jnp.mean(jnp.abs(self._w)) / h

    @eqx.filter_jit
    @jax.named_scope("MDKETheta.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))  # (na, nt, nz)
        f1 = jnp.moveaxis(f, 1, -1)  # (na, nz, nt) - convolved axis last
        fd_f = jnp.moveaxis(f1 @ self._fd.T, -1, 1)
        bd_f = jnp.moveaxis(f1 @ self._bd.T, -1, 1)
        w = self._w
        df = w * ((w > 0) * bd_f + (w <= 0) * fd_f)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0].set(
                self._scale * f[idx, 0, 0],
                indices_are_sorted=True,
                unique_indices=True,
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("MDKETheta.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        fd = jnp.diag(self._fd)[None, :, None]
        bd = jnp.diag(self._bd)[None, :, None]
        w = self._w
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0].set(
                self._scale, indices_are_sorted=True, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("MDKETheta.block_diagonal")
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.na)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        fd = self._fd[None, :, None, :]
        bd = self._bd[None, :, None, :]
        w = self._w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0, :]
            .set(0, indices_are_sorted=True, unique_indices=True)
            .at[idx, 0, 0, 0]
            .set(self._scale, indices_are_sorted=True, unique_indices=True),
            df,
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
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class MDKEZeta(lx.AbstractLinearOperator):
    """Advection operator in zeta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v in units of V*s/m
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
    pitchgrid: AbstractPitchAngleGrid
    erhohat: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)
    _fd: Float[Array, "nz nz"]
    _bd: Float[Array, "nz nz"]
    _w: Float[Array, "na nt nz"]
    _scale: Float[Array, ""]

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert field.nzeta > fd_coeffs[1][p1].size // 2
        assert field.nzeta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.erhohat = jnp.array(erhohat)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        h = 2 * np.pi / field.nzeta / field.NFP
        self._fd, self._bd = _advection_matrices(
            field.zeta, p1, bc_type="periodic", domain=(0, 2 * np.pi / field.NFP)
        )
        self._w = dkes_w_zeta(field, pitchgrid, self.erhohat)
        self._scale = jnp.mean(jnp.abs(self._w)) / h

    @eqx.filter_jit
    @jax.named_scope("MDKEZeta.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))  # (na, nt, nz)
        # convolved (zeta) axis is already last
        fd_f = f @ self._fd.T
        bd_f = f @ self._bd.T
        w = self._w
        df = w * ((w > 0) * bd_f + (w <= 0) * fd_f)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0].set(
                self._scale * f[idx, 0, 0],
                indices_are_sorted=True,
                unique_indices=True,
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("MDKEZeta.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        fd = jnp.diag(self._fd)[None, None, :]
        bd = jnp.diag(self._bd)[None, None, :]
        w = self._w
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0].set(
                self._scale, indices_are_sorted=True, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("MDKEZeta.block_diagonal")
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.na)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        fd = self._fd[None, None, :, :]
        bd = self._bd[None, None, :, :]
        w = self._w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0, :]
            .set(0, indices_are_sorted=True, unique_indices=True)
            .at[idx, 0, 0, 0]
            .set(self._scale, indices_are_sorted=True, unique_indices=True),
            df,
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
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class MDKEPitch(lx.AbstractLinearOperator):
    """Advection operator in pitch angle direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v in units of V*s/m
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
    pitchgrid: AbstractPitchAngleGrid
    erhohat: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)
    _fd: Float[Array, "na na"]
    _bd: Float[Array, "na na"]
    _w: Float[Array, "na nt nz"]
    _scale: Float[Array, ""]

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert pitchgrid.na > fd_coeffs[1][p1].size // 2
        assert pitchgrid.na > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.erhohat = jnp.array(erhohat)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        h = np.pi / pitchgrid.na
        self._fd, self._bd = _advection_matrices(
            pitchgrid.a, p1, bc_type="symmetric", domain=(0, np.pi)
        )
        self._w = dkes_w_pitch(field, pitchgrid)
        self._scale = jnp.mean(jnp.abs(self._w)) / h

    @eqx.filter_jit
    @jax.named_scope("MDKEPitch.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))  # (na, nt, nz)
        f1 = jnp.moveaxis(f, 0, -1)  # (nt, nz, na) - convolved axis last
        fd_f = jnp.moveaxis(f1 @ self._fd.T, -1, 0)
        bd_f = jnp.moveaxis(f1 @ self._bd.T, -1, 0)
        w = self._w
        df = w * ((w > 0) * bd_f + (w <= 0) * fd_f)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0].set(
                self._scale * f[idx, 0, 0],
                indices_are_sorted=True,
                unique_indices=True,
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("MDKEPitch.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        fd = jnp.diag(self._fd)[:, None, None]
        bd = jnp.diag(self._bd)[:, None, None]
        w = self._w
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0].set(
                self._scale, indices_are_sorted=True, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("MDKEPitch.block_diagonal")
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.na, self.axorder
        )
        fd = self._fd[:, None, None, :]
        bd = self._bd[:, None, None, :]
        w = self._w[:, :, :, None]
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idx = self.pitchgrid.na // 2
        df = jnp.where(
            self.gauge,
            df.at[idx, 0, 0, :]
            .set(0, indices_are_sorted=True, unique_indices=True)
            .at[idx, 0, 0, idx]
            .set(self._scale, indices_are_sorted=True, unique_indices=True),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.pitchgrid.na, self.pitchgrid.na))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class MDKE(lx.AbstractLinearOperator):
    """Monoenergetic Drift Kinetic Equation operator.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    erhohat : float
        Monoenergetic electric field, Erho/v in units of V*s/m
    nuhat : float
        Monoenergetic collisionality, nu/v in units of 1/m
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
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    erhohat: Float[Array, ""]
    nuhat: Float[Array, ""]
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
        pitchgrid: AbstractPitchAngleGrid,
        erhohat: Float[ArrayLike, ""],
        nuhat: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = MonoenergeticSpeedGrid(jnp.array(1.0))
        self.erhohat = jnp.array(erhohat)
        self.nuhat = jnp.array(nuhat)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        self._opa = MDKEPitch(field, pitchgrid, erhohat, p1, p2, axorder, gauge)
        self._opt = MDKETheta(field, pitchgrid, erhohat, p1, p2, axorder, gauge)
        self._opz = MDKEZeta(field, pitchgrid, erhohat, p1, p2, axorder, gauge)
        self._opp = MDKEPitchAngleScattering(
            field, pitchgrid, nuhat, p1, p2, axorder, gauge
        )

    @eqx.filter_jit
    @jax.named_scope("MDKE.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f0 = self._opa.mv(vector)
        f1 = self._opt.mv(vector)
        f2 = self._opz.mv(vector)
        f3 = self._opp.mv(vector)
        return f0 + f1 + f2 + f3

    @eqx.filter_jit
    @jax.named_scope("MDKE.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        d0 = self._opa.diagonal()
        d1 = self._opt.diagonal()
        d2 = self._opz.diagonal()
        d3 = self._opp.diagonal()
        return d0 + d1 + d2 + d3

    @eqx.filter_jit
    @jax.named_scope("MDKE.block_diagonal")
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        d0 = self._opa.block_diagonal()
        d1 = self._opt.block_diagonal()
        d2 = self._opz.block_diagonal()
        d3 = self._opp.block_diagonal()
        return d0 + d1 + d2 + d3

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.na,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


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
    pitchgrid: AbstractPitchAngleGrid,
    Erho: Float[Array, ""],
    v: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in theta direction for SFINCS DKE."""
    v = v[:, :, None, None, None]
    xi = pitchgrid.xi[None, None, :, None, None]
    vpar = v * xi
    w = (
        field.B_sup_t / field.Bmag * vpar
        + field.B_sub_z / field.Bmag**2 / field.sqrtg * (-Erho)
    )
    return w


def sfincs_w_zeta(
    field: Field,
    pitchgrid: AbstractPitchAngleGrid,
    Erho: Float[Array, ""],
    v: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in zeta direction for SFINCS DKE."""
    v = v[:, :, None, None, None]
    xi = pitchgrid.xi[None, None, :, None, None]
    vpar = v * xi
    w = (
        field.B_sup_z / field.Bmag * vpar
        - field.B_sub_t / field.Bmag**2 / field.sqrtg * (-Erho)
    )
    return w


def sfincs_w_pitch(
    field: Field,
    pitchgrid: AbstractPitchAngleGrid,
    Erho: Float[Array, ""],
    v: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in xi/pitch direction for SFINCS DKE."""
    xi = pitchgrid.xi[None, None, :, None, None]
    v = v[:, :, None, None, None]
    sina = jnp.sqrt(1 - xi**2)
    w1 = -field.bdotgradB / (2 * field.Bmag) * (1 - xi**2) / sina * v
    w2 = (
        xi * (1 - xi**2) / (2 * field.Bmag**3) * (-Erho) * field.BxgradrhodotgradB
    ) / sina
    return w1 + w2


def sfincs_w_speed(
    field: Field,
    pitchgrid: AbstractPitchAngleGrid,
    Erho: Float[Array, ""],
    x: Float[Array, "ns nx"],
) -> Float[Array, "ns nx na nt nz"]:
    """Wind in speed/x direction for SFINCS DKE."""
    xi = pitchgrid.xi[None, None, :, None, None]
    x = x[:, :, None, None, None]
    w = (1 + xi**2) * x / (2 * field.Bmag**3) * field.BxgradrhodotgradB * (-Erho)
    return w


class DKETheta(lx.AbstractLinearOperator):
    """Advection operator in theta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
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
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]

    _fd: Float[Array, "nt nt"]
    _bd: Float[Array, "nt nt"]
    _w: Float[Array, "ns nx na nt nz"]
    _scale: Float[Array, "ns nidx"]

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        assert field.ntheta > fd_coeffs[1][p1].size // 2
        assert field.ntheta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.Erho = jnp.array(Erho)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        h = 2 * np.pi / field.ntheta
        self._fd, self._bd = _advection_matrices(
            field.theta, p1, bc_type="periodic", domain=(0, 2 * np.pi)
        )
        vth = jnp.array([s.v_thermal for s in species])
        w = sfincs_w_theta(
            field, pitchgrid, self.Erho, speedgrid.x[None, :] * vth[:, None]
        )
        self._w = w
        idxx = speedgrid.gauge_idx
        self._scale = jnp.mean(jnp.abs(w), axis=(2, 3, 4))[:, idxx] / h

    @eqx.filter_jit
    @jax.named_scope("DKETheta.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))  # (ns, nx, na, nt, nz)
        f1 = jnp.moveaxis(f, 3, -1)  # (ns, nx, na, nz, nt) - convolved axis last
        fd_f = jnp.moveaxis(f1 @ self._fd.T, -1, 3)
        bd_f = jnp.moveaxis(f1 @ self._bd.T, -1, 3)
        w = self._w
        df = w * ((w > 0) * bd_f + (w <= 0) * fd_f)
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                self._scale * f[:, idxx, idxa, 0, 0],
                indices_are_sorted=True,
                unique_indices=True,
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("DKETheta.diagonal")
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = jnp.diag(self._fd)[None, None, None, :, None]
        bd = jnp.diag(self._bd)[None, None, None, :, None]
        w = self._w
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                self._scale, indices_are_sorted=True, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("DKETheta.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        assert fmt in ["dense", "banded"]

        if self.axorder[-1] != "t":  # its just diagonal
            if bw is None:
                bw = 0
            df = self.diagonal()
            sizes = {
                "s": len(self.species),
                "x": self.speedgrid.nx,
                "a": self.pitchgrid.na,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            df = df.reshape((-1, sizes[self.axorder[-1]]))
            if fmt == "dense":
                op = jax.vmap(jnp.diag)
            else:
                op = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
            return op(df)

        if bw is None:
            bw = max(fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2)

        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = dense_to_banded(bw, bw, self._fd)
        bd = dense_to_banded(bw, bw, self._bd)
        w = self._w
        w1 = jnp.moveaxis(w, 3, -1)[..., None, :]
        wf = w1 * (w1 <= 0)
        wb = w1 * (w1 > 0)
        dff, _, _ = banded_mm(0, 0, bw, bw, wf, fd)
        dfb, _, _ = banded_mm(0, 0, bw, bw, wb, bd)
        df = dff + dfb

        idxa = jnp.atleast_1d(self.pitchgrid.na // 2)
        idxx = self.speedgrid.gauge_idx

        bandwidth = 2 * bw + 1
        bands = jnp.arange(bandwidth)
        cols = (bw - bands) % self.field.ntheta
        basis = jnp.zeros(bandwidth, dtype=df.dtype).at[bw].set(1.0)
        vals = self._scale[:, :, None] * basis[None, None, :]
        idxx_mesh = idxx[:, None]
        idxa_mesh = idxa[:, None]
        bands_mesh = bands[None, :]
        cols_mesh = cols[None, :]
        df = jnp.where(
            self.gauge,
            df.at[:, idxx_mesh, idxa_mesh, 0, bands_mesh, cols_mesh].set(
                vals, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, 4, 3)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, 2 * bw + 1, self.field.ntheta))
        if fmt == "dense":
            df = banded_to_dense(bw, bw, df)
        return df

    @eqx.filter_jit
    @jax.named_scope("DKETheta.block_diagonal2")
    def block_diagonal2(self):
        """Block diagonal of operator as (N,M,M) array. Unfolds s,x"""
        assert self.axorder[-2:] == "sx"
        if self.axorder[2] == "a":
            return _refold(self.block_diagonal(), len(self.species) * self.pitchgrid.na)
        if self.axorder[2] == "z":
            return _refold(self.block_diagonal(), len(self.species) * self.field.nzeta)

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = self._fd
        bd = self._bd
        Is = jnp.eye(len(self.species))
        Ix = jnp.eye(self.speedgrid.nx)

        ff = functools.reduce(jnp.kron, [fd, Is, Ix])
        bb = functools.reduce(jnp.kron, [bd, Is, Ix])

        w1 = jnp.moveaxis(self._w, (0, 1, 2, 3, 4), caxorder)
        w1 = w1.reshape(w1.shape[0] * w1.shape[1], -1, 1)
        df = w1 * ((w1 > 0) * bb + (w1 <= 0) * ff)
        df = df.reshape(*shape, self.field.ntheta, len(self.species), self.speedgrid.nx)
        df = jnp.moveaxis(df, caxorder, (0, 1, 2, 3, 4))
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        idxs = jnp.arange(len(self.species))
        idxsx = idxs[:, None] * self.speedgrid.nx + idxx
        idxs, idxx = jnp.unravel_index(idxsx, (len(self.species), self.speedgrid.nx))

        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0, :, :, :]
            .set(0, indices_are_sorted=True, unique_indices=True)
            .at[idxs, idxx, idxa, 0, 0, 0, idxs, idxx]
            .set(self._scale, indices_are_sorted=True, unique_indices=True),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        N = self.in_size()
        M = self.field.ntheta * len(self.species) * self.speedgrid.nx
        return df.reshape(N // M, M, M)

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
                * self.pitchgrid.na
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
                * self.pitchgrid.na
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class DKEZeta(lx.AbstractLinearOperator):
    """Advection operator in zeta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
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
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]

    _fd: Float[Array, "nz nz"]
    _bd: Float[Array, "nz nz"]
    _w: Float[Array, "ns nx na nt nz"]
    _scale: Float[Array, "ns nidx"]

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        assert field.nzeta > fd_coeffs[1][p1].size // 2
        assert field.nzeta > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.Erho = jnp.array(Erho)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        h = 2 * np.pi / field.nzeta / field.NFP
        self._fd, self._bd = _advection_matrices(
            field.zeta, p1, bc_type="periodic", domain=(0, 2 * np.pi / field.NFP)
        )
        vth = jnp.array([s.v_thermal for s in species])
        w = sfincs_w_zeta(
            field, pitchgrid, self.Erho, speedgrid.x[None, :] * vth[:, None]
        )
        self._w = w
        idxx = speedgrid.gauge_idx
        self._scale = jnp.mean(jnp.abs(w), axis=(2, 3, 4))[:, idxx] / h

    @eqx.filter_jit
    @jax.named_scope("DKEZeta.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))  # (ns, nx, na, nt, nz)
        # convolved (zeta) axis already last
        fd_f = f @ self._fd.T
        bd_f = f @ self._bd.T
        w = self._w
        df = w * ((w > 0) * bd_f + (w <= 0) * fd_f)
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                self._scale * f[:, idxx, idxa, 0, 0],
                indices_are_sorted=True,
                unique_indices=True,
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("DKEZeta.diagonal")
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = jnp.diag(self._fd)[None, None, None, None, :]
        bd = jnp.diag(self._bd)[None, None, None, None, :]
        w = self._w
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                self._scale, indices_are_sorted=True, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("DKEZeta.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        assert fmt in ["dense", "banded"]

        if self.axorder[-1] != "z":  # its just diagonal
            if bw is None:
                bw = 0
            df = self.diagonal()
            sizes = {
                "s": len(self.species),
                "x": self.speedgrid.nx,
                "a": self.pitchgrid.na,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            df = df.reshape((-1, sizes[self.axorder[-1]]))
            if fmt == "dense":
                op = jax.vmap(jnp.diag)
            else:
                op = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
            return op(df)

        if bw is None:
            bw = max(fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2)

        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = dense_to_banded(bw, bw, self._fd)
        bd = dense_to_banded(bw, bw, self._bd)
        w = self._w
        w1 = jnp.moveaxis(w, 4, -1)[..., None, :]
        wf = w1 * (w1 <= 0)
        wb = w1 * (w1 > 0)
        dff, _, _ = banded_mm(0, 0, bw, bw, wf, fd)
        dfb, _, _ = banded_mm(0, 0, bw, bw, wb, bd)
        df = dff + dfb

        idxa = jnp.atleast_1d(self.pitchgrid.na // 2)
        idxx = self.speedgrid.gauge_idx
        bandwidth = 2 * bw + 1
        bands = jnp.arange(bandwidth)
        cols = (bw - bands) % self.field.nzeta
        basis = jnp.zeros(bandwidth, dtype=df.dtype).at[bw].set(1.0)
        vals = self._scale[:, :, None] * basis[None, None, :]
        idxx_mesh = idxx[:, None]
        idxa_mesh = idxa[:, None]
        bands_mesh = bands[None, :]
        cols_mesh = cols[None, :]
        df = jnp.where(
            self.gauge,
            df.at[:, idxx_mesh, idxa_mesh, 0, bands_mesh, cols_mesh].set(
                vals, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, 2 * bw + 1, self.field.nzeta))
        if fmt == "dense":
            df = banded_to_dense(bw, bw, df)
        return df

    @eqx.filter_jit
    @jax.named_scope("DKEZeta.block_diagonal2")
    def block_diagonal2(self):
        """Block diagonal of operator as (N,M,M) array. Unfolds s,x"""
        assert self.axorder[-2:] == "sx"
        if self.axorder[2] == "a":
            return _refold(self.block_diagonal(), len(self.species) * self.pitchgrid.na)
        if self.axorder[2] == "t":
            return _refold(self.block_diagonal(), len(self.species) * self.field.ntheta)

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = self._fd
        bd = self._bd
        Is = jnp.eye(len(self.species))
        Ix = jnp.eye(self.speedgrid.nx)

        ff = functools.reduce(jnp.kron, [fd, Is, Ix])
        bb = functools.reduce(jnp.kron, [bd, Is, Ix])

        w1 = jnp.moveaxis(self._w, (0, 1, 2, 3, 4), caxorder)
        w1 = w1.reshape(w1.shape[0] * w1.shape[1], -1, 1)
        df = w1 * ((w1 > 0) * bb + (w1 <= 0) * ff)
        df = df.reshape(*shape, self.field.nzeta, len(self.species), self.speedgrid.nx)
        df = jnp.moveaxis(df, caxorder, (0, 1, 2, 3, 4))
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        idxs = jnp.arange(len(self.species))
        idxsx = idxs[:, None] * self.speedgrid.nx + idxx
        idxs, idxx = jnp.unravel_index(idxsx, (len(self.species), self.speedgrid.nx))

        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0, :, :, :]
            .set(0, indices_are_sorted=True, unique_indices=True)
            .at[idxs, idxx, idxa, 0, 0, 0, idxs, idxx]
            .set(self._scale, indices_are_sorted=True, unique_indices=True),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        N = self.in_size()
        M = self.field.nzeta * len(self.species) * self.speedgrid.nx
        return df.reshape(N // M, M, M)

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
                * self.pitchgrid.na
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
                * self.pitchgrid.na
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class DKEPitch(lx.AbstractLinearOperator):
    """Advection operator in pitch angle direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
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
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]

    _fd: Float[Array, "na na"]
    _bd: Float[Array, "na na"]
    _w: Float[Array, "ns nx na nt nz"]
    _scale: Float[Array, "ns nidx"]

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        assert pitchgrid.na > fd_coeffs[1][p1].size // 2
        assert pitchgrid.na > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.Erho = jnp.array(Erho)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        h = np.pi / pitchgrid.na
        self._fd, self._bd = _advection_matrices(
            pitchgrid.a, p1, bc_type="symmetric", domain=(0, np.pi)
        )
        vth = jnp.array([s.v_thermal for s in species])
        w = sfincs_w_pitch(
            field, pitchgrid, self.Erho, speedgrid.x[None, :] * vth[:, None]
        )
        self._w = w
        idxx = speedgrid.gauge_idx
        self._scale = jnp.mean(jnp.abs(w), axis=(2, 3, 4))[:, idxx] / h

    @eqx.filter_jit
    @jax.named_scope("DKEPitch.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))  # (ns, nx, na, nt, nz)
        f1 = jnp.moveaxis(f, 2, -1)  # (ns, nx, nt, nz, na) - convolved axis last
        fd_f = jnp.moveaxis(f1 @ self._fd.T, -1, 2)
        bd_f = jnp.moveaxis(f1 @ self._bd.T, -1, 2)
        w = self._w
        df = w * ((w > 0) * bd_f + (w <= 0) * fd_f)
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                self._scale * f[:, idxx, idxa, 0, 0],
                indices_are_sorted=True,
                unique_indices=True,
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("DKEPitch.diagonal")
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = jnp.diag(self._fd)[None, None, :, None, None]
        bd = jnp.diag(self._bd)[None, None, :, None, None]
        w = self._w
        df = w * ((w > 0) * bd + (w <= 0) * fd)
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                self._scale, indices_are_sorted=True, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("DKEPitch.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        assert fmt in ["dense", "banded"]

        if self.axorder[-1] != "a":  # its just diagonal
            if bw is None:
                bw = 0
            df = self.diagonal()
            sizes = {
                "s": len(self.species),
                "x": self.speedgrid.nx,
                "a": self.pitchgrid.na,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            df = df.reshape((-1, sizes[self.axorder[-1]]))
            if fmt == "dense":
                op = jax.vmap(jnp.diag)
            else:
                op = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
            return op(df)

        if bw is None:
            bw = max(fd_coeffs[1][self.p1].size // 2, fd_coeffs[2][self.p2].size // 2)

        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = dense_to_banded(bw, bw, self._fd)
        bd = dense_to_banded(bw, bw, self._bd)
        w = self._w
        w1 = jnp.moveaxis(w, 2, -1)[..., None, :]
        wf = w1 * (w1 <= 0)
        wb = w1 * (w1 > 0)
        dff, _, _ = banded_mm(0, 0, bw, bw, wf, fd)
        dfb, _, _ = banded_mm(0, 0, bw, bw, wb, bd)
        df = dff + dfb

        idxa = jnp.atleast_1d(self.pitchgrid.na // 2)
        idxx = self.speedgrid.gauge_idx

        bandwidth = 2 * bw + 1
        bands = jnp.arange(bandwidth)
        cols = (idxa[:, None] + bw - bands[None, :]) % self.pitchgrid.na
        basis = jnp.zeros(bandwidth, dtype=df.dtype).at[bw].set(1.0)
        vals = self._scale[:, :, None] * basis[None, None, :]
        idxx_mesh = idxx[:, None]
        bands_mesh = bands[None, :]
        df = jnp.where(
            self.gauge,
            df.at[:, idxx_mesh, 0, 0, bands_mesh, cols].set(vals, unique_indices=True),
            df,
        )
        df = jnp.moveaxis(df, 4, 2)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, 2 * bw + 1, self.pitchgrid.na))
        if fmt == "dense":
            df = banded_to_dense(bw, bw, df)
        return df

    @eqx.filter_jit
    @jax.named_scope("DKEPitch.block_diagonal2")
    def block_diagonal2(self):
        """Block diagonal of operator as (N,M,M) array. Unfolds s,x"""
        assert self.axorder[-2:] == "sx"
        if self.axorder[2] == "t":
            return _refold(self.block_diagonal(), len(self.species) * self.field.ntheta)
        if self.axorder[2] == "z":
            return _refold(self.block_diagonal(), len(self.species) * self.field.nzeta)

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        fd = self._fd
        bd = self._bd
        Is = jnp.eye(len(self.species))
        Ix = jnp.eye(self.speedgrid.nx)

        ff = functools.reduce(jnp.kron, [fd, Is, Ix])
        bb = functools.reduce(jnp.kron, [bd, Is, Ix])

        w1 = jnp.moveaxis(self._w, (0, 1, 2, 3, 4), caxorder)
        w1 = w1.reshape(w1.shape[0] * w1.shape[1], -1, 1)
        df = w1 * ((w1 > 0) * bb + (w1 <= 0) * ff)
        df = df.reshape(*shape, self.pitchgrid.na, len(self.species), self.speedgrid.nx)
        df = jnp.moveaxis(df, caxorder, (0, 1, 2, 3, 4))
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        idxs = jnp.arange(len(self.species))
        idxsx = idxs[:, None] * self.speedgrid.nx + idxx
        idxs, idxx = jnp.unravel_index(idxsx, (len(self.species), self.speedgrid.nx))

        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0, :, :, :]
            .set(0, indices_are_sorted=True, unique_indices=True)
            .at[idxs, idxx, idxa, 0, 0, idxa, idxs, idxx]
            .set(self._scale, indices_are_sorted=True, unique_indices=True),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        N = self.in_size()
        M = self.pitchgrid.na * len(self.species) * self.speedgrid.nx
        return df.reshape(N // M, M, M)

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
                * self.pitchgrid.na
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
                * self.pitchgrid.na
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class DKESpeed(lx.AbstractLinearOperator):
    """Advection operator in speed direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    Erho: Float[Array, ""]
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.Erho = jnp.array(Erho)
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    @jax.named_scope("DKESpeed.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        w = sfincs_w_speed(
            self.field,
            self.pitchgrid,
            self.Erho,
            self.speedgrid.x[None, :] * jnp.ones(len(self.species))[:, None],
        )
        df = jnp.einsum("yx,sxatz->syatz", self.speedgrid.Dx_pseudospectral, f)
        df = w * df
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        scale = jnp.mean(jnp.abs(w), axis=(2, 3, 4))[:, idxx] / jnp.mean(
            self.speedgrid.wx
        )
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                scale * f[:, idxx, idxa, 0, 0],
                indices_are_sorted=True,
                unique_indices=True,
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("DKESpeed.diagonal")
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        w = sfincs_w_speed(
            self.field,
            self.pitchgrid,
            self.Erho,
            self.speedgrid.x[None, :] * jnp.ones(len(self.species))[:, None],
        )
        df = jnp.diag(self.speedgrid.Dx_pseudospectral)[None, :, None, None, None]
        df = w * df
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        scale = jnp.mean(jnp.abs(w), axis=(2, 3, 4))[:, idxx] / jnp.mean(
            self.speedgrid.wx
        )
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0].set(
                scale, indices_are_sorted=True, unique_indices=True
            ),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("DKESpeed.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        assert fmt in ["dense", "banded"]

        if self.axorder[-1] != "x":  # its just diagonal
            if bw is None:
                bw = 0
            df = self.diagonal()
            sizes = {
                "s": len(self.species),
                "x": self.speedgrid.nx,
                "a": self.pitchgrid.na,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            df = df.reshape((-1, sizes[self.axorder[-1]]))
            if fmt == "dense":
                op = jax.vmap(jnp.diag)
            else:
                op = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
            return op(df)

        # nx is basically always small and these matrices are usually dense
        # so we always compute it using dense fmt and convert to banded at the
        # end if needed.
        if bw is None:
            bw = self.speedgrid.nx // 2

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.na,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        w = sfincs_w_speed(
            self.field,
            self.pitchgrid,
            self.Erho,
            self.speedgrid.x[None, :] * jnp.ones(len(self.species))[:, None],
        )
        df = self.speedgrid.Dx_pseudospectral[None, :, None, None, None, :]
        w1 = w[:, :, :, :, :, None]
        df = w1 * df
        idxa = self.pitchgrid.na // 2
        idxx = self.speedgrid.gauge_idx
        scale = jnp.mean(jnp.abs(w), axis=(2, 3, 4))[:, idxx] / jnp.mean(
            self.speedgrid.wx
        )
        df = jnp.where(
            self.gauge,
            df.at[:, idxx, idxa, 0, 0, :]
            .set(0, indices_are_sorted=True, unique_indices=True)
            .at[:, idxx, idxa, 0, 0, idxx]
            .set(scale, indices_are_sorted=True, unique_indices=True),
            df,
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, self.speedgrid.nx, self.speedgrid.nx))
        if fmt == "banded":
            df = dense_to_banded(bw, bw, df)
        return df

    @eqx.filter_jit
    @jax.named_scope("DKESpeed.block_diagonal2")
    def block_diagonal2(self):
        """Block diagonal of operator as (N,M,M) array. Unfolds s,x"""
        assert self.axorder[-2:] == "sx"
        if self.axorder[2] == "a":
            return _refold(self.block_diagonal(), len(self.species) * self.pitchgrid.na)
        elif self.axorder[2] == "t":
            return _refold(self.block_diagonal(), len(self.species) * self.field.ntheta)
        elif self.axorder[2] == "z":
            return _refold(self.block_diagonal(), len(self.species) * self.field.nzeta)
        else:
            # unreachable, just kept to appease type checker
            raise ValueError()  # pragma: no cover

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
                * self.pitchgrid.na
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
                * self.pitchgrid.na
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


class DKE(lx.AbstractLinearOperator):
    """Drift Kinetic Equation operator.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : AbstractPitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts
    background : list[LocalMaxwellian]
        Background species to include in the collision operator without solving for df.
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
    pitchgrid: AbstractPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    Erho: Float[Array, ""]
    background: list[LocalMaxwellian]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    operator_weights: jax.Array
    _opx: DKESpeed
    _opa: DKEPitch
    _opt: DKETheta
    _opz: DKEZeta
    _C: FokkerPlanckLandau

    def __init__(
        self,
        field: Field,
        pitchgrid: AbstractPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: Optional[list[LocalMaxwellian]] = None,
        potentials: Optional[RosenbluthPotentials] = None,
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        operator_weights: Optional[jax.Array] = None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if potentials is None:
            potentials = RosenbluthPotentials(speedgrid, species)
        self.potentials = potentials
        if background is None:
            background = []
        self.background = background
        self.Erho = jnp.array(Erho)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        if operator_weights is None:
            operator_weights = (
                jnp.ones(8).at[-1].set(0, indices_are_sorted=True, unique_indices=True)
            )
        self.operator_weights = jnp.asarray(operator_weights)

        self._opx = DKESpeed(field, pitchgrid, speedgrid, species, Erho, axorder, gauge)
        self._opa = DKEPitch(
            field, pitchgrid, speedgrid, species, Erho, p1, p2, axorder, gauge
        )
        self._opt = DKETheta(
            field, pitchgrid, speedgrid, species, Erho, p1, p2, axorder, gauge
        )
        self._opz = DKEZeta(
            field, pitchgrid, speedgrid, species, Erho, p1, p2, axorder, gauge
        )
        self._C = FokkerPlanckLandau(
            field,
            pitchgrid,
            speedgrid,
            species,
            background,
            potentials,
            p2,
            axorder,
            gauge,
            operator_weights=self.operator_weights[4:7],
        )

    @eqx.filter_jit
    @jax.named_scope("DKE.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f0 = self._opx.mv(vector)
        f1 = self._opa.mv(vector)
        f2 = self._opt.mv(vector)
        f3 = self._opz.mv(vector)
        f4 = self._C.mv(vector)
        return (
            self.operator_weights[0] * f0
            + self.operator_weights[1] * f1
            + self.operator_weights[2] * f2
            + self.operator_weights[3] * f3
            + f4
            + self.operator_weights[-1] * vector
        )

    @eqx.filter_jit
    @jax.named_scope("DKE.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        d0 = self._opx.diagonal()
        d1 = self._opa.diagonal()
        d2 = self._opt.diagonal()
        d3 = self._opz.diagonal()
        d4 = self._C.diagonal()
        return (
            self.operator_weights[0] * d0
            + self.operator_weights[1] * d1
            + self.operator_weights[2] * d2
            + self.operator_weights[3] * d3
            + d4
            + self.operator_weights[-1] * jnp.ones_like(d0)
        )

    @eqx.filter_jit
    @jax.named_scope("DKE.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        d0 = self._opx.block_diagonal(fmt, bw)
        d1 = self._opa.block_diagonal(fmt, bw)
        d2 = self._opt.block_diagonal(fmt, bw)
        d3 = self._opz.block_diagonal(fmt, bw)
        d4 = self._C.block_diagonal(fmt, bw)
        if fmt == "dense":
            eye = jnp.broadcast_to(jnp.identity(d0.shape[1]), d0.shape)
        else:
            eye = jnp.zeros_like(d0).at[:, d0.shape[1] // 2, :].set(1)
        return (
            self.operator_weights[0] * d0
            + self.operator_weights[1] * d1
            + self.operator_weights[2] * d2
            + self.operator_weights[3] * d3
            + d4
            + self.operator_weights[-1] * eye
        )

    @eqx.filter_jit
    @jax.named_scope("DKE.block_diagonal2")
    def block_diagonal2(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        d0 = self._opx.block_diagonal2()
        d1 = self._opa.block_diagonal2()
        d2 = self._opt.block_diagonal2()
        d3 = self._opz.block_diagonal2()
        d4 = self._C.block_diagonal2()
        eye = jnp.broadcast_to(jnp.identity(d0.shape[1]), d0.shape)
        return (
            self.operator_weights[0] * d0
            + self.operator_weights[1] * d1
            + self.operator_weights[2] * d2
            + self.operator_weights[3] * d3
            + d4
            + self.operator_weights[-1] * eye
        )

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
                * self.pitchgrid.na
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
                * self.pitchgrid.na
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


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
