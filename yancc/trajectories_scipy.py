"""MDKE as scipy sparse arrays."""

import jax
import numpy as np
import scipy

from yancc.finite_diff import fd2, fd_coeffs, fdbwd, fdfwd
from yancc.trajectories import dkes_w_pitch, dkes_w_theta, dkes_w_zeta


def dfdtheta(
    field,
    pitchgrid,
    E_psi,
    p="1a",
    gauge=False,
):
    """Advection operator in theta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    df : scipy sparse array
    """
    assert field.ntheta > fd_coeffs[1][p].size // 2

    w = np.array(dkes_w_theta(field, pitchgrid, E_psi).flatten())[:, None]
    h = 2 * np.pi / field.ntheta

    f = np.ones(field.ntheta)
    fd = scipy.sparse.csr_array(jax.jacfwd(fdfwd)(f, p, h=h, bc="periodic"))
    bd = scipy.sparse.csr_array(jax.jacfwd(fdbwd)(f, p, h=h, bc="periodic"))
    Iz = scipy.sparse.eye_array(field.nzeta)
    Ix = scipy.sparse.eye_array(pitchgrid.nxi)

    Af = scipy.sparse.kron(scipy.sparse.kron(Ix, fd), Iz)
    Ab = scipy.sparse.kron(scipy.sparse.kron(Ix, bd), Iz)
    df = w * ((w > 0) * Ab + (w <= 0) * Af)

    if gauge:
        idx = np.ravel_multi_index(
            (pitchgrid.nxi // 2, 0, 0),
            (pitchgrid.nxi, field.ntheta, field.nzeta),
            mode="clip",
        )
        mask = np.zeros(df.shape[0])
        mask[idx] = np.mean(np.abs(w)) / h
        df *= mask[:, None] == 0
        df += scipy.sparse.diags_array(mask)

    return df


def dfdzeta(
    field,
    pitchgrid,
    E_psi,
    p="1a",
    gauge=False,
):
    """Advection operator in zeta direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    E_psi : float
        Normalized electric field, E_psi/v
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    df : scipy sparse array
    """
    assert field.nzeta > fd_coeffs[1][p].size // 2
    w = np.array(dkes_w_zeta(field, pitchgrid, E_psi).flatten())[:, None]
    h = 2 * np.pi / field.nzeta / field.NFP

    f = np.ones(field.nzeta)
    fd = scipy.sparse.csr_array(jax.jacfwd(fdfwd)(f, p, h=h, bc="periodic"))
    bd = scipy.sparse.csr_array(jax.jacfwd(fdbwd)(f, p, h=h, bc="periodic"))
    It = scipy.sparse.eye_array(field.ntheta)
    Ix = scipy.sparse.eye_array(pitchgrid.nxi)

    Af = scipy.sparse.kron(scipy.sparse.kron(Ix, It), fd)
    Ab = scipy.sparse.kron(scipy.sparse.kron(Ix, It), bd)
    df = w * ((w > 0) * Ab + (w <= 0) * Af)

    if gauge:
        idx = np.ravel_multi_index(
            (pitchgrid.nxi // 2, 0, 0),
            (pitchgrid.nxi, field.ntheta, field.nzeta),
            mode="clip",
        )
        mask = np.zeros(df.shape[0])
        mask[idx] = np.mean(np.abs(w)) / h
        df *= mask[:, None] == 0
        df += scipy.sparse.diags_array(mask)

    return df


def dfdxi(
    field,
    pitchgrid,
    nu,
    p="1a",
    gauge=False,
):
    """Advection operator in xi/pitch direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    nu : float
        Normalized collisionality, nu/v
    p : str
        Stencil to use. Generally of the form "1a", "2b" etc. Number denotes
        formal order of accuracy, letter denotes degree of upwinding. "a" is fully
        upwinded, "b" and "c" if they exist are upwind biased but not fully.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    df : scipy sparse array
    """
    assert pitchgrid.nxi > fd_coeffs[1][p].size // 2
    w = np.array(dkes_w_pitch(field, pitchgrid).flatten())[:, None]
    h = np.pi / pitchgrid.nxi

    f = np.ones(pitchgrid.nxi)
    fd = scipy.sparse.csr_array(jax.jacfwd(fdfwd)(f, p, h=h, bc="symmetric"))
    bd = scipy.sparse.csr_array(jax.jacfwd(fdbwd)(f, p, h=h, bc="symmetric"))
    It = scipy.sparse.eye_array(field.ntheta)
    Iz = scipy.sparse.eye_array(field.nzeta)

    Af = scipy.sparse.kron(scipy.sparse.kron(fd, It), Iz)
    Ab = scipy.sparse.kron(scipy.sparse.kron(bd, It), Iz)
    df = w * ((w > 0) * Ab + (w <= 0) * Af)

    if gauge:
        idx = np.ravel_multi_index(
            (pitchgrid.nxi // 2, 0, 0),
            (pitchgrid.nxi, field.ntheta, field.nzeta),
            mode="clip",
        )
        mask = np.zeros(df.shape[0])
        mask[idx] = np.mean(np.abs(w)) / h
        df *= mask[:, None] == 0
        df += scipy.sparse.diags_array(mask)

    return df


def dfdpitch(
    field,
    pitchgrid,
    nu,
    p=2,
    gauge=False,
):
    """Diffusion operator in xi/pitch direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    nu : float
        Normalized collisionality, nu/v
    p : int
        Order of approximation for derivatives.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    df : scipy sparse array
    """
    assert pitchgrid.nxi > p
    h = np.pi / pitchgrid.nxi

    f = np.ones(pitchgrid.nxi)
    sina = np.sqrt(1 - pitchgrid.xi**2)
    cosa = -pitchgrid.xi

    f1 = scipy.sparse.csr_array(jax.jacfwd(fdfwd)(f, str(p) + "z", h=h, bc="symmetric"))
    f1 *= -(nu / 2 * cosa / sina)[:, None]
    f2 = scipy.sparse.csr_array(jax.jacfwd(fd2)(f, p, h=h, bc="symmetric"))
    f2 *= -nu / 2
    df = f1 + f2
    It = scipy.sparse.eye_array(field.ntheta)
    Iz = scipy.sparse.eye_array(field.nzeta)
    df = scipy.sparse.kron(scipy.sparse.kron(df, It), Iz)

    if gauge:
        idx = np.ravel_multi_index(
            (pitchgrid.nxi // 2, 0, 0),
            (pitchgrid.nxi, field.ntheta, field.nzeta),
            mode="clip",
        )
        mask = np.zeros(df.shape[0])
        mask[idx] = nu / h**2
        df *= mask[:, None] == 0
        df += scipy.sparse.diags_array(mask)

    return df


def mdke(
    field,
    pitchgrid,
    E_psi,
    nu,
    p1="1a",
    p2=2,
    gauge=False,
):
    """MDKE operator.

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
    p1 : int
        Order of approximation for first derivatives.
    p2 : int
        Order of approximation for second derivatives.
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.

    Returns
    -------
    df : scipy sparse array

    """
    dt = dfdtheta(field, pitchgrid, E_psi, p1, gauge=gauge)
    dz = dfdzeta(field, pitchgrid, E_psi, p1, gauge=gauge)
    di = dfdxi(field, pitchgrid, nu, p1, gauge=gauge)
    dp = dfdpitch(field, pitchgrid, nu, p2, gauge=gauge)
    return dt + dz + di + dp
