"""Preconditioner for the DKE in axisymmetric fields."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from ._fluid import FluidOperator, _stencil_halfwidth
from ._linalg import (
    BorderedOperator,
    TransposedLinearOperator,
    _bordered_as_matrix,
    block_tridiag_factor,
    block_tridiag_solve,
)
from ._trajectories import DKE


def _moment_basis(A: DKE, moments: int):
    """Legendre modes in pitch at each speed, and the projection onto them.

    Returns shapes and moments, both (ns, nx*moments, nx, na), in the format of the
    fluid basis: shape ``x*moments + l`` is the Legendre mode l at speed node x.
    """
    ns, nx = len(A.species), A.speedgrid.nx
    na = A.pitchgrid.nalpha
    Txi = A._C.CF.Txi[:, :moments]
    Txi_inv = A._C.CF.Txi_inv[:moments]
    eye = jnp.eye(nx)
    V = jnp.einsum("yx,al->ylxa", eye, Txi).reshape(nx * moments, nx, na)
    Psi = jnp.einsum("yx,la->ylxa", eye, Txi_inv).reshape(nx * moments, nx, na)
    shape = (ns, nx * moments, nx, na)
    return jnp.broadcast_to(V, shape), jnp.broadcast_to(Psi, shape)


def _pitch_groups(na, g):
    """Split the pitch nodes into consecutive groups of ``g`` nodes.

    Returns an (m, g) index array of nodes, -1 marking padding. Nodes more than ``g``
    apart are uncoupled, so only neighbouring groups couple.
    """
    m = -(-na // g)
    groups = -np.ones(m * g, dtype=int)
    groups[:na] = np.arange(na)
    return groups.reshape(m, g)


class _PitchThetaSolver(eqx.Module):
    """Solver for the DKE on the (pitch, theta) plane of each species and speed.

    The plane problem keeps the advection in theta and pitch, pitch angle scattering
    and the diagonal of the terms that couple speeds and species. It is block
    tridiagonal over groups of pitch nodes, with dense blocks in theta.
    """

    factors: tuple
    groups: tuple = eqx.field(static=True)
    shape: tuple = eqx.field(static=True)

    def __init__(self, A: DKE):
        ns, nx = len(A.species), A.speedgrid.nx
        na, nt, nz = A.pitchgrid.nalpha, A.field.ntheta, A.field.nzeta
        assert nz == 1
        self.shape = (ns, nx, na, nt, nz)
        w = A.operator_weights
        groups = _pitch_groups(na, _stencil_halfwidth(A))
        self.groups = tuple(map(tuple, groups.tolist()))
        m, g = groups.shape
        valid = jnp.asarray(groups >= 0)
        idx = np.maximum(groups, 0)
        eye_t, eye_g = jnp.eye(nt), jnp.eye(g)

        # theta advection, upwinded by the sign of the wind, and the diagonal of the
        # speed advection, energy scattering and field particle terms: (ns, nx, na,
        # nt, nt)
        wt = A._opt._w[..., 0]
        stencil = jnp.where(wt[..., None] > 0, A._opt._bd, A._opt._fd)
        shift = (
            w[0] * A._opx.diagonal()
            + w[5] * A._C.CE.diagonal()
            + w[6] * A._C.CF.diagonal()
            + w[7]
        ).reshape(ns, nx, na, nt)
        T = w[2] * wt[..., None] * stencil + shift[..., None] * eye_t
        # pitch advection and pitch angle scattering: (ns, nx, nt, na, na)
        P = w[1] * A._opa.block_diagonal("dense", None, "sxtza") + w[
            4
        ] * A._C.CL.block_diagonal("dense", None, "sxtza")
        P = P.reshape(ns, nx, nt, na, na)

        def coupling(T, P, offset):
            """(m, g nt, g nt) blocks coupling group k + offset into group k."""
            rows, cols = idx, np.roll(idx, -offset, axis=0)
            both = valid[:, :, None] & jnp.roll(valid, -offset, axis=0)[:, None, :]
            blk = jnp.einsum(
                "tkij,tu->kitju", P[:, rows[:, :, None], cols[:, None, :]], eye_t
            )
            blk = jnp.where(both[:, :, None, :, None], blk, 0.0)
            if offset == 0:
                # padded unknowns are decoupled, with identity rows
                Tk = jnp.where(valid[:, :, None, None], T[idx], eye_t)
                blk = blk + jnp.einsum("kitu,ij->kitju", Tk, eye_g)
            return blk.reshape(m, g * nt, g * nt)

        def factor(T, P):
            return block_tridiag_factor(
                coupling(T, P, 0), coupling(T, P, -1), coupling(T, P, 1)
            )

        self.factors = jax.vmap(factor)(
            T.reshape(ns * nx, na, nt, nt), P.reshape(ns * nx, nt, na, na)
        )

    def solve(self, f):
        """Solve the plane problems for a right hand side of the size of f."""
        ns, nx, na, nt, _ = self.shape
        groups = np.array(self.groups)
        m, g = groups.shape
        r = f.reshape(ns * nx, na, nt)[:, np.maximum(groups, 0)]  # (ns nx, m, g, nt)
        r = jnp.where(jnp.asarray(groups >= 0)[:, :, None], r, 0.0)
        x = jax.vmap(block_tridiag_solve)(self.factors, r.reshape(ns * nx, m, g * nt))
        x = x.reshape(ns * nx, m * g, nt)[:, np.flatnonzero(groups.reshape(-1) >= 0)]
        return x.reshape(f.shape)


class TokamakPreconditioner(lx.AbstractLinearOperator):
    """Two level preconditioner for the bordered DKE in an axisymmetric field.

    The coarse level is the Galerkin projection of the bordered operator on the first
    Legendre modes in pitch at every poloidal point, speed and species, together with
    the sources. These hold the null space of the collision operator, which the
    constraints remove, and the coupling between speeds and species that dominates at
    high collisionality. The residual of the coarse solution is then corrected by a
    direct solve on the (pitch, theta) plane of each species and speed, with advection
    in theta and pitch, pitch angle scattering and the diagonal of the other terms.

    Parameters
    ----------
    operator : BorderedOperator
        Bordered DKE operator, with an ungauged DKE on a field with ``nzeta=1``.
    moments : int
        Number of Legendre modes in the coarse level. At least 3 are needed: streaming
        and the mirror force leave a parallel flow undetermined until the l=2 mode,
        which carries the viscosity, is kept.
    """

    operator: BorderedOperator
    coarse: FluidOperator
    planes: _PitchThetaSolver
    moments: int = eqx.field(static=True)

    def __init__(self, operator: BorderedOperator, moments: int = 4):
        A = operator.A
        assert isinstance(A, DKE)
        if A.field.nzeta != 1:
            raise ValueError("TokamakPreconditioner requires a field with nzeta=1.")
        nL = A._C.CF.Txi.shape[1]
        if not 3 <= moments <= nL:
            raise ValueError(f"moments must be between 3 and nL={nL}, got {moments}.")
        self.operator = operator
        self.moments = moments
        self.coarse = FluidOperator(
            operator,
            A.species,
            A.speedgrid,
            A.pitchgrid,
            A.field,
            weights=jnp.ones(len(A.species)),
            basis=_moment_basis(A, moments),
            pins=(0, moments),
        )
        self.planes = _PitchThetaSolver(A)

    def mv(self, vector):
        """Matrix vector product."""
        f, h = self.coarse.mv(vector)
        Af, _ = self.operator.mv((f, h))
        return (f + self.planes.solve(vector[0] - Af), h)

    def print_resolution_summary(self) -> None:
        """Print the sizes of the two levels."""
        A = self.operator.A
        assert isinstance(A, DKE)
        ns, nx = len(A.species), A.speedgrid.nx
        na, nt = A.pitchgrid.nalpha, A.field.ntheta
        jax.debug.print(
            f"Moments: ns={ns:4d}, nx={nx:4d}, nl={self.moments:4d}, nθ={nt:4d}, "
            f"N={ns * nx * self.moments * nt:,d}",
            ordered=True,
        )
        jax.debug.print(
            f"Planes:  ns={ns:4d}, nx={nx:4d}, nα={na:4d}, nθ={nt:4d}, N={na * nt:,d}",
            ordered=True,
        )

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        return _bordered_as_matrix(self)

    def in_structure(self):
        """Pytree structure of expected input."""
        return self.operator.in_structure()

    def out_structure(self):
        """Pytree structure of expected output."""
        return self.operator.in_structure()

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


@lx.is_symmetric.register(TokamakPreconditioner)
@lx.is_diagonal.register(TokamakPreconditioner)
@lx.is_tridiagonal.register(TokamakPreconditioner)
@lx.is_positive_semidefinite.register(TokamakPreconditioner)
@lx.is_negative_semidefinite.register(TokamakPreconditioner)
def _(operator):
    return False
