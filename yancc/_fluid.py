"""Galerkin correction of a DKE preconditioner on the fluid subspace."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from ._finite_diff import fd_coeffs
from ._linalg import (
    TransposedLinearOperator,
    _ruiz_scale,
    block_tridiag_periodic_factor,
    block_tridiag_periodic_solve,
)
from .species import _nustar_species


def _fluid_basis(species, speedgrid, pitchgrid):
    """Fluid velocity shapes and conservation moments, both (ns, 3, nx, na).

    The shapes are F, (x^2 - a) F and xi x F for each species' Maxwellian F, with a
    chosen to make the first two orthogonal under the density moment. The moments
    1, x^2 - a and xi x are combined so that moment i of shape j is delta_ij.
    """
    x, xi = speedgrid.x, pitchgrid.xi
    wv = (x**2 * speedgrid.wx)[:, None] * pitchgrid.wxi[None, :]
    one = jnp.ones_like(xi)
    V, Psi = [], []
    for sp in species:
        F = sp(x * sp.v_thermal)
        a = jnp.sum(x**4 * speedgrid.wx * F) / jnp.sum(x**2 * speedgrid.wx * F)
        v = jnp.stack(
            [
                F[:, None] * one,
                ((x**2 - a) * F)[:, None] * one,
                (x * F)[:, None] * xi[None, :],
            ]
        )
        psi = jnp.stack(
            [
                jnp.ones_like(wv),
                (x**2 - a)[:, None] * one,
                x[:, None] * xi[None, :],
            ]
        )
        G = jnp.einsum("ixa,xa,jxa->ij", psi, wv, v)
        V.append(v)
        Psi.append(jnp.einsum("ij,jxa->ixa", jnp.linalg.inv(G), psi * wv))
    return jnp.stack(V), jnp.stack(Psi)


# Collisionality at which a species gets half the fluid correction, and how sharply the
# weight switches between 0 and 1 around it. Below the crossover the fluid moments are
# not slow modes of the operator, and correcting them exactly slows the Krylov solve.
_NUSTAR_CROSSOVER = 20.0
_NUSTAR_POWER = 2.0
# Weights below this are set to 0. At low collisionality the fluid solve has a large
# norm, so even a small weight slows the Krylov solve.
_MIN_WEIGHT = 5e-2


def fluid_weights(species, field, background=(), coulomb_log=None) -> jax.Array:
    """Weight of the fluid correction for each species, from its collisionality.

    Goes smoothly from 0 for collisionless species to 1 for collisional ones, and is
    exactly 0 for species whose weight would be small.

    Parameters
    ----------
    species : list[LocalMaxwellian]
        Species being solved for.
    field : Field
        Magnetic field information.
    background : list[LocalMaxwellian]
        Background species, included in each species' collisionality.
    coulomb_log : float, optional
        Coulomb logarithm override.

    Returns
    -------
    weights : jax.Array, shape (ns,)
    """
    # The fluid moments are Maxwellian weighted averages, carried by particles near the
    # thermal speed, so whether they are slow modes depends on the collisionality at
    # x = 1. Speed nodes far from it are more or less collisional, but hold little of
    # these moments however many of them the grid has.
    nu = _nustar_species(species, field, 1.0, list(background), lnlambda=coulomb_log)
    r = (nu / _NUSTAR_CROSSOVER) ** _NUSTAR_POWER
    w = r / (1 + r)
    return jnp.where(w >= _MIN_WEIGHT, w, 0.0)


def _local_mv(A, f):
    """Terms of the DKE operator that act pointwise on the flux surface."""
    w = A.operator_weights
    return w[0] * A._opx.mv(f) + w[1] * A._opa.mv(f) + A._C._local_mv(f) + w[-1] * f


def _stencil_halfwidth(op):
    """Largest finite difference offset along the surface of a DKE operator."""
    return max(fd_coeffs[1][op.p1].size // 2, fd_coeffs[2][op.p2].size // 2)


def _line_blocks(op, axis, weight, V, Psi):
    """Fluid projection of an operator that couples points along one surface line.

    ``op`` is the theta (``axis="t"``) or zeta (``axis="z"``) advection term of an
    ungauged DKE. Returns ``(ns, 3, 3, n_other, n_line, n_line)`` dense line blocks,
    with rows indexed by the fluid moment and columns by the fluid shape.
    """
    # The term is w times an upwind difference along the line, the backward stencil
    # where w > 0 and the forward one elsewhere. The difference acts on the line
    # coordinate only, so the projection only needs the fluid moments of the positive
    # and negative parts of w, each multiplying its (dense, periodic) stencil matrix.
    w = op._w  # (ns, nx, na, nt, nz)
    wp = jnp.where(w > 0, w, 0.0)
    ap = jnp.einsum("spxa,sixa,sxatz->spitz", Psi, V, wp)
    am = jnp.einsum("spxa,sixa,sxatz->spitz", Psi, V, w - wp)
    if axis == "t":
        blk = jnp.einsum("spitz,tu->spiztu", ap, op._bd) + jnp.einsum(
            "spitz,tu->spiztu", am, op._fd
        )
    else:
        blk = jnp.einsum("spitz,zu->spitzu", ap, op._bd) + jnp.einsum(
            "spitz,zu->spitzu", am, op._fd
        )
    return weight * blk


def _plane_groups(nz, bw):
    """Split periodic zeta planes into consecutive groups of at least ``bw`` planes.

    Returns an (m, g) index array of planes, -1 marking padding. Planes more than
    ``bw`` apart are uncoupled, so with m >= 3 groups only neighbouring groups couple
    and the system is periodic block tridiagonal. Otherwise one group holds every
    plane and the system is solved densely.
    """
    m = nz // max(bw, 1)
    if m < 3:
        return np.arange(nz)[None, :]
    sizes = np.full(m, nz // m)
    sizes[: nz % m] += 1
    groups = -np.ones((m, sizes.max()), dtype=int)
    start = 0
    for k, size in enumerate(sizes):
        groups[k, :size] = np.arange(start, start + size)
        start += size
    return groups


class _FluidSolver(eqx.Module):
    """Solver for the Galerkin projection of the bordered DKE onto the fluid space.

    Fluid unknowns are the amplitudes of each species' fluid shapes at every point of
    the flux surface, plus the bordered source unknowns. The projected operator is
    split into a part that is periodic block tridiagonal over groups of zeta planes,
    with a few pinned diagonal entries removing its null space, plus low rank terms
    (sources, constraints, pins and the surface averaged collisional exchange) handled
    through a Schur complement.
    """

    V: jax.Array
    Psi: jax.Array
    factors: tuple
    Bx: jax.Array
    Cx: jax.Array
    Z: jax.Array
    S_lu: tuple
    groups: tuple = eqx.field(static=True)
    shape: tuple = eqx.field(static=True)
    nborder: int = eqx.field(static=True)
    dense: bool = eqx.field(static=True)

    def __init__(self, A, B, C, species, speedgrid, pitchgrid, field):
        ns, nx = len(species), speedgrid.nx
        na, nt, nz = pitchgrid.nalpha, field.ntheta, field.nzeta
        self.shape = (ns, nx, na, nt, nz)
        self.nborder = B.in_size()
        self.V, self.Psi = _fluid_basis(species, speedgrid, pitchgrid)
        V, Psi = self.V, self.Psi
        w = A.operator_weights

        # projected local terms: the response to a surface constant fluid shape gives
        # the (ns 3, ns 3) block at every point
        def probe(e):
            c = jnp.broadcast_to(e.reshape(ns, 3, 1, 1), (ns, 3, nt, nz))
            return self.restrict(_local_mv(A, self.prolong(c)))

        # (s, i, s', i', t, z) for column (s, i) and row (s', i')
        Kloc = jax.vmap(probe)(jnp.eye(3 * ns)).reshape(ns, 3, ns, 3, nt, nz)
        # line terms: (s, i', i, z, t, t') and (s, i', i, t, z, z')
        Kt = _line_blocks(A._opt, "t", w[2], V, Psi)
        Kz = _line_blocks(A._opz, "z", w[3], V, Psi)

        bw = _stencil_halfwidth(A)
        groups = _plane_groups(nz, bw if nz > 1 else 1)
        self.groups = tuple(map(tuple, groups.tolist()))
        self.dense = groups.shape[0] == 1
        m, g = groups.shape
        b0 = ns * 3 * nt
        eye_s, eye_t = jnp.eye(ns), jnp.eye(nt)

        # in plane blocks (z, s', i', t, s, i, t'), without the zeta term
        Dp = jnp.einsum("sipjtz,tu->zpjtsiu", Kloc, eye_t) + jnp.einsum(
            "sjizth,sp->zpjtsih", Kt, eye_s
        )

        def coupling(zr, zc, pad):
            """(b0, b0) block coupling plane zc into plane zr (-1 for padding)."""
            valid = (zr >= 0) & (zc >= 0)
            zr_, zc_ = jnp.maximum(zr, 0), jnp.maximum(zc, 0)
            blk = jnp.einsum(
                "sjit,sp,tu->pjtsiu", Kz[:, :, :, :, zr_, zc_], eye_s, eye_t
            )
            blk = blk + jnp.where(zr_ == zc_, 1.0, 0.0) * Dp[zr_]
            blk = blk.reshape(b0, b0)
            return jnp.where(valid, blk, 0.0) + pad * jnp.eye(b0)

        def group_blocks(offsets):
            """(len(offsets), m, g b0, g b0) blocks coupling group k + offset into k.

            All blocks are built by a single vmapped call, so the block assembly is
            traced once however many groups there are.
            """
            kr = np.tile(np.arange(m), len(offsets))
            kc = (kr + np.repeat(offsets, m)) % m
            rows, cols = groups[kr], groups[kc]  # (nblk, g)
            # padded unknowns are decoupled, with identity rows
            pad = (kr == kc)[:, None, None] * np.eye(g) * (rows < 0)[:, :, None]
            zr = np.repeat(rows, g, axis=1)
            zc = np.tile(cols, (1, g))
            blocks = jax.vmap(coupling)(
                jnp.asarray(zr.reshape(-1)),
                jnp.asarray(zc.reshape(-1)),
                jnp.asarray(pad.reshape(-1)),
            )
            blocks = blocks.reshape(len(offsets), m, g, g, b0, b0)
            return blocks.transpose(0, 1, 2, 4, 3, 5).reshape(
                len(offsets), m, g * b0, g * b0
            )

        # diagonal blocks first, then the sub- and super-diagonal ones if needed
        blocks = group_blocks([0] if self.dense else [0, -1, 1])
        D = blocks[0]

        # pin the fluid density and energy of each species at one point, removing the
        # null space of the ungauged operator; the pins are undone in the Schur step
        pins = np.array([s * 3 * nt + i * nt for s in range(ns) for i in (0, 1)])
        tau = jnp.max(jnp.abs(D[0, pins]), axis=-1)
        D = D.at[0, pins, pins].add(tau)
        N = m * g * b0

        if self.dense:
            r, c = _ruiz_scale(D[0])
            lu = jax.scipy.linalg.lu_factor(r[:, None] * D[0] * c[None, :])
            self.factors = (lu, r, c)
        else:
            self.factors = block_tridiag_periodic_factor(D, blocks[1], blocks[2])

        # low rank border: sources/constraints, pins and the surface exchange term
        Bm = B.as_matrix().T.reshape(-1, *self.shape)  # (nb, ns, nx, na, nt, nz)
        Cm = C.as_matrix().reshape(-1, *self.shape)
        Bf = jax.vmap(self.restrict)(Bm.reshape(Bm.shape[0], -1))
        Cf = jnp.einsum("sixa,bsxatz->bsitz", V, Cm)
        pin_vec = jnp.zeros((len(pins), N)).at[jnp.arange(len(pins)), pins].set(1.0)
        Bx = [jax.vmap(self._to_solver)(Bf), -tau[:, None] * pin_vec]
        Cx = [jax.vmap(self._to_solver)(Cf), pin_vec]
        Dx = [jnp.zeros(self.nborder), -jnp.ones(len(pins))]
        if A._C._exchange is not None:
            # The exchange term maps the surface average of the fluid amplitudes through
            # a (ns 3, ns 3) block G0 to the same amplitudes at every point, so it is
            # G0 (1 <.>), of rank 3 ns.
            def exchange(e):
                c = jnp.broadcast_to(e.reshape(ns, 3, 1, 1), (ns, 3, nt, nz))
                return self.restrict(A._C._surface_exchange(self.prolong(c)))

            G0 = jax.vmap(exchange)(jnp.eye(3 * ns))[..., 0, 0]
            U = jnp.broadcast_to(
                G0.reshape(3 * ns, ns, 3, 1, 1), (3 * ns, ns, 3, nt, nz)
            )
            Vt = jnp.einsum(
                "ij,tz->ijtz", jnp.eye(3 * ns), A._C._surface_weights
            ).reshape(3 * ns, ns, 3, nt, nz)
            Bx.append(jax.vmap(self._to_solver)(U))
            Cx.append(jax.vmap(self._to_solver)(Vt))
            Dx.append(-jnp.ones(3 * ns))
        self.Bx, self.Cx = jnp.concatenate(Bx), jnp.concatenate(Cx)
        self.Z = jax.vmap(self._solve_sparse)(self.Bx)
        S = jnp.diag(jnp.concatenate(Dx)) - self.Cx @ self.Z.T
        self.S_lu = jax.scipy.linalg.lu_factor(S)

    def prolong(self, c):
        """Fluid amplitudes (ns, 3, nt, nz) to a distribution function."""
        return jnp.einsum("sixa,sitz->sxatz", self.V, c).reshape(-1)

    def restrict(self, f):
        """Distribution function to fluid moments (ns, 3, nt, nz)."""
        return jnp.einsum("sixa,sxatz->sitz", self.Psi, f.reshape(self.shape))

    def _to_solver(self, c):
        """Fluid amplitudes (ns, 3, nt, nz) to solver order (group, plane, s, i, t)."""
        groups = np.array(self.groups)
        u = jnp.moveaxis(c, -1, 0)[np.maximum(groups, 0)]  # (m, g, ns, 3, nt)
        u = jnp.where(jnp.asarray(groups >= 0)[..., None, None, None], u, 0.0)
        return u.reshape(-1)

    def _from_solver(self, u):
        """Inverse of _to_solver, dropping padding."""
        ns, _, _, nt, _ = self.shape
        z = np.array(self.groups).reshape(-1)
        c = u.reshape(-1, ns, 3, nt)[np.flatnonzero(z >= 0)]
        c = c[np.argsort(z[z >= 0])]
        return jnp.moveaxis(c, 0, -1)

    def _solve_sparse(self, v):
        if self.dense:
            lu, r, c = self.factors
            return c * jax.scipy.linalg.lu_solve(lu, r * v)
        m = len(self.groups)
        return block_tridiag_periodic_solve(self.factors, v.reshape(m, -1)).reshape(-1)

    def solve(self, c, h):
        """Solve the bordered fluid system for rhs (c, h)."""
        u0 = self._solve_sparse(self._to_solver(c))
        rhs = jnp.zeros(self.Bx.shape[0]).at[: self.nborder].set(h)
        z = jax.scipy.linalg.lu_solve(self.S_lu, rhs - self.Cx @ u0)
        u = u0 - self.Z.T @ z
        return self._from_solver(u), z[: self.nborder]


class FluidOperator(lx.AbstractLinearOperator):
    """Galerkin fluid space approximation to the inverse of the bordered DKE.

    Applies ``F = P W K^-1 R``. ``P`` maps fluid density, energy and parallel momentum
    perturbations at every point of the flux surface (and the sources) to distribution
    functions, ``R`` takes the corresponding conservation moments, ``K = R A P``, and
    ``W`` scales the fluid amplitudes and sources of each species by its weight. A
    preconditioner ``M`` for ``A`` is corrected on the fluid space by composing
    ``F + M (I - A F)``.

    Parameters
    ----------
    A : BorderedOperator
        Bordered DKE operator, with an ungauged DKE.
    species, speedgrid, pitchgrid, field
        Discretization of ``A``.
    weights : array-like, optional
        Weight of the fluid correction for each species, between 0 (no correction)
        and 1. Defaults to weights from each species' collisionality.
    background : list of LocalMaxwellian, optional
        Background species, used for the default weights.
    coulomb_log : float, optional
        Coulomb logarithm, used for the default weights.
    """

    fluid: _FluidSolver
    weights: jax.Array
    structure: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(
        self,
        A,
        species,
        speedgrid,
        pitchgrid,
        field,
        weights=None,
        background=(),
        coulomb_log=None,
    ):
        if weights is None:
            weights = jax.lax.stop_gradient(
                fluid_weights(species, field, background, coulomb_log)
            )
        self.weights = jnp.asarray(weights)
        assert self.weights.shape == (len(species),)
        self.structure = A.in_structure()
        # The ungauged DKE is singular on the fluid density and energy of each species,
        # and the sources and constraints of the bordered system are what make it
        # invertible, so the fluid projection includes them. This makes F the exact
        # Galerkin inverse of the operator the Krylov solver sees.
        self.fluid = _FluidSolver(A.A, A.B, A.C, species, speedgrid, pitchgrid, field)

    def mv(self, vector):
        """Matrix vector product."""
        fluid = self.fluid
        n = int(np.prod(fluid.shape))
        # each species has 3 fluid amplitudes per point and 2 sources
        wc = self.weights[:, None, None, None]
        wh = jnp.repeat(self.weights, fluid.nborder // self.weights.size)
        c, h = fluid.solve(fluid.restrict(vector[:n]), vector[n:])
        return jnp.concatenate([fluid.prolong(wc * c), wh * h])

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv, out_axes=-1)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return self.structure

    def out_structure(self):
        """Pytree structure of expected output."""
        return self.structure

    def transpose(self):
        """Transpose of the operator."""
        return TransposedLinearOperator(self)


@lx.is_symmetric.register(FluidOperator)
@lx.is_diagonal.register(FluidOperator)
@lx.is_tridiagonal.register(FluidOperator)
@lx.is_positive_semidefinite.register(FluidOperator)
@lx.is_negative_semidefinite.register(FluidOperator)
def _(operator):
    return False
