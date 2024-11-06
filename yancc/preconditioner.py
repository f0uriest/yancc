"""Preconditioner for DKE."""

import functools

import cola
import jax.numpy as jnp
import monkes
from monkes import Field, LocalMaxwellian

from .velocity_grids import PitchAngleGrid, SpeedGrid


class MONKESPreconditioner(cola.ops.LinearOperator):
    """Preconditioner for full DKE based on monoenergetic approximation using MONKES.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    E_psi : float
        Radial electric field.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        E_psi: float,
    ):

        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.E_psi = E_psi
        self.operators = []
        self._Clus = []

        for spec in species:
            operators = []
            Clus = []
            for x in speedgrid.x:
                v = x * spec.v_thermal
                nu = monkes._species.collisionality(spec, v, *species)
                Erhat = E_psi / v
                nuhat = nu / v
                op = monkes._core.MonoenergeticDKOperator(
                    field, pitchgrid.nxi, Erhat, nuhat
                )
                Clu = monkes._linalg.block_tridiagonal_factor(
                    op.D, op.L, op.U, reverse=True
                )
                operators.append(op)
                Clus.append(Clu)

            self.operators.append(operators)
            self._Clus.append(Clus)

    @functools.partial(jnp.vectorize, excluded=[0], signature="(n)->(n)")
    def _matmat(self, x):
        f = x[
            : len(self.species)
            * self.field.ntheta
            * self.field.nzeta
            * self.speedgrid.nx
            * self.pitchgrid.nxi
        ]
        rest = x[len(f) :]
        f = f.reshape(
            (
                len(self.species),
                self.speedgrid.nx,
                self.pitchgrid.nxi,
                self.field.ntheta,
                self.field.nzeta,
            )
        )
        # convert to modal in xi and nodal in x
        f = jnp.einsum("li, skitz->skltz", self.pitchgrid.xivander_inv, f)
        f = jnp.einsum("xk, skltz->sxltz", self.speedgrid.xvander, f)
        # multiply by mono-energetic DK operator inverse
        # for each species and velocity
        out = jnp.zeros_like(f)
        for i in range(self.ns):
            for j in range(self.nx):
                Cluij = self._Clus[i][j]
                fij = f[i, j].flatten()
                outij = monkes._linalg.block_tridiagonal_solve(Cluij, fij)
                # make f have mean 0.
                fsa = self.field.flux_surface_average(outij[:, 0])
                outij = outij.at[:, 0].add(-fsa[:, None, None])
                out = out.at[i, j].set(outij.reshape(*f[i, j].shape))
        # convert modal back to nodal
        out = jnp.einsum("il, sxltz->sxitz", self.pitchgrid.xivander, out)
        out = jnp.concatenate([out.flatten(), rest])
        return out
