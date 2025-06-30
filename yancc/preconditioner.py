"""Stuff for preconditioners."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from .field import Field
from .multigrid import MultigridOperator, get_multigrid_preconditioner
from .species import LocalMaxwellian, collisionality
from .velocity_grids import SpeedGrid, UniformPitchAngleGrid


class DKEPreconditioner(lx.AbstractLinearOperator):
    """Preconditioner for the DKE using block diagonal MDKE preconditioners."""

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    E_psi: Float[Array, ""]
    M: MultigridOperator
    vs: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        E_psi: Float[ArrayLike, ""],
        **options
    ):

        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.E_psi = jnp.asarray(E_psi)

        Ers = []
        nus = []
        vs = []
        for spec in species:
            temp_nu = []
            temp_Er = []
            temp_vs = []
            for x in speedgrid.x:
                v = x * spec.v_thermal
                nu = collisionality(spec, v, *species)
                Erhat = E_psi / v
                nuhat = nu / v
                temp_Er.append(Erhat)
                temp_nu.append(nuhat)
                temp_vs.append(v)

            Ers.append(temp_Er)
            nus.append(temp_nu)
            vs.append(temp_vs)

        Ers = jnp.array(Ers)
        nus = jnp.array(nus)
        self.vs = jnp.array(vs)

        def get_mdke_precond(nu, E_psi):
            return get_multigrid_preconditioner(
                field=field,
                E_psi=E_psi,
                nu=nu,
                nl=pitchgrid.nxi,
                nt=field.ntheta,
                nz=field.nzeta,
                p1=options.get("p1a", "2d"),
                p2=options.get("p2a", 2),
                cycle_index=options.get("cycle_index", 1),
                v1=options.get("v1", -1),
                v2=options.get("v2", -1),
                smooth_weights=options.get("smooth_weights", None),
                interp_method=options.get("interp_method", "linear"),
                smooth_method=options.get("smooth_method", "standard"),
                smooth_solver=options.get("smooth_solver", "dense"),
                coarse_overweight=options.get("coarse_overweight", 2),
                coarse_N=options.get("coarse_N", 8000),
                coarsening=options.get("coarsening", 2),
                gauge=options.get("gauge", True),
            )

        self.M = jax.vmap(jax.vmap(get_mdke_precond))(nus, Ers)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix-vector product."""
        vector = vector.reshape((len(self.species), self.speedgrid.nx, -1))

        def _mv(M, v):
            return M.mv(v)

        out = jax.vmap(jax.vmap(_mv))(self.M, vector)
        out = out / self.vs[:, :, None]
        return out.flatten()

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


@lx.is_symmetric.register(DKEPreconditioner)
@lx.is_diagonal.register(DKEPreconditioner)
@lx.is_tridiagonal.register(DKEPreconditioner)
def _(operator):
    return False
