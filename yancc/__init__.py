"""yancc: Yet Another NeoClassical Code."""

from jax import config

from . import (
    _version,
    collisions,
    field,
    krylov,
    linalg,
    misc,
    multigrid,
    smoothers,
    species,
    trajectories,
    velocity_grids,
)
from .solve import solve_dke, solve_mdke

# lots of big and small numbers that will over/underflow in 32bit so
# we set this here to ensure we don't get nans elsewhere.
config.update("jax_enable_x64", True)

__version__ = _version.get_versions()["version"]
