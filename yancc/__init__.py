"""yancc: Yet Another NeoClassical Code."""

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

__version__ = _version.get_versions()["version"]
