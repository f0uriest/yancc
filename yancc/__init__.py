"""yancc: Yet Another NeoClassical Code."""

from jax import config

from . import _version, field, solution, solve, species, velocity_grids
from .field import Field
from .solution import DKESolution, MDKESolution
from .solve import solve_dke, solve_dke_ambipolar, solve_mdke
from .species import (
    Beryllium,
    Beryllium9,
    Boron,
    Boron10,
    Boron11,
    Deuterium,
    Electron,
    GlobalMaxwellian,
    Helium,
    Helium4,
    Hydrogen,
    Lithium,
    Lithium6,
    Lithium7,
    LocalMaxwellian,
    Nitrogen,
    Nitrogen14,
    Oxygen,
    Oxygen16,
    Species,
    Tritium,
)
from .velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

# lots of big and small numbers that will over/underflow in 32bit so
# we set this here to ensure we don't get nans elsewhere.
config.update("jax_enable_x64", True)

__version__ = _version.get_versions()["version"]

__all__ = [
    "__version__",
    # solvers
    "solve_dke",
    "solve_dke_ambipolar",
    "solve_mdke",
    # fields and velocity grids
    "Field",
    "MaxwellSpeedGrid",
    "UniformPitchAngleGrid",
    # species
    "Species",
    "LocalMaxwellian",
    "GlobalMaxwellian",
    "Electron",
    "Hydrogen",
    "Deuterium",
    "Tritium",
    "Helium4",
    "Helium",
    "Lithium6",
    "Lithium7",
    "Lithium",
    "Beryllium9",
    "Beryllium",
    "Boron10",
    "Boron11",
    "Boron",
    "Nitrogen14",
    "Nitrogen",
    "Oxygen16",
    "Oxygen",
    # solutions
    "DKESolution",
    "MDKESolution",
    # public modules
    "field",
    "solution",
    "solve",
    "species",
    "velocity_grids",
]
