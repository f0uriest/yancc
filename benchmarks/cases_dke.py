"""Case matrix for the DKE convergence benchmark.

Each :class:`Case` fully specifies a drift-kinetic benchmark solve: equilibrium, species
count, target ion collisionality ``nustar``, target ion normalized electric field
``estar`` (E* at x=1), grid resolution, and the per-problem tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from yancc.field import Field
from yancc.species import Electron, Estar, Hydrogen, LocalMaxwellian
from yancc.species import nustar as _nustar_of

# Fixed profile: temperature, Coulomb log, and normalized inverse gradient scale
# lengths. nu* depends only on density and T (not the gradients), so a target
# collisionality can be root-found in density alone; the gradients set the
# drift-kinetic drive, and the *iteration count* is mostly insensitive to it, so these
# representative core values just give a nonzero, field-independent RHS.
TEMP_EV = 3.0e3
LNLAMBDA = 17.0
ALT = 1.0  # a/L_T = -(a/T) dT/dr
ALN = 0.3  # a/L_n = -(a/n) dn/dr

# name -> ("desc"|"vmec"|"booz", reference). DESC names are uniform across machines;
# vmec/booz entries point at bundled test files (no desc dependency).
EQUILS: dict[str, tuple[str, str]] = {
    "NCSX": ("desc", "NCSX"),
    "W7X": ("desc", "W7-X"),
    "HSX": ("desc", "HSX"),
    "HELIOTRON": ("desc", "HELIOTRON"),
    "precise_QA": ("desc", "precise_QA"),
    "precise_QH": ("desc", "precise_QH"),
    "ATF": ("desc", "ATF"),
    "WISTELL-A": ("desc", "WISTELL-A"),
    "ESTELL": ("desc", "ESTELL"),
    "reactor_QA": ("desc", "reactor_QA"),
    "NCSX_vmec": ("vmec", "tests/data/wout_NCSX.nc"),
    "DSHAPE": ("vmec", "tests/data/wout_DSHAPE.nc"),
}


def load_field(name: str, ntheta: int, nzeta: int, rho: float = 0.5) -> Field:
    """Load a :class:`Field` by short equilibrium name (see ``EQUILS``)."""
    kind, ref = EQUILS[name]
    if kind == "desc":
        import desc  # pyright: ignore[reportMissingImports]

        eq = desc.examples.get(ref)
        if isinstance(eq, (list, tuple)):
            eq = eq[-1]
        return Field.from_desc(eq, rho, ntheta, nzeta)
    if kind == "vmec":
        return Field.from_vmec(ref, rho, ntheta, nzeta)
    if kind == "booz":
        return Field.from_booz_xform(ref, rho, ntheta, nzeta, cutoff=1e-5)
    raise ValueError(f"unknown equilibrium loader {kind!r} for {name!r}")


def make_species(ns: int, density: float) -> list[LocalMaxwellian]:
    """1 species -> [Hydrogen]; 2 species -> [Electron, Hydrogen], ion last.

    Uses fixed normalized gradient scale lengths, so no field geometry is needed.
    """
    kinds = [Hydrogen] if ns == 1 else [Electron, Hydrogen]
    return [
        LocalMaxwellian.from_scale_lengths(kind, TEMP_EV, density, ALT, ALN)
        for kind in kinds
    ]


def _nustar_at(field: Field, ns: int, density: float) -> float:
    sp = make_species(ns, density)
    return float(_nustar_of(sp[-1], field, 1.0, *sp[:-1], lnlambda=LNLAMBDA))


def density_for_nustar(field: Field, ns: int, target: float) -> float:
    """Root-find the density giving exactly ``target`` ion nu* (fixed T, lnlambda)."""

    def resid(logn: float) -> float:
        return np.log10(_nustar_at(field, ns, 10.0**logn)) - np.log10(target)

    logn = brentq(resid, 6.0, 30.0)
    return 10.0 ** float(logn)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Case:
    """One benchmark problem. ``res`` is (nx, na, nt, nz)."""

    name: str
    field: str
    species: int
    nustar: float
    res: tuple[int, int, int, int]
    estar: float  # target ion E* = E_r /(v <B>) at x=1 (field-dependent Erho)
    rtol: float = 1e-5
    coulomb_log: float = 17.0
    tier: str = "nightly"

    def build(self):
        """Return (field, pitchgrid, speedgrid, species, Erho) ready for solve_dke."""
        from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

        nx, na, nt, nz = self.res
        field = load_field(self.field, nt, nz)
        species = make_species(
            self.species, density_for_nustar(field, self.species, self.nustar)
        )
        # Erho (Volts) giving the target ion E* at x=1. Estar is linear in Erho,
        # so divide by its unit-Erho coefficient to invert.
        Erho = self.estar / float(Estar(species[-1], field, 1.0, 1.0))
        return field, UniformPitchAngleGrid(na), MaxwellSpeedGrid(nx), species, Erho


CASES: list[Case] = [
    # --- smoke: cheap, all converge, quick regression signal ---
    Case(
        "smoke_ncsx_2sp_1e-1",
        "NCSX",
        2,
        1e-1,
        (6, 25, 15, 21),
        estar=3e-3,
        tier="smoke",
    ),
    Case(
        "smoke_w7x_2sp_1e-1",
        "W7X",
        2,
        1e-1,
        (6, 25, 15, 21),
        estar=1e-3,
        tier="smoke",
    ),
    Case(
        "smoke_ncsx_1sp_1e-1",
        "NCSX",
        1,
        1e-1,
        (6, 25, 15, 21),
        estar=3e-3,
        tier="smoke",
    ),
    # --- NCSX collisionality scan (fixed field/res, vary nu*)
    Case("ncsx_2sp_nu1e-3", "NCSX", 2, 1e-3, (6, 61, 15, 31), estar=3e-3),
    Case("ncsx_2sp_nu1e-2", "NCSX", 2, 1e-2, (6, 61, 15, 31), estar=3e-3),
    Case("ncsx_2sp_nu3e-2", "NCSX", 2, 3e-2, (6, 61, 15, 31), estar=3e-3),
    Case("ncsx_2sp_nu1e-1", "NCSX", 2, 1e-1, (6, 61, 15, 31), estar=3e-3),
    Case("ncsx_2sp_nu3e-1", "NCSX", 2, 3e-1, (6, 61, 15, 31), estar=3e-3),
    Case("ncsx_2sp_nu1e0", "NCSX", 2, 1.0, (6, 61, 15, 31), estar=3e-3),
    Case("ncsx_2sp_nu1e1", "NCSX", 2, 10.0, (6, 61, 15, 31), estar=3e-3),
    # --- NCSX resolution scan (fixed nu*, vary na)
    Case("ncsx_2sp_na25", "NCSX", 2, 1e-1, (6, 25, 15, 21), estar=3e-3),
    Case("ncsx_2sp_na41", "NCSX", 2, 1e-1, (6, 41, 15, 21), estar=3e-3),
    Case("ncsx_2sp_na81", "NCSX", 2, 1e-1, (6, 81, 15, 31), estar=3e-3),
    # --- geometry spread at failure-prone operating points
    Case("estell_2sp_1e-1", "ESTELL", 2, 1e-1, (6, 25, 15, 21), estar=2e-3),
    Case("w7x_2sp_3e-2", "W7X", 2, 3e-2, (6, 41, 15, 21), estar=1e-3),
    Case("hsx_2sp_1e-1", "HSX", 2, 1e-1, (6, 61, 15, 31), estar=2e-3),
    Case("reactor_qa_2sp_1e-2", "reactor_QA", 2, 1e-2, (6, 25, 15, 21), estar=4e-4),
    Case("precise_qa_2sp_1e-1", "precise_QA", 2, 1e-1, (6, 41, 15, 21), estar=2e-3),
    Case("heliotron_2sp_1e-1", "HELIOTRON", 2, 1e-1, (6, 25, 15, 21), estar=7e-3),
    Case("w7x_2sp_nu1e0", "W7X", 2, 1.0, (6, 41, 15, 21), estar=1e-3),
    # --- low collisionality cases
    Case("estell_2sp_1e-3", "ESTELL", 2, 1e-3, (6, 41, 15, 21), estar=2e-3),
    Case("w7x_2sp_1e-3", "W7X", 2, 1e-3, (6, 41, 15, 21), estar=1e-3),
    Case("hsx_2sp_1e-3", "HSX", 2, 1e-3, (6, 41, 15, 21), estar=2e-3),
    Case("precise_qa_2sp_1e-3", "precise_QA", 2, 1e-3, (6, 41, 15, 21), estar=2e-3),
    Case("estell_2sp_1e-4", "ESTELL", 2, 1e-4, (6, 41, 15, 21), estar=2e-3),
    Case("w7x_2sp_1e-4", "W7X", 2, 1e-4, (6, 41, 15, 21), estar=1e-3),
    Case("hsx_2sp_1e-4", "HSX", 2, 1e-4, (6, 41, 15, 21), estar=2e-3),
    Case("precise_qa_2sp_1e-4", "precise_QA", 2, 1e-4, (6, 41, 15, 21), estar=2e-3),
    # --- high collisionality cases
    Case("estell_2sp_1e1", "ESTELL", 2, 1e1, (6, 41, 15, 21), estar=2e-3),
    Case("w7x_2sp_1e1", "W7X", 2, 1e1, (6, 41, 15, 21), estar=1e-3),
    Case("hsx_2sp_1e1", "HSX", 2, 1e1, (6, 41, 15, 21), estar=2e-3),
    Case("precise_qa_2sp_1e1", "precise_QA", 2, 1e1, (6, 41, 15, 21), estar=2e-3),
    # --- high nx
    Case("ncsx_2sp_nx12", "NCSX", 2, 1e-1, (12, 61, 15, 31), estar=3e-3),
    Case("w7x_2sp_nx12", "W7X", 2, 1.0, (12, 61, 15, 31), estar=1e-3),
    Case("hsx_2sp_nx12", "W7X", 2, 1.0, (12, 61, 15, 31), estar=1e-3),
    # --- high Er cases
    Case("estell_2sp_er1e-2", "ESTELL", 2, 1e-2, (6, 41, 15, 21), estar=1e-2),
    Case("w7x_2sp_er1e-2", "W7X", 2, 1e-2, (6, 41, 15, 21), estar=1e-2),
    Case("hsx_2sp_er1e-2", "HSX", 2, 1e-2, (6, 41, 15, 21), estar=1e-2),
    Case("precise_qa_2sp_er1e-2", "precise_QA", 2, 1e-2, (6, 41, 15, 21), estar=1e-2),
    Case("estell_2sp_er1e-1", "ESTELL", 2, 1e-2, (6, 41, 15, 21), estar=1e-1),
    Case("w7x_2sp_er1e-1", "W7X", 2, 1e-2, (6, 41, 15, 21), estar=1e-1),
    Case("hsx_2sp_er1e-1", "HSX", 2, 1e-2, (6, 41, 15, 21), estar=1e-1),
    Case("precise_qa_2sp_er1e-1", "precise_QA", 2, 1e-2, (6, 41, 15, 21), estar=1e-1),
    # --- 1-species controls
    Case("ncsx_1sp_1e-1", "NCSX", 1, 1e-1, (6, 25, 15, 21), estar=3e-3),
    Case("heliotron_1sp_1e-1", "HELIOTRON", 1, 1e-1, (6, 25, 15, 21), estar=7e-3),
    Case("w7x_1sp_1e-3", "W7X", 1, 1e-3, (6, 25, 15, 21), estar=1e-3),
    Case("hsx_1sp_1e0", "HSX", 1, 1e0, (6, 25, 15, 21), estar=2e-3),
]


def cases_for_tier(tier: str) -> list[Case]:
    """``smoke`` -> smoke only; ``nightly``/``all`` -> everything."""
    if tier == "smoke":
        return [c for c in CASES if c.tier == "smoke"]
    return list(CASES)


def cases_by_names(names: list[str]) -> list[Case]:
    """Select cases by exact ``name``, preserving the requested order.

    Raises ``KeyError`` listing any unknown names (with the full catalog) so a
    typo fails loudly instead of silently running nothing.
    """
    by_name = {c.name: c for c in CASES}
    missing = [n for n in names if n not in by_name]
    if missing:
        known = ", ".join(sorted(by_name))
        raise KeyError(f"unknown case(s): {', '.join(missing)}. known cases: {known}")
    return [by_name[n] for n in names]


def all_case_names() -> list[str]:
    """Every case name in catalog order (for ``--list`` / help output)."""
    return [c.name for c in CASES]
