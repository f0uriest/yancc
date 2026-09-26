"""Case matrix for the DKE convergence benchmark.

Each :class:`Case` fully specifies a drift-kinetic benchmark solve: equilibrium,
species, target collisionality ``nustar`` and normalized electric field ``estar`` (E* at
x=1) of the reference species (by default the last one in the list), grid resolution,
and the per-problem tolerances.

``species`` is either a count (1 -> [Hydrogen]; 2 -> [Electron, Hydrogen]) or a tuple of
kinds, so mass, charge and composition can be varied independently:

    species=("e", "D")            electrons + deuterium
    species=("H", (12.0, 6.0))    hydrogen + fully stripped carbon (arbitrary
                                  (mass, charge) in proton / elementary units)

Optional per-case settings:

- ``tratio``: temperature of the last species relative to the first.
- ``nkinetic``: only the first ``nkinetic`` species are solved for; the rest enter only
  through the collision operator as static backgrounds (``solve_dke(background=...)``).
- ``reference``: index of the species whose collisionality, E* and density set the
  plasma (default: the last one).
- ``density_ratios``: species densities relative to the reference species, replacing
  the default quasineutral ratios (e.g. to make one species dilute). Required for more
  than two species.

Fields named here that are DESC examples (NCSX, W7X, ...) require ``desc`` to be
importable; the two bundled VMEC cases (NCSX_vmec, DSHAPE) do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from yancc.field import Field
from yancc.species import (
    Boron,
    Deuterium,
    Electron,
    Helium,
    Hydrogen,
    Lithium,
    LocalMaxwellian,
    Nitrogen,
    Species,
    Tritium,
)
from yancc.species import _Estar as Estar
from yancc.species import _nustar as _nustar_of

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


# short names accepted in ``Case.species``; anything else is given as a
# (mass, charge) pair in proton-mass / elementary-charge units.
SPECIES_KINDS: dict[str, Species] = {
    "e": Electron,
    "H": Hydrogen,
    "D": Deuterium,
    "T": Tritium,
    "He": Helium,
    "Li": Lithium,
    "B": Boron,
    "N": Nitrogen,
}


def resolve_kinds(species: int | tuple) -> list[Species]:
    """``species`` -> list of :class:`Species`, reference species last.

    Accepts a count (1 -> [Hydrogen]; 2 -> [Electron, Hydrogen]) or a tuple of short
    names (see ``SPECIES_KINDS``) and/or (mass, charge) pairs.
    """
    if isinstance(species, int):
        return [Hydrogen] if species == 1 else [Electron, Hydrogen]
    kinds = []
    for k in species:
        if isinstance(k, str):
            if k not in SPECIES_KINDS:
                raise KeyError(
                    f"unknown species {k!r}; known: {', '.join(sorted(SPECIES_KINDS))} "
                    "(or give a (mass, charge) pair)"
                )
            kinds.append(SPECIES_KINDS[k])
        else:
            mass, charge = k
            kinds.append(Species(mass, charge))
    return kinds


def make_species(
    species: int | tuple,
    density: float,
    tratio: float = 1.0,
    density_ratios: tuple[float, ...] | None = None,
    reference: int = -1,
) -> list[LocalMaxwellian]:
    """Build the Maxwellians in the order given.

    ``density`` is the density of the ``reference`` species. The others follow
    ``density_ratios`` (n_s / n_reference, one per species). The default is
    quasineutrality, n_s = n_ref |q_ref| / |q_s|, which only makes sense for up to two
    species, so more require explicit ratios. ``tratio`` scales the last species'
    temperature relative to the first, with ``TEMP_EV`` the reference (absolute
    temperature is redundant with density for setting collisionality).

    Uses fixed normalized gradient scale lengths, so no field geometry is needed.
    """
    kinds = resolve_kinds(species)
    if density_ratios is None:
        if len(kinds) > 2:
            raise ValueError("more than two species need explicit density_ratios")
        q_ref = abs(float(kinds[reference].charge))
        density_ratios = tuple(q_ref / abs(float(k.charge)) for k in kinds)
    elif len(density_ratios) != len(kinds):
        raise ValueError(f"need {len(kinds)} density_ratios, got {len(density_ratios)}")
    temps = [TEMP_EV] * len(kinds)
    temps[-1] = TEMP_EV * tratio
    return [
        LocalMaxwellian.from_scale_lengths(kind, t, density * r, ALT, ALN)
        for kind, t, r in zip(kinds, temps, density_ratios)
    ]


def _nustar_at(
    field: Field,
    species: int | tuple,
    density: float,
    tratio: float = 1.0,
    density_ratios: tuple[float, ...] | None = None,
    reference: int = -1,
) -> float:
    sp = make_species(species, density, tratio, density_ratios, reference)
    ref = sp.pop(reference)
    return float(_nustar_of(ref, field, 1.0, *sp, lnlambda=LNLAMBDA))


def density_for_nustar(
    field: Field,
    species: int | tuple,
    target: float,
    tratio: float = 1.0,
    density_ratios: tuple[float, ...] | None = None,
    reference: int = -1,
) -> float:
    """Root-find the reference density giving ``target`` nu* (fixed T, lnlambda)."""

    def resid(logn: float) -> float:
        nu = _nustar_at(field, species, 10.0**logn, tratio, density_ratios, reference)
        return np.log10(nu) - np.log10(target)

    logn = brentq(resid, 6.0, 30.0)
    return 10.0 ** float(logn)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Case:
    """One benchmark problem. ``res`` is (nx, na, nt, nz)."""

    name: str
    field: str
    species: int | tuple  # count, or explicit kinds e.g. ("e", "He")
    nustar: float
    res: tuple[int, int, int, int]
    estar: float  # target E* = E_r /(v <B>) of the reference species at x=1
    rtol: float = 1e-5
    coulomb_log: float = 17.0
    tier: str = "nightly"
    tratio: float = 1.0  # T of the last species relative to the first (T_last/T_first)
    # Number of LEADING species solved kinetically; the rest are static collisional
    # backgrounds. ``None`` = all kinetic. The plasma (densities, nu*, Erho) is the same
    # either way.
    nkinetic: int | None = None
    # Species densities relative to the reference species. ``None`` gives quasineutral
    # ratios (two species at most). Quasineutrality is not enforced when given, so a
    # minority species can be made arbitrarily dilute.
    density_ratios: tuple[float, ...] | None = None
    # Index of the species whose nu*, E* and density set the plasma. Species order
    # matters for ``nkinetic`` (background species must be last), so this lets a main
    # ion be the reference while an impurity trails as a background.
    reference: int = -1

    def build(self):
        """Return (field, pitchgrid, speedgrid, species, Erho, background)."""
        from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

        nx, na, nt, nz = self.res
        field = load_field(self.field, nt, nz)
        dens = density_for_nustar(
            field,
            self.species,
            self.nustar,
            self.tratio,
            self.density_ratios,
            self.reference,
        )
        species = make_species(
            self.species, dens, self.tratio, self.density_ratios, self.reference
        )
        # Erho (Volts) giving the target E* of the reference species at x=1. Estar is
        # linear in Erho, so divide by its unit-Erho coefficient to invert. Built from
        # the full species list so a background variant shares its parent's field.
        Erho = self.estar / float(Estar(species[self.reference], field, 1.0, 1.0))
        nk = len(species) if self.nkinetic is None else self.nkinetic
        return (
            field,
            UniformPitchAngleGrid(na),
            MaxwellSpeedGrid(nx),
            species[:nk],
            Erho,
            species[nk:],
        )


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

# --- temperature ratio: ion/electron temperature differs from unity
CASES += [
    Case("ncsx_2sp_tratio3", "NCSX", 2, 1e-1, (6, 41, 15, 21), estar=3e-3, tratio=3.0),
    Case(
        "ncsx_2sp_tratio0.3", "NCSX", 2, 1e-1, (6, 41, 15, 21), estar=3e-3, tratio=0.3
    ),
]

# --- kinetic electrons on a static ion background (nkinetic=1). Only the electron block
# is solved, so the tolerance applies to it directly, but the ion still shapes the
# electron collision operator. These sit at grid/collisionality combinations where the
# speed-grid nodes make the operator's low-speed structure hard for the preconditioner.
CASES += [
    Case("ncsx_2sp_nu3e-2_ebg", "NCSX", 2, 3e-2, (6, 61, 15, 31), 3e-3, nkinetic=1),
    Case("ncsx_nx7_nu5e-3_ebg", "NCSX", 2, 5e-3, (7, 61, 15, 31), 3e-3, nkinetic=1),
    Case("hsx_nx8_nu2e-3_ebg", "HSX", 2, 2e-3, (8, 61, 15, 31), 2e-3, nkinetic=1),
]

# --- full 2-species solves at speed-grid / collisionality combinations where the
# operator is hardest for the preconditioner (grid-dependent, not monotonic in nu*)
CASES += [
    Case("probe_ncsx_nx7_nu5e-3", "NCSX", 2, 5e-3, (7, 61, 15, 31), estar=3e-3),
    Case("probe_ncsx_nx10_nu1e-3", "NCSX", 2, 1e-3, (10, 61, 15, 31), estar=3e-3),
    Case("probe_w7x_nx7_nu5e-3", "W7X", 2, 5e-3, (7, 41, 15, 21), estar=1e-3),
    Case("probe_hsx_nx8_nu2e-3", "HSX", 2, 2e-3, (8, 61, 15, 31), estar=2e-3),
]

# --- extreme mass / temperature ratios. Light species (m=1, kinetic) on a static heavy
# background (m_heavy, T = tratio * T_light), both Z=1. The mass ratio enters the
# collision operator both explicitly and through the thermal-speed ratio
# sqrt(m_heavy / tratio), which temperature moves independently of mass. The electron
# mass ratio at equal T, and a hot light species matching its speed ratio, are the
# extremes; the others are intermediate points on the same grid.
CASES += [
    Case(
        f"massT_{tag}",
        "NCSX",
        ((1.0, 1.0), (mh, 1.0)),
        1e-1,
        (6, 61, 15, 31),
        estar=3e-3,
        tratio=tr,
        nkinetic=1,
    )
    for tag, mh, tr in [
        ("m100_t01", 100.0, 0.1),
        ("m1000_t10", 1000.0, 10.0),
        ("m1836_t1", 1836.0, 1.0),
        ("m1000_t054", 1000.0, 0.54),
    ]
]


# --- impurities. Kinetic electrons + hydrogen (the reference species, so nu* and E* are
# the main ion's) plus an impurity of given (mass, charge) and number fraction
# n_imp / n_H; electron density follows from quasineutrality, n_e = n_H (1 + Z f).
# "kin" solves the impurity kinetically as well, "bg" keeps it as a static collisional
# background of the electron and hydrogen equations (its charge enters as Z^2).
def _impurity_case(name, field, mc, frac, res, estar, cfg) -> Case:
    """(e, H, impurity) case; nu* and E* refer to hydrogen (index 1)."""
    return Case(
        name,
        field,
        species=("e", "H", mc),
        nustar=1e-1,
        res=res,
        estar=estar,
        nkinetic=(2 if cfg == "bg" else None),
        reference=1,
        density_ratios=(1.0 + mc[1] * frac, 1.0, frac),
    )


# impurity carrying as much ion charge as hydrogen (Z f = 1), so its effect is strong
_IMPURITIES = [
    ("he", (4.0, 2.0)),  # light, low charge
    ("c6", (12.0, 6.0)),  # canonical impurity
    ("w", (184.0, 40.0)),  # extreme mass and charge
]
CASES += [
    _impurity_case(
        f"imp_ncsx_{tag}_{cfg}", "NCSX", mc, 1 / mc[1], (6, 61, 15, 31), 3e-3, cfg
    )
    for tag, mc in _IMPURITIES
    for cfg in ("kin", "bg")
]
CASES += [
    _impurity_case(
        f"imp_hsx_c6_{cfg}", "HSX", (12.0, 6.0), 1 / 6, (6, 41, 15, 21), 2e-3, cfg
    )
    for cfg in ("kin", "bg")
]
# dilute impurity: 1% carbon by number
CASES += [
    _impurity_case(
        f"imp_ncsx_c6_dilute_{cfg}",
        "NCSX",
        (12.0, 6.0),
        0.01,
        (6, 61, 15, 31),
        3e-3,
        cfg,
    )
    for cfg in ("kin", "bg")
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
