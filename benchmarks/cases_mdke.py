"""Case matrix for the monoenergetic DKE (mDKE) convergence benchmark.

Each :class:`Case` fully specifies a monoenergetic drift-kinetic solve: equilibrium,
monoenergetic collisionality ``nuhat`` (ν/v, in 1/m), DKES-normalized radial electric
field ``erhat`` (Er/v), grid resolution ``(na, nt, nz)`` and the solve tolerance.

Unlike the full DKE, the mDKE has no species or speed grid - it is a single 3D solve
per RHS parametrized directly by ``(nuhat, erhat)``, so a case sweeps the DKES database
axes (collisionality and electric field) across geometry.

``erhat`` is the DKES normalization Er/v; ``solve_mdke`` wants ``erhohat = Erho/v``,
which is ``erhat * field.a_minor`` (this mirrors ``tests/test_solve.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from cases_dke import load_field

from yancc.velocity_grids import UniformPitchAngleGrid


@dataclass(frozen=True)
class Case:
    """One monoenergetic benchmark problem. ``res`` is (na, nt, nz)."""

    name: str
    field: str
    nuhat: float  # monoenergetic collisionality ν/v in 1/m
    erhat: float  # DKES-normalized radial electric field Er/v (>= 0)
    res: tuple[int, int, int]
    rtol: float = 1e-5
    tier: str = "nightly"

    def build(self):
        """Return (field, pitchgrid, erhohat, nuhat) ready for solve_mdke."""
        na, nt, nz = self.res
        field = load_field(self.field, nt, nz)
        pitchgrid = UniformPitchAngleGrid(na)
        # solve_mdke takes erhohat = Erho/v = erhat * a_minor.
        erhohat = self.erhat * float(field.a_minor)
        return field, pitchgrid, erhohat, self.nuhat


CASES: list[Case] = [
    # --- smoke: cheap, all converge, quick regression signal ---
    Case("smoke_w7x_1e-1_er0", "W7X", 1e-1, 0.0, (25, 15, 21), tier="smoke"),
    Case("smoke_ncsx_1e-2_er0", "NCSX", 1e-2, 0.0, (25, 15, 21), tier="smoke"),
    Case("smoke_w7x_1e-3_er1e-3", "W7X", 1e-3, 1e-3, (25, 15, 21), tier="smoke"),
    # --- W7X collisionality scan at Er = 0 (full nuhat range) ---
    Case("w7x_nu1e-5_er0", "W7X", 1e-5, 0.0, (101, 17, 33)),
    Case("w7x_nu1e-4_er0", "W7X", 1e-4, 0.0, (101, 17, 33)),
    Case("w7x_nu1e-3_er0", "W7X", 1e-3, 0.0, (61, 17, 33)),
    Case("w7x_nu1e-2_er0", "W7X", 1e-2, 0.0, (61, 17, 33)),
    Case("w7x_nu1e-1_er0", "W7X", 1e-1, 0.0, (61, 17, 33)),
    Case("w7x_nu1e0_er0", "W7X", 1e0, 0.0, (61, 17, 33)),
    Case("w7x_nu1e1_er0", "W7X", 1e1, 0.0, (61, 17, 33)),
    Case("w7x_nu1e2_er0", "W7X", 1e2, 0.0, (61, 17, 33)),
    # --- W7X electric-field scan at nuhat = 1e-3 (full erhat range) ---
    Case("w7x_nu1e-3_er1e-4", "W7X", 1e-3, 1e-4, (61, 17, 33)),
    Case("w7x_nu1e-3_er1e-3", "W7X", 1e-3, 1e-3, (61, 17, 33)),
    Case("w7x_nu1e-3_er1e-2", "W7X", 1e-3, 1e-2, (61, 17, 33)),
    Case("w7x_nu1e-3_er1e-1", "W7X", 1e-3, 1e-1, (61, 17, 33)),
    # --- low-collisionality resonant corner (finite Er, high pitch res) ---
    Case("w7x_nu1e-5_er1e-3", "W7X", 1e-5, 1e-3, (101, 17, 33)),
    Case("w7x_nu1e-5_er1e-2", "W7X", 1e-5, 1e-2, (101, 17, 33)),
    Case("hsx_nu1e-4_er1e-3", "HSX", 1e-4, 1e-3, (101, 17, 33)),
    Case("estell_nu1e-4_er1e-3", "ESTELL", 1e-4, 1e-3, (101, 17, 33)),
    # --- geometry spread at nuhat = 1e-3, Er = 0 ---
    Case("ncsx_nu1e-3_er0", "NCSX", 1e-3, 0.0, (61, 17, 33)),
    Case("hsx_nu1e-3_er0", "HSX", 1e-3, 0.0, (61, 17, 33)),
    Case("heliotron_nu1e-3_er0", "HELIOTRON", 1e-3, 0.0, (61, 17, 33)),
    Case("estell_nu1e-3_er0", "ESTELL", 1e-3, 0.0, (61, 17, 33)),
    Case("precise_qa_nu1e-3_er0", "precise_QA", 1e-3, 0.0, (61, 17, 33)),
    Case("precise_qh_nu1e-3_er0", "precise_QH", 1e-3, 0.0, (61, 17, 33)),
    # --- geometry spread at nuhat = 1e-1 with finite Er ---
    Case("ncsx_nu1e-1_er1e-2", "NCSX", 1e-1, 1e-2, (61, 17, 33)),
    Case("hsx_nu1e-1_er1e-2", "HSX", 1e-1, 1e-2, (61, 17, 33)),
    Case("w7x_nu1e-1_er1e-2", "W7X", 1e-1, 1e-2, (61, 17, 33)),
    # --- high-collisionality geometry spread (diffusion dominated) ---
    Case("ncsx_nu1e1_er0", "NCSX", 1e1, 0.0, (61, 17, 33)),
    Case("hsx_nu1e1_er0", "HSX", 1e1, 0.0, (61, 17, 33)),
    # --- axisymmetric (tokamak) control: nzeta = 1 ---
    Case("dshape_nu1e-2_er0", "DSHAPE", 1e-2, 0.0, (61, 17, 1)),
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
