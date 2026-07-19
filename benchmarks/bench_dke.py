#!/usr/bin/env python
"""DKE convergence benchmark: run the case matrix, record deterministic solve
metrics, and diff two runs to catch regressions.

This is **not** a pytest test (it takes ~hours or needs a GPU); it is a
standalone regression / profiling tool. It records only hardware-independent,
deterministic quantities - matvec count (``nmv``), restart count (``niter``),
convergence flag (``success``) and final residual (``res``). Per-iteration timing
is deliberately out of scope (measure that with a microbenchmark of matvec cost).

The solver is always run with whatever ``solve_dke`` uses by default, so a change
in ``nmv``/``success`` is a change in the shipped code. A/B testing is done by
 the two result files.

Usage
-----
    # Run a tier and write results (env header records git sha / device / jax).
    python benchmarks/bench_dke.py run --tier smoke   --out smoke.json
    python benchmarks/bench_dke.py run --tier nightly --out nightly.json

    # Run specific case(s) by name (overrides --tier); --list shows the names.
    python benchmarks/bench_dke.py run --list
    python benchmarks/bench_dke.py run --case ncsx_2sp_nu1e-2 --out one.json
    python benchmarks/bench_dke.py run --case hsx_2sp_1e-1,w7x_2sp_3e-2 --out two.json

    # Diff current (B) against a baseline (A); exits nonzero on any regression.
    python benchmarks/bench_dke.py compare baseline.json nightly.json

Typical regression check across a code change::

    git checkout main    && python benchmarks/bench_dke.py run --out base.json
    git checkout feature && python benchmarks/bench_dke.py run --out feat.json
    python benchmarks/bench_dke.py compare base.json feat.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import jax

jax.config.update("jax_enable_x64", True)

from cases_dke import (  # noqa: E402  (sibling module)
    CASES,
    Case,
    all_case_names,
    cases_by_names,
    cases_for_tier,
)

import yancc  # noqa: E402
from yancc.solve import solve_dke  # noqa: E402

# nmv is deterministic given the code, but allow a hair of slack so a trivial
# reordering that shifts the count by a few iterations doesn't trip the gate.
NMV_REL_TOL = 0.05

# Descriptive columns (field / species / nu* / E* / grid) shared by run and
# compare so the case being solved is legible without cross-referencing cases_dke.py.
_CASE_HDR = f"{'field':>12} {'sp':>2} {'nu*':>10} {'E*':>10} {'grid':>13}"


def _case_cols(case: Case) -> str:
    """Format a case's identifying parameters as fixed-width columns."""
    nx, na, nt, nz = case.res
    grid = f"{nx}x{na}x{nt}x{nz}"
    return (
        f"{case.field:>12} {case.species:>2} "
        f"{case.nustar:>10.1e} {case.estar:>10.2e} {grid:>13}"
    )


def _env_header() -> dict:
    dev = jax.devices()[0]
    return {
        "yancc_version": yancc.__version__,
        "jax_version": jax.__version__,
        "device": f"{dev.platform}:{dev.device_kind}",
        "x64": jax.config.read("jax_enable_x64"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_case(case: Case, verbose: int = 0) -> dict:
    """Solve one case with production defaults; return its deterministic metrics."""
    field, pg, sg, sp, Erho = case.build()
    _sol, info = solve_dke(
        field,
        pg,
        sg,
        sp,
        Erho,
        rtol=case.rtol,
        coulomb_log=case.coulomb_log,
        verbose=verbose,
        # this is the only departure from standard settings. We want things
        # to converge within 1 restart, so 3 is already more than enough. The production
        # default is 10 for robustness but  if things regress we want to fail fast.
        maxiter=3,
    )
    _sol = jax.block_until_ready(_sol)
    return {
        "nmv": int(info["nmv"]),
        "niter": int(info["niter"]),
        "success": bool(info["success"]),
        "res": float(info["res"]),
    }


def _split_names(values: list[str]) -> list[str]:
    """Flatten repeated ``--case`` flags and comma-separated lists into names."""
    names: list[str] = []
    for v in values:
        names.extend(n.strip() for n in v.split(",") if n.strip())
    return names


def cmd_run(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    if args.list:
        for name in all_case_names():
            print(name)
        return 0
    if args.case:
        names = _split_names(args.case)
        try:
            cases = cases_by_names(names)
        except KeyError as e:
            print(f"error: {e.args[0]}", file=sys.stderr)
            return 2
        label = f"cases[{','.join(names)}]"
    else:
        cases = cases_for_tier(args.tier)
        label = args.tier
    header = _env_header()
    print(
        f"# bench_dke {label}  yancc={header['yancc_version']}  {header['device']}",
        flush=True,
    )
    print(
        f"  {'case':>22} {_CASE_HDR} {'nmv':>6} {'nit':>4} {'ok':>3} "
        f"{'res':>10}  {'sec':>5}",
        flush=True,
    )
    results: dict[str, dict] = {}
    for case in cases:
        t0 = time.perf_counter()
        try:
            r = run_case(case, verbose=2 if args.verbose else 0)
            err = None
        except Exception as e:  # a build/solve blowup is a data point, not a crash
            r = {"nmv": None, "niter": None, "success": False, "res": None}
            err = f"{type(e).__name__}: {e}"
        r["error"] = err
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        results[case.name] = r
        okstr = "ok" if r["success"] else "X"
        nmv = "ERR" if r["nmv"] is None else r["nmv"]
        res = "-" if r["res"] is None else f"{r['res']:.2e}"
        print(
            f"  {case.name:>22} {_case_cols(case)} {str(nmv):>6} "
            f"{str(r['niter']):>4} {okstr:>3} {res:>10}  {r['wall_s']:>5.0f}",
            flush=True,
        )
        if err:
            print(f"      -> {err}", flush=True)

    if args.out:
        out = {"header": header, "tier": label, "results": results}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"# wrote {args.out}", flush=True)
    return 0


def _ordered_names(names: set[str]) -> list[str]:
    """Order result names by their declaration order in ``cases.CASES``.

    Any names not in the current catalog (e.g. cases dropped since a baseline was
    recorded) are appended alphabetically after the known ones.
    """
    catalog = all_case_names()
    rank = {name: i for i, name in enumerate(catalog)}
    known = [n for n in catalog if n in names]
    unknown = sorted(n for n in names if n not in rank)
    return known + unknown


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare benchmark results."""
    with open(args.baseline) as f:
        base = json.load(f)
    with open(args.current) as f:
        cur = json.load(f)
    bh, ch = base["header"], cur["header"]
    print(
        f"# baseline {bh['yancc_version']} ({bh['device']})  vs  current "
        f"{ch['yancc_version']} ({ch['device']})"
    )
    print(f"  {'case':>22} {_CASE_HDR} {'base':>12} {'cur':>12}  verdict")

    # Case params aren't stored in the results JSON; recover them from the current
    # catalog by name (blank for cases no longer present, e.g. dropped baselines).
    by_name = {c.name: c for c in CASES}
    blank = " " * len(_CASE_HDR)

    regressions = 0
    names = _ordered_names(set(base["results"]) | set(cur["results"]))
    for name in names:
        case = by_name.get(name)
        info = _case_cols(case) if case is not None else blank
        b = base["results"].get(name)
        c = cur["results"].get(name)
        if b is None:
            print(f"  {name:>22} {info} {'--':>12} {'(new case)':>12}  info")
            continue
        if c is None:
            print(f"  {name:>22} {info} {'(dropped)':>12} {'--':>12}  info")
            continue
        bstr = _fmt(b)
        cstr = _fmt(c)
        verdict, is_reg = _verdict(b, c)
        if is_reg:
            regressions += 1
        print(f"  {name:>22} {info} {bstr:>12} {cstr:>12}  {verdict}")

    if regressions:
        print(f"\n# {regressions} REGRESSION(S)")
        return 1
    print("\n# no regressions")
    return 0


def _fmt(r: dict) -> str:
    if not r["success"]:
        return "FAIL"
    return f"{r['nmv']}mv/{r['niter']}r"


def _verdict(b: dict, c: dict) -> tuple[str, bool]:
    """Compare current (c) to baseline (b). Regression = strictly worse."""
    # Convergence flip dominates everything.
    if b["success"] and not c["success"]:
        return "REGRESSED (converged -> FAIL)", True
    if not b["success"] and c["success"]:
        return "fixed (FAIL -> converged)", False
    if not b["success"] and not c["success"]:
        return "still failing", False
    # Both converged: compare matvec count.
    dn = c["nmv"] - b["nmv"]
    if c["nmv"] > b["nmv"] * (1 + NMV_REL_TOL):
        return f"REGRESSED (+{dn} mv, +{100 * dn / b['nmv']:.0f}%)", True
    if c["nmv"] < b["nmv"] * (1 - NMV_REL_TOL):
        return f"improved ({dn} mv, {100 * dn / b['nmv']:.0f}%)", False
    return "same", False


def main() -> int:
    """Command line interface."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser(
        "run", help="run a tier (or named cases) and write a results JSON"
    )
    pr.add_argument("--tier", default="nightly", choices=["smoke", "nightly", "all"])
    pr.add_argument(
        "--case",
        action="append",
        metavar="NAME",
        help="run only these named case(s); repeatable and/or comma-separated. "
        "Overrides --tier. Use --list to see available names.",
    )
    pr.add_argument(
        "--list",
        action="store_true",
        help="print every available case name and exit (no solve)",
    )
    pr.add_argument(
        "--out", help="results JSON path; if omitted, only print to the terminal"
    )
    pr.add_argument(
        "--verbose",
        action="store_true",
        help="pass verbose=2 to solve_dke to monitor krylov progress per case",
    )
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser(
        "compare", help="diff current against baseline; nonzero exit on regression"
    )
    pc.add_argument("baseline")
    pc.add_argument("current")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
