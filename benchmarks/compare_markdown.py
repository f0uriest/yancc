"""Render a baseline-vs-current benchmark comparison as GitHub-flavored markdown.

Works on the results JSON written by ``bench_dke.py run`` / ``bench_mdke.py run``:

    python benchmarks/compare_markdown.py --base base.json --current current.json \
        --title "DKE"

Either side can be given several files (e.g. one per group of a split run); their
results are merged. Files that do not exist are skipped.

Every case gets a success, matvec count (``nmv``), compile time, run time and compiled
memory estimate comparison. Improvements are marked 🟢 and regressions 🔴. Rows with
any change are listed first; unchanged rows are collapsed. Only ``success`` and ``nmv``
count as regressions, using the same rules as the ``compare`` subcommand of the
benchmark scripts. Times are machine and load dependent, so they are marked but
reported as informational, as is the memory estimate. Results files from before
compile time, run time and memory were recorded simply show the current values.

The baseline may be missing (e.g. the baseline run failed), in which case only the
current results are shown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

GOOD = "🟢"
BAD = "🔴"

# Keep in sync with NMV_REL_TOL in bench_dke.py / bench_mdke.py.
NMV_REL_TOL = 0.05
# Times are only marked when they move by both a relative and an absolute margin, so
# neither noise on long runs nor small absolute changes on short ones get flagged.
TIME_REL_TOL = 0.15
TIME_ABS_TOL = 2.0  # seconds
# same idea for the compiled memory estimate (GiB)
MEM_REL_TOL = 0.10
MEM_ABS_TOL = 0.0625


def _load(paths: list[str]) -> dict | None:
    """Merge the results in every existing file of ``paths``; None if there are none."""
    merged: dict | None = None
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if merged is None:
            merged = data
        else:
            merged["results"].update(data["results"])
    return merged


def _pct(new: float, old: float) -> str:
    return f"{100 * (new - old) / old:+.0f}%" if old else "n/a"


def _ok(r: dict) -> str:
    return "ok" if r["success"] else "FAIL"


def _status_cell(b: dict | None, c: dict) -> tuple[str, int]:
    """Success comparison; returns (text, mark) with mark +1 good, -1 bad, 0 neutral."""
    if b is None:
        return _ok(c), 0
    if b["success"] and not c["success"]:
        return f"ok → FAIL {BAD}", -1
    if not b["success"] and c["success"]:
        return f"FAIL → ok {GOOD}", 1
    return _ok(c), 0


def _nmv_cell(b: dict | None, c: dict) -> tuple[str, int]:
    """Matvec count comparison, only meaningful when both runs converged."""
    cn = "ERR" if c["nmv"] is None else str(c["nmv"])
    if b is None:
        return cn, 0
    bn = "ERR" if b["nmv"] is None else str(b["nmv"])
    text = f"{bn} → {cn}"
    if not (b["success"] and c["success"]) or b["nmv"] is None or c["nmv"] is None:
        return text, 0
    text += f" ({_pct(c['nmv'], b['nmv'])})"
    if c["nmv"] > b["nmv"] * (1 + NMV_REL_TOL):
        return f"{text} {BAD}", -1
    if c["nmv"] < b["nmv"] * (1 - NMV_REL_TOL):
        return f"{text} {GOOD}", 1
    return text, 0


def _metric_cell(
    b: dict | None,
    c: dict,
    key: str,
    scale: float,
    unit: str,
    fmt: str,
    rel_tol: float,
    abs_tol: float,
) -> tuple[str, int]:
    """Compare ``key`` (raw units, shown divided by ``scale``); lower is better.

    Returns (text, mark) with mark +1 good, -1 bad, 0 neutral. Only marked when the
    change exceeds both the relative and the absolute (in displayed units) tolerance.
    """
    if c.get(key) is None:
        return "", 0
    cv = c[key] / scale
    bv = None if b is None or b.get(key) is None else b[key] / scale
    if bv is None:
        return f"{cv:{fmt}} {unit}", 0
    text = f"{bv:{fmt}} → {cv:{fmt}} {unit} ({_pct(cv, bv)})"
    if abs(cv - bv) < abs_tol:
        return text, 0
    if cv > bv * (1 + rel_tol):
        return f"{text} {BAD}", -1
    if cv < bv * (1 - rel_tol):
        return f"{text} {GOOD}", 1
    return text, 0


def _time_cell(b: dict | None, c: dict, key: str) -> tuple[str, int]:
    return _metric_cell(b, c, key, 1, "s", "g", TIME_REL_TOL, TIME_ABS_TOL)


def _mem_cell(b: dict | None, c: dict) -> tuple[str, int]:
    return _metric_cell(
        b, c, "mem_bytes", 2**30, "GiB", ".2f", MEM_REL_TOL, MEM_ABS_TOL
    )


def _row(
    name: str, b: dict | None, c: dict | None, is_new: bool
) -> tuple[str, int, int, dict[str, int]]:
    """Return (markdown row, gate mark, other mark, per-metric marks) for one case."""
    if c is None:
        return f"| {name} | (dropped) | | | | |", 0, 0, {}
    s, ms = _status_cell(b, c)
    n, mn = _nmv_cell(b, c)
    cells = {
        "compile": _time_cell(b, c, "compile_s"),
        "run": _time_cell(b, c, "run_s"),
        "memory": _mem_cell(b, c),
    }
    marks = {k: m for k, (_, m) in cells.items()}
    other = min(marks.values()) or max(marks.values())
    tag = " (new)" if is_new else ""
    row = (
        f"| {name}{tag} | {s} | {n} | "
        + " | ".join(t for t, _ in cells.values())
        + " |"
    )
    return row, min(ms, mn) or max(ms, mn), other, marks


def render(base: dict | None, cur: dict, title: str) -> str:
    """Return the markdown section comparing ``cur`` to ``base``."""
    bres = {} if base is None else base["results"]
    names = list(cur["results"]) + [n for n in bres if n not in cur["results"]]
    changed, same = [], []
    n_bad = n_good = 0
    counts = {k: [0, 0] for k in ("compile", "run", "memory")}  # [worse, better]
    for name in names:
        b, c = bres.get(name), cur["results"].get(name)
        is_new = base is not None and b is None
        row, gate, other, marks = _row(name, b, c, is_new)
        n_bad += gate < 0
        n_good += gate > 0
        for k, m in marks.items():
            counts[k][0] += m < 0
            counts[k][1] += m > 0
        notable = gate or other or is_new or c is None or base is None
        (changed if notable else same).append((gate, other, row))

    def rank(t):
        # gate regressions, gate improvements, worse, better, then new cases
        gate, other, _ = t
        return (
            0 if gate < 0 else 1 if gate > 0 else 2 if other < 0 else 3 if other else 4
        )

    changed.sort(key=rank)

    hdr = f"#### {title}\n\n"
    if base is None:
        hdr += "_No baseline results were produced, showing the PR results only._\n\n"
    else:

        def total(res: dict, key: str) -> float:
            return sum(r.get(key) or 0 for r in res.values())

        parts = []
        for label, key in (("compile", "compile_s"), ("run", "run_s")):
            tb, tc = total(bres, key), total(cur["results"], key)
            worse, better = counts[label]
            # no baseline total if it predates timing being recorded
            tot = f"{tb:.0f} → {tc:.0f} s ({_pct(tc, tb)})" if tb else f"{tc:.0f} s"
            parts.append(f"{label} time {worse} slower / {better} faster, total {tot}")
        worse, better = counts["memory"]
        parts.append(f"compiled memory {worse} larger / {better} smaller")
        hdr += (
            f"{BAD} {n_bad} regression(s), {GOOD} {n_good} improvement(s) in "
            f"success / matvec count. Informational: {'; '.join(parts)}.\n\n"
        )
    table_hdr = (
        "| case | success | nmv (base → PR) | compile (base → PR) | run (base → PR) "
        "| compiled memory (base → PR) |\n|---|---|---|---|---|---|\n"
    )
    out = hdr
    if changed:
        out += table_hdr + "\n".join(r for *_, r in changed) + "\n\n"
    if same:
        out += (
            f"<details><summary>{len(same)} unchanged case(s)</summary>\n\n"
            + table_hdr
            + "\n".join(r for *_, r in same)
            + "\n\n</details>\n\n"
        )
    return out


def main() -> int:
    """Command line interface."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", nargs="+", default=[], help="baseline results JSON(s)")
    p.add_argument("--current", nargs="+", default=[], help="current results JSON(s)")
    p.add_argument("--title", default="Benchmark results")
    args = p.parse_args()

    cur = _load(args.current)
    if cur is None:
        print(f"#### {args.title}\n\n{BAD} The PR benchmark run produced no results.\n")
        return 0
    sys.stdout.write(render(_load(args.base), cur, args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
