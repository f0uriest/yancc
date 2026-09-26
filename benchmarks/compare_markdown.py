"""Render a baseline-vs-current benchmark comparison as GitHub-flavored markdown.

Works on the results JSON written by ``bench_dke.py run`` / ``bench_mdke.py run``:

    python benchmarks/compare_markdown.py --base base.json --current current.json \
        --title "DKE"

Either side can be given several files (e.g. one per group of a split run); their
results are merged. Files that do not exist are skipped.

Every case gets a success, matvec count (``nmv``) and wall time comparison. Improvements
are marked 🟢 and regressions 🔴. Rows with any change are listed first; unchanged rows
are collapsed. Only ``success`` and ``nmv`` count as regressions, using the same rules
as the ``compare`` subcommand of the benchmark scripts. Wall time is machine and load
dependent, so it is marked but reported as informational.

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
# Wall time is only marked when it moves by both a relative and an absolute margin, so
# neither noise on long solves nor small absolute changes on short ones get flagged.
WALL_REL_TOL = 0.15
WALL_ABS_TOL = 2.0


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


def _wall_cell(b: dict | None, c: dict) -> tuple[str, int]:
    """Wall time comparison (informational)."""
    cw = c.get("wall_s")
    if b is None or cw is None or b.get("wall_s") is None:
        return "" if cw is None else f"{cw:g} s", 0
    bw = b["wall_s"]
    text = f"{bw:g} → {cw:g} s ({_pct(cw, bw)})"
    if abs(cw - bw) < WALL_ABS_TOL:
        return text, 0
    if cw > bw * (1 + WALL_REL_TOL):
        return f"{text} {BAD}", -1
    if cw < bw * (1 - WALL_REL_TOL):
        return f"{text} {GOOD}", 1
    return text, 0


def _row(
    name: str, b: dict | None, c: dict | None, is_new: bool
) -> tuple[str, int, int]:
    """Return (markdown row, gate mark, wall mark) for one case."""
    if c is None:
        return f"| {name} | (dropped) | | |", 0, 0
    s, ms = _status_cell(b, c)
    n, mn = _nmv_cell(b, c)
    w, mw = _wall_cell(b, c)
    tag = " (new)" if is_new else ""
    return f"| {name}{tag} | {s} | {n} | {w} |", min(ms, mn) or max(ms, mn), mw


def render(base: dict | None, cur: dict, title: str) -> str:
    """Return the markdown section comparing ``cur`` to ``base``."""
    bres = {} if base is None else base["results"]
    names = list(cur["results"]) + [n for n in bres if n not in cur["results"]]
    changed, same = [], []
    n_bad = n_good = w_bad = w_good = 0
    for name in names:
        b, c = bres.get(name), cur["results"].get(name)
        is_new = base is not None and b is None
        row, gate, wall = _row(name, b, c, is_new)
        n_bad += gate < 0
        n_good += gate > 0
        w_bad += wall < 0
        w_good += wall > 0
        notable = gate or wall or is_new or c is None or base is None
        (changed if notable else same).append((gate, wall, row))

    def rank(t):
        # gate regressions, gate improvements, slower, faster, then new cases
        gate, wall, _ = t
        return 0 if gate < 0 else 1 if gate > 0 else 2 if wall < 0 else 3 if wall else 4

    changed.sort(key=rank)

    hdr = f"#### {title}\n\n"
    if base is None:
        hdr += "_No baseline results were produced, showing the PR results only._\n\n"
    else:
        totb = sum(r.get("wall_s") or 0 for r in bres.values())
        totc = sum(r.get("wall_s") or 0 for r in cur["results"].values())
        hdr += (
            f"{BAD} {n_bad} regression(s), {GOOD} {n_good} improvement(s) in "
            f"success / matvec count. Wall time (informational): {w_bad} slower, "
            f"{w_good} faster; total {totb:.0f} → {totc:.0f} s "
            f"({_pct(totc, totb)}).\n\n"
        )
    table_hdr = (
        "| case | success | nmv (base → PR) | wall (base → PR) |\n|---|---|---|---|\n"
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
