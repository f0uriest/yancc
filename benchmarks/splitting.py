"""Split a benchmark case list into balanced groups that can run on separate machines.

Mirrors the ``--splits`` / ``--group`` interface of pytest-split. Cases differ in
cost by well over an order of magnitude, so a plain round-robin split leaves some groups
far slower than others. Instead cases are assigned greedily (longest first, to the least
loaded group) using per-case wall times recorded in ``durations.json``, and each group
keeps the catalog order of its cases. The split is deterministic, so every machine
computes the same partition from the same files.

Cases missing from ``durations.json`` (e.g. newly added ones) are weighted by the median
of the known cases. Refresh the file from full-suite results with:

    python benchmarks/splitting.py update --dke dke.json --mdke mdke.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Sequence
from typing import TypeVar

DURATIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "durations.json"
)

T = TypeVar("T")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``--splits`` and ``--group`` options to a ``run`` parser."""
    parser.add_argument(
        "--splits",
        type=int,
        metavar="N",
        help="split the selected cases into N groups balanced by recorded durations "
        "(durations.json) and run only the one given by --group",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=1,
        metavar="G",
        help="which of the --splits groups to run, 1-indexed (default: 1). Running "
        "G = 1..N covers every case exactly once.",
    )


def select_from_args(cases: Sequence[T], harness: str, args: argparse.Namespace):
    """Apply the ``--splits`` / ``--group`` options to ``cases``.

    Returns ``(cases, suffix)``, with ``suffix`` a description for the run label
    (empty if not splitting). Raises ``ValueError`` for an invalid option combination.
    """
    if args.splits is None:
        if args.group != 1:
            raise ValueError("--group requires --splits")
        return list(cases), ""
    if args.splits < 1 or not 1 <= args.group <= args.splits:
        raise ValueError(
            f"need --splits >= 1 and 1 <= --group <= --splits, got splits="
            f"{args.splits} group={args.group}"
        )
    selected = select_group(cases, harness, args.splits, args.group)
    return selected, f" group {args.group}/{args.splits}"


def load_durations(harness: str, path: str = DURATIONS_PATH) -> dict[str, float]:
    """Recorded wall times (s) by case name for ``harness`` (``"dke"``/``"mdke"``)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get(harness, {})


def select_group(
    cases: Sequence[T],
    harness: str,
    splits: int,
    group: int,
    durations: dict[str, float] | None = None,
) -> list[T]:
    """Return the cases in ``group`` (1-indexed) of ``splits`` balanced groups.

    Every case lands in exactly one group. ``cases`` need a ``name`` attribute.
    """
    if durations is None:
        durations = load_durations(harness)
    known = [durations[c.name] for c in cases if c.name in durations]  # type: ignore[attr-defined]
    default = statistics.median(known) if known else 1.0
    weight = [durations.get(c.name, default) for c in cases]  # type: ignore[attr-defined]

    # longest first; ties broken by catalog position so the result is deterministic
    order = sorted(range(len(cases)), key=lambda k: (-weight[k], k))
    load = [0.0] * splits
    owner = [0] * len(cases)
    for k in order:
        g = min(range(splits), key=lambda j: (load[j], j))
        owner[k] = g
        load[g] += weight[k]
    return [c for k, c in enumerate(cases) if owner[k] == group - 1]


def _update(args: argparse.Namespace) -> int:
    out: dict[str, dict[str, float]] = {}
    if os.path.exists(DURATIONS_PATH):
        with open(DURATIONS_PATH) as f:
            out = json.load(f)
    # merge, so a partial results file only refreshes the cases it contains
    for harness in ("dke", "mdke"):
        path = getattr(args, harness)
        if path is None:
            continue
        with open(path) as f:
            results = json.load(f)["results"]
        out.setdefault(harness, {}).update(
            {n: r["wall_s"] for n, r in results.items() if r.get("wall_s")}
        )
    with open(DURATIONS_PATH, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"# wrote {DURATIONS_PATH}")
    return 0


def main() -> int:
    """Command line interface."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pu = sub.add_parser(
        "update",
        help="refresh durations.json from results JSONs (merged into existing)",
    )
    pu.add_argument("--dke", help="results JSON from bench_dke.py run")
    pu.add_argument("--mdke", help="results JSON from bench_mdke.py run")
    pu.set_defaults(func=_update)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
