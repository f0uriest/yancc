"""Ahead-of-time compilation with compile and run timed separately.

Used by the benchmark harnesses so that compilation cost, warm runtime and the size of
the compiled program are reported as separate quantities.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax


def _memory_bytes(compiled: Any) -> int | None:
    """XLA's static memory estimate for a compiled executable, in bytes.

    Arguments + outputs + temporaries + generated code, minus buffers aliased between
    them. This is the compiler's buffer assignment, not a measured peak. None if the
    backend does not provide an analysis.
    """
    try:
        stats = compiled.memory_analysis()
    except Exception:  # not all backends implement it
        return None
    if stats is None:
        return None
    return int(
        stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        + stats.temp_size_in_bytes
        + stats.generated_code_size_in_bytes
        - stats.alias_size_in_bytes
    )


def aot_run(fn: Callable, *args: Any) -> tuple[Any, dict[str, float | int | None]]:
    """Compile ``fn(*args)`` ahead of time, then run the compiled executable once.

    ``fn`` is jitted with ``equinox.filter_jit``, so array arguments are traced and
    everything else is static. Returns ``(output, timings)``, where ``timings`` has

    - ``compile_s``: tracing, lowering and XLA compilation (wall time, seconds)
    - ``run_s``: one execution of the compiled executable, until its outputs are ready
    - ``mem_bytes``: XLA's memory estimate for the executable (see ``_memory_bytes``)
    """
    jitted: Any = eqx.filter_jit(fn)  # typed as a plain function, but has .lower
    t0 = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = jax.block_until_ready(compiled(*args))
    run_s = time.perf_counter() - t0
    return out, {
        "compile_s": round(compile_s, 1),
        "run_s": round(run_s, 1),
        "mem_bytes": _memory_bytes(compiled.compiled),
    }


# Columns for the results table: compile time, run time (s) and memory estimate (GiB).
TIMING_HDR = f"{'comp_s':>6} {'run_s':>6} {'mem_GiB':>7}"


def format_timing(r: dict) -> str:
    """Format the compile / run / memory fields of a result as table columns."""
    comp, run, mem = r.get("compile_s"), r.get("run_s"), r.get("mem_bytes")
    return (
        f"{'-' if comp is None else f'{comp:.0f}':>6} "
        f"{'-' if run is None else f'{run:.0f}':>6} "
        f"{'-' if mem is None else f'{mem / 2**30:.2f}':>7}"
    )
