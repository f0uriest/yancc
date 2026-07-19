# DKE convergence benchmark

A standalone regression / profiling harness for the drift-kinetic solver. Not part of
the regular pytest suite, this takes ~1 hour on CPU, but git tracked so runs
are reproducible.

There are two harnesses with identical CLIs: `bench_dke.py` (full multi-species DKE,
cases in `cases_dke.py`) and `bench_mdke.py` (monoenergetic DKE, cases in
`cases_mdke.py`). Everything below describes `bench_dke.py`; `bench_mdke.py` works the
same way (swap the script name)

It records: matvec count (`nmv`), restart count (`niter`), convergence flag (`success`),
 final residual (`res`). This is meant to catch regressions in overall deterministic
performance (ie due to multigrid and krylov settings). It is not meant to compare wall
time, this is easier to measure separately with a microbenchmark of matvec cost.
A `wall_s` is stored per case for eyeballing, but never used in the comparison.

It is also not designed to catch **physics** regressions (ie, giving the wrong answer),
and many of these cases are significantly under-resolved. Physics benchmarks are
included in the existing test suite (``tests/test_solve.py``) where yancc is compared
against MONKES and SFINCS.

The cases here are some that have been found to be difficult for the multigrid
preconditioner, along with some easier cases. Many of the cases have been made much
faster by tuning the default multigrid settings, so we keep them here to catch any
changes that undo that tuning.

The solver always runs with the default settings for `solve_dke`, so any
change in `nmv`/`success` reflects a change in the shipped code. There is no
config knob — A/B testing is done by running on two git checkouts.

Run from the repository root:

```bash
# quick (~5 min) sanity subset
python benchmarks/bench_dke.py run --tier smoke --out smoke.json

# full matrix (collisionality + resolution scans, geometry spread,
# high-nu*/high-nx corners, 1-species controls)
python benchmarks/bench_dke.py run --tier all --out bench.json

# run specific case(s) by name (overrides --tier); repeatable / comma-separated
python benchmarks/bench_dke.py run --list                 # show all case names
python benchmarks/bench_dke.py run --case ncsx_2sp_nu1e-2 --out one.json
python benchmarks/bench_dke.py run --case hsx_2sp_1e-1,w7x_2sp_3e-2 --out two.json

# diff current against the committed baseline; exits nonzero on any regression
python benchmarks/bench_dke.py compare benchmarks/baseline.json nightly.json
```

## Regression check across a change

```bash
git checkout main    && python benchmarks/bench_dke.py run --out base.json
git checkout feature && python benchmarks/bench_dke.py run --out feat.json
python benchmarks/bench_dke.py compare base.json feat.json
```

`compare` flags a **regression** (nonzero exit) only when current is strictly
worse than baseline: a `success` flip `converged -> FAIL`, or `nmv` up by more
than 5%. Fixes, speed-ups, and new/dropped cases are reported but don't fail the
gate.

## Baseline

Commit a single blessed run as `benchmarks/baseline.json` and refresh it
deliberately when the shipped behavior changes for a known-good reason.

## The case matrix

Defined in `cases_dke.py` as a list of `Case` dataclasses (equilibrium, species
count, target ion `nustar`, Er, grid resolution, tolerances).

## The monoenergetic benchmark

`bench_mdke.py` is the monoenergetic sibling of `bench_dke.py`, with the same CLI
(`run`/`compare`, `--tier`, `--case`, `--list`, `--out`) and the same regression rules.
It solves `solve_mdke` with production defaults, so a change in `nmv`/`success`
reflects a change in the shipped code.

The monoenergetic problem has no species or speed grid: each case is parametrized
directly by the two DKES database axes - collisionality `nuhat` (ν/v, in 1/m) and
normalized radial electric field `erhat` (Er/v). The matrix in `cases_mdke.py` sweeps
`nuhat` in `[1e-5, 1e2]` and `erhat` in `[0, 1e-1]` across geometry; the
low-collisionality / finite-`erhat` corner is the resonant regime that stresses the
preconditioner. `solve_mdke` solves two RHS per case, so the recorded `nmv`/`niter` are
the sums, `success` is the AND, and `res` is the worse of the two (per-RHS matvec counts
are also stored but never compared).

```bash
python benchmarks/bench_mdke.py run --tier smoke   --out smoke_mdke.json
python benchmarks/bench_mdke.py run --tier nightly --out nightly_mdke.json
python benchmarks/bench_mdke.py run --case w7x_nu1e-3_er0 --out one.json
python benchmarks/bench_mdke.py compare base_mdke.json feat_mdke.json
```
