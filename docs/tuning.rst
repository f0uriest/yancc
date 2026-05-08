================
Advanced Tuning
================

This page documents the kwargs accepted by :func:`~yancc.solve.solve_dke` and
:func:`~yancc.solve.solve_mdke` beyond the ones shown in the quickstart, and
the keys of the ``multigrid_options`` dictionary that controls the
preconditioner. These are intended for users who already have a working
solve and want to make it faster or push it into a regime where the defaults
no longer converge.

.. note::

   The Krylov, multigrid, and smoother modules are considered internal — the
   options described here are user-tunable but their *defaults* and *names*
   may change between releases. Pin a yancc version if you depend on a
   specific tuning.

Krylov solver options
=====================

Both solvers run a flexible GCROT(m,k) Krylov method preconditioned by a
multigrid sweep to solve the linear system :math:`Af=b`. The kwargs below are accepted
by both :func:`~yancc.solve.solve_dke` and :func:`~yancc.solve.solve_mdke` (passed as
``**options``):

``rtol`` *(float, default 1e-5)*
   Relative tolerance: convergence is reached when
   :math:`\|r\| \le \max(\mathrm{rtol}\,\|b\|,\; \mathrm{atol})`. Loosen to
   ``1e-4`` for transport moments where you don't need more digits;
   tighten only when comparing against an analytic limit.

``atol`` *(float, default 0.0)*
   Absolute tolerance. Useful when the right-hand side norm is very small
   (e.g. a near-equilibrium drive) and ``rtol`` would set an unreasonably
   tight target.

``m`` *(int, default 150)*
   Number of inner FGMRES iterations per outer restart. Larger ``m`` builds
   a richer Krylov subspace and converges in fewer outer cycles, at linear
   cost in memory.

``k`` *(int, default 10)*
   Number of recycled vectors carried between outer cycles. Per the
   GCROT(m,k) literature, values around ``m`` are best in theory, but small
   ``k`` is usually adequate when the preconditioner is good (which it is
   here).

``maxiter`` *(int, default 10)*
   Maximum number of *outer* cycles. Total matrix-vector products are at
   most ``m * maxiter``. If you see solves bumping into ``maxiter`` without
   converging, try increasing ``m`` or tuning the preconditioner - see below.

``print_every`` *(int, default 10)*
   Print Krylov residuals every N inner iterations when ``verbose >= 2``.

DKE-only options
----------------

``nL`` *(int, default 4)*
   Number of Legendre modes used to expand the Rosenbluth potentials in the
   field-particle collision operator. This very rarely needs to be increased, but the
   cost to do so is negligible.

``quad`` *(bool, default False)*
   Use quadrature to compute the Rosenbluth potential Green's functions, otherwise uses
   an exact formula involving incomplete gamma functions. Quadrature is usually slower
   so should only be used for verification or if using a non-Maxwellian speed grid.

``operator_weights`` *(array, length 8)*
   Scale factors for the 8 sub-operators that make up the DKE:

      0. :math:`\dot{x} \frac{\partial f}{\partial x}`
      1. :math:`\dot{a} \frac{\partial f}{\partial a}`
      2. :math:`\dot{\theta} \frac{\partial f}{\partial \theta}`
      3. :math:`\dot{\zeta} \frac{\partial f}{\partial \zeta}`
      4. :math:`C_L`
      5. :math:`C_E`
      6. :math:`C_F`
      7. :math:`f`

   The last entry is zeroed by default; the others are 1. The last entry can be used to
   add Krook style diffusion. Use this to selectively turn parts of the operator off
   (e.g. zero out :math:`C_E` and :math:`C_F` to recover a Lorentz operator).

Warm-starting
-------------

Both solvers accept hooks to skip the construction of the operator pieces
when running many similar problems:

``M`` *(preconditioner, default None — built automatically)*
   Pre-built preconditioner. Reuse across calls with the same field, grids,
   species, and ``Erho`` to avoid rebuilding it.

``B``, ``C`` *(default None — built automatically)*
   Source and constraint blocks of the bordered system. Constant for fixed
   field/grids/species.

``U`` *(default None — empty)*
   Initial recycled subspace for GCROT(m,k). Pass the ``U`` from a previous
   solve (returned in the ``info`` dict) to warm-start a similar problem.
   Particularly effective for parameter sweeps for related problems.

``f1`` *(default None — zeros)*, ``f2`` *(MDKE only)*
   Initial guess for the distribution function. Useful when continuing from
   a related solve.

The MDKE solves two right-hand sides internally and accepts ``f1, f2, U1, U2``
as separate warm-start hooks for each.

Multigrid preconditioner options
================================

The preconditioner is a geometric multigrid cycle on the
:math:`(\theta, \zeta, \xi)` axes. Pass options as a dict via the
``multigrid_options`` argument to either solver:

.. code-block:: python

    sol, info = solve_dke(
        field, pitchgrid, speedgrid, species, Erho=Erho,
        multigrid_options={
            "coarse_N": 8000,
            "v1": 3, "v2": 3,
            "smooth_method": "standard",
            "cycle_index": 1,
        },
    )

Coarsening
----------

``coarse_N`` *(int, default 8000)*
   Target size of the coarsest grid (product of the active dimensions).
   Smaller is cheaper per cycle but less effective at damping long-wavelength
   error. Increase if you see slow convergence on large problems.

``coarsening_factor`` *(int or float, default 2)*
   Factor by which each axis is coarsened between levels. Mutually exclusive
   with ``max_grids``. Can be as large as 3-4 before convergence significantly decays
   for many problems.

``max_grids`` *(int, default None — derived from coarse_N)*
   Maximum number of multigrid levels. Mutually exclusive with
   ``coarsening_factor``. Compile time is superlinear in the number of grid levels,
   so capping this to 3 or 4 will keep compile time reasonable without affecting runtime
   for most problems.

``min_nt``, ``min_nz``, ``min_na`` *(int, default 5)*
   Minimum resolution in each axis on the coarsest grid. Bumping ``min_na``
   up to 9 or 11 sometimes helps at very low collisionality where the
   trapped/passing boundary layer is the bottleneck. Note that values less than 5 may
   require lower order finite difference stencils.

``resolutions`` *(list of (ns, nx, na, nt, nz) tuples, default None — auto)*
   Manually specify the resolutions at every level, fine to coarse. Pass
   this only if you have already characterized the problem and the
   automatic coarsening from ``coarse_N`` / ``coarsening_factor`` is
   inadequate.

Cycle and smoothing
-------------------

``cycle_index`` *(int, default 1 for DKE, 3 for MDKE)*
   1 = V cycle, 2 = W cycle, etc. Higher cycle indices cost more per outer
   iteration but may converge in fewer outer iterations in cases where the
   preconditioner fully captures the spectrum, but often this is not the case and
   krylov iterations are needed to damp isolated unstable eigenvalues; for the MDKE the
   3 cycle is the default, for the more challenging DKE the V cycle + additional
   krylov iterations pays off.

``v1``, ``v2`` *(int, default 3)*
   Number of pre- and post-smoothing iterations on each level. Increasing
   to 4–5 is sometimes the cheapest way to recover convergence on stiff
   problems.

``smooth_method`` *(str, default "standard")*
   How the smoother is applied. Choices: ``"standard"`` (block Jacobi, which is tuned
   to decrease the error but may not decrease the residual), ``"krylov1"``,
   ``"krylov2"``, ``"krylov1s"``, ``"krylov2s"`` (Krylov accelerated smoothers
   ensure a strict decrease in the residual but may degrade multigrid performance.
   The ``s`` variants only attempt to reduce the high frequency residuals). Krylov
   smoothing is usually overkill for the MDKE but can help the full DKE at
   very high collisionality.

``coarse_method`` *(str, default "standard")*
   Same set of choices, applied at the coarse-grid correction step.

``interp_method`` *(str, default "linear")*
   Inter-grid interpolation method, passed through to ``interpax.interp1d``.
   For a 2nd order PDE like the DKE linear interpolation is sufficient, so there is
   little benefit in changing this.

``smooth_solver`` *(default None)*
   Override the linear solver used inside each smoothing step. Either ``"dense"`` or
   ``"banded"``. ``None`` uses a sensible per-level default. ``"banded"`` requires
   significantly less memory and is faster at high resolution but may be numerically
   unstable at very low collisionality for the MDKE.

``smooth_weights`` *(default None)*
   Optional damping weights applied to each smoother. The default ``None`` uses
   specially tuned weights based on the collisionality of the problem.

``p1``, ``p2`` *(default "2d", 2)*
   Finite-difference order used inside the preconditioner. Lower order than
   the operator (which defaults to ``p1="4d", p2=4``) is intentional — the
   preconditioner only needs to be a good approximation, and lower order is
   cheaper.

``gauge`` *(bool, default True)*
   Whether to fix the gauge of the constraint equations inside the
   preconditioner. Keep this True; it exists primarily for verification.

DKE-only multigrid options
--------------------------

``smooth_type`` *(int, default 1)*
   Selects between two structurally different smoothers for the DKE
   preconditioner. ``1`` is the default; ``2`` uses a different block
   factorization that occasionally helps for multiple species at high collisionality.

``operator_weights``, ``smoother_weights``
   As above for the operator. The preconditioner defaults to the same
   weights as the main operator but allows them to be specified
   independently.

Diagnosing convergence problems
===============================

When a solve does not converge to ``rtol`` within ``maxiter`` outer cycles,
the right thing to look at depends on what you see with ``verbose=2``:

- **Krylov residual stalls early on, far from converged.** The
  preconditioner is the issue. Try ``v1=v2=4``, then ``smooth_method="krylov1"``,
  then increase ``coarse_N``.
- **Krylov residual decreases steadily but slowly.** Try increasing ``m``
  (richer subspace) or ``cycle_index``.
- **Inner residual decreases, but outer residual doesn't.** This usually indicates a
  sort of breakdown due to the preconditioner, try ``"smooth_solver"="dense"`` or
  reducing ``v1`` and/or ``v2`` and increasing ``maxiter``.

Setting ``verbose=3`` adds residual prints at each multigrid level so you
can see whether a particular grid is failing to smooth its error component.
