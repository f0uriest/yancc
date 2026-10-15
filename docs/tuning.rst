================
Advanced Tuning
================

This page documents the kwargs accepted by :func:`~yancc.solve_dke`,
:func:`~yancc.solve_mdke` and :func:`~yancc.solve_dke_ambipolar` beyond
the ones shown in the quickstart, the keys of the ``multigrid_options`` dictionary
that controls the preconditioner, and the keys of the ``root_options`` dictionary that
controls the search for ambipolar roots. These are intended for users who already have a working
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
by both :func:`~yancc.solve_dke` and :func:`~yancc.solve_mdke` (passed as
``**options``):

``rtol`` *(float, default 1e-5)*
   Relative tolerance: convergence is reached when
   :math:`\|r\| \le \max(\mathrm{rtol}\,\|b\|,\; \mathrm{atol})`. Relative error in
   output moments and fluxes is usually 1-2 orders of magnitude larger than ``rtol``.
   Number of iterations is roughly linear in ``log(rtol)``, but can be much larger at
   very low ``rtol``.

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

``nL`` *(int, default 8)*
   Number of Legendre modes used to expand the Rosenbluth potentials in the
   field-particle collision operator. In most cases 4 is sufficient, but the
   cost of increasing is negligible so we use a default of 8.

``quad`` *(bool, default False)*
   Use adaptive quadrature to compute the Rosenbluth potential Green's functions,
   otherwise uses a fixed Gauss-Legendre quadrature rule. The fixed Gauss-Legendre is
   accurate to machine precision up to nx=20 and much faster so adaptive quadrature
   should generally only be used for verification or extreme cases.

``operator_weights`` *(array, length 8)*
   Scale factors for the 8 sub-operators that make up the DKE:

      0. :math:`\dot{x} \frac{\partial f}{\partial x}`
      1. :math:`\dot{\alpha} \frac{\partial f}{\partial \alpha}`
      2. :math:`\dot{\theta} \frac{\partial f}{\partial \theta}`
      3. :math:`\dot{\zeta} \frac{\partial f}{\partial \zeta}`
      4. :math:`C_L`
      5. :math:`C_E`
      6. :math:`C_F`
      7. :math:`f`

   The last entry is zeroed by default; the others are 1. The last entry can be used to
   add Krook style diffusion. Use this to selectively turn parts of the operator off
   (e.g. zero out :math:`C_E` and :math:`C_F` to recover a Lorentz operator).

``fluid_correction`` *(bool, default True)*
   Whether to precede each application of the multigrid preconditioner with an
   exact correction of the density, energy and parallel momentum of each species
   at every point of the flux surface. This makes the solve much more robust at high
   collisionality, at the cost of a setup step and one extra operator application
   per iteration (usually less than 2% additional cost per iteration, and can reduce
   the required number of iterations by 5-10x). The correction of each species is
   weighted smoothly from 0 at low collisionality to 1 at high collisionality, where
   it is needed most, and it is skipped when no species is collisional.

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
:math:`(a, \theta, \zeta)` axes. Pass options as a dict via the
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

``coarsening_factor`` *(int or float, default 2.5)*
   Factor by which each axis is coarsened between levels. Mutually exclusive
   with ``max_grids``. Can be as large as 3-4 before convergence significantly decays
   for many problems.

``max_grids`` *(int, default None — derived from coarsening_factor)*
   Maximum number of multigrid levels. Mutually exclusive with
   ``coarsening_factor``. Compile time is superlinear in the number of grid levels,
   so capping this to 3 or 4 will keep compile time reasonable without affecting runtime
   for most problems.

``min_nt``, ``min_nz``, ``min_na`` *(int, default 5)*
   Minimum resolution in each axis on the coarsest grid. Note that values less than
   5 may require lower order finite difference stencils.

``resolutions`` *(list of (ns, nx, na, nt, nz) tuples, default None — auto)*
   Manually specify the resolutions at every level, fine to coarse. Pass
   this only if you have already characterized the problem and the
   automatic coarsening from ``coarse_N`` / ``coarsening_factor`` is
   inadequate.

``as_matrix_chunk`` *(int, default 512)*
   Number of columns of the coarse grid operator to build at a time for the direct
   solve there. Peak memory during the build scales with this, so lower it if you are
   memory limited and raise it (or pass ``None`` to build every column at once) to
   save a small amount of setup time. Does not affect the result.

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

``smooth_type`` *(str, default "plane,s,x,a,l01t,l01z" for DKE, "plane,a,t,z" for MDKE)*
   Comma separated list of the smoothers applied on each multigrid level, in the
   order given. Each entry is one of:

   - ``"a"``, ``"t"``, ``"z"``: block-Jacobi line smoothers along pitch angle, theta
     and zeta. The DKE also has ``"x"`` and ``"s"``, line smoothers along speed and
     species.
   - ``"plane"``: a (theta, zeta)-plane smoother that uses only the surface averaged
     drift and diagonalizes via a 2d Fourier transform so it is fast and memory
     efficient.
   - ``"l01t"``, ``"l01z"``: theta and zeta line smoothers acting only on the
     lowest two Legendre moments in pitch (the flux-surface density- and
     flow-like parts of the distribution).

   For example ``"z,t,a,x,s"`` uses only the block-Jacobi line smoothers for the DKE.
   The moment smoothers only act on part of the distribution function, so they should
   be combined with smoothers that cover the rest. Adding the block-Jacobi theta and
   zeta lines to the defaults (e.g. ``"plane,s,x,a,l01t,l01z,t,z"`` for the DKE or
   ``"plane,a,t,z,l01t,l01z"`` for the MDKE) usually reduces the number of iterations
   further, but increases the cost of each iteration by more. ``smooth_weights``
   applies to the line and plane smoothers.

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
   specially tuned weights based on the collisionality of the problem. A single value
   applies to every smoother. A dict with entries from ``smooth_type`` as keys, e.g.
   ``{"plane": 0.8, "l01t": 0.9}``, sets the weight of each of those smoothers, with
   the others using their defaults.

``p1``, ``p2`` *(default "2d", 2)*
   Finite-difference order used inside the preconditioner. Lower order than
   the operator (which defaults to ``p1="4d", p2=4``) is intentional — the
   preconditioner only needs to be a good approximation, and lower order is
   cheaper.

``gauge`` *(default True for the MDKE, "shift" for the DKE)*
   How the preconditioner handles the null space of the operator (for the DKE, a
   density and an energy mode for each species). ``True`` fixes it by replacing the
   equations at one grid point on every level. For the DKE, ``"shift"`` keeps the
   null space on every level and shifts it away from zero in the direct solve on the
   coarsest level, which converges more reliably; ``True`` is still accepted.
   ``False`` leaves the null space in place, so the coarsest solve is singular; it
   exists primarily for verification. The MDKE accepts only ``True`` or ``False``.

DKE-only multigrid options
--------------------------

``operator_weights``, ``smoother_weights``
   As above for the operator. The preconditioner defaults to the same
   weights as the main operator but allows them to be specified
   independently.

Ambipolar root finding options
==============================

:func:`~yancc.solve_dke_ambipolar` searches for the values of ``Erho`` where the
radial current :math:`J_\rho = \sum_s q_s \Gamma_s` vanishes, solving the DKE at every
trial value. Each solve is done by :func:`~yancc.solve_dke`, so the Krylov and
DKE-only options above, and ``multigrid_options``, apply to every solve of the search
and can be passed as ``**options``, with the following differences:

- ``rtol`` defaults to ``1e-2 * ftol`` (``ftol`` is described below). It sets the
  accuracy of the converged roots.
- ``k`` defaults to ``10 * num_roots``. The recycled subspace is carried across every
  solve of the search, so it pays to make it larger than for a single solve.
- ``f1`` and ``U`` are not accepted, since the search manages the warm start between
  solves itself. ``M``, ``B`` and ``C`` are accepted as for
  :func:`~yancc.solve_dke`; note that a given ``M`` is used at every value of
  ``Erho``.

The search itself is controlled by the following kwargs:

``reuse_preconditioner`` *(bool, default False)*
   Build the preconditioner once, at the first guess, and use it for every solve
   instead of rebuilding it at each value of ``Erho``. This saves the setup cost of
   each solve, at the cost of more iterations far from the first guess, since the
   preconditioner depends on ``Erho``. Most useful when the bounds are narrow or the
   preconditioner build dominates the cost of a solve.

``scale`` *("auto" or float, default "auto")*
   Scale :math:`J` used to normalize the radial current, so that roots are found for
   :math:`\sum_s q_s \Gamma_s / J`. With ``"auto"``,
   :math:`J = \sum_s |q_s \Gamma_s|` at the first guess, which costs one extra solve
   (it also warm starts the search, so it is not wasted). Pass a value in A·m⁻³ to
   skip that solve, or to make ``ftol`` mean the same thing across a set of runs.

``adaptive_rtol`` *(bool, default True)*
   Solve the DKE loosely while the normalized radial current is far from zero, and
   tighten the tolerance to ``rtol`` as a root is approached. Away from a root the
   solve only has to locate the next step of the search, so this saves iterations
   without affecting the accuracy of the roots. Set to False to solve every step to
   ``rtol``.

Root finding options
--------------------

Roots are found one at a time by a Newton type iteration, using deflation to avoid
converging again to roots already found. The first searches start from the guesses in
``Erho0``. Afterwards, the points already evaluated are used to look for sign changes
of the radial current that the roots found so far do not explain, and a search is
started inside each such interval. When none are left, the bounds are sampled, then
points in the widest unexplored intervals (which is how a pair of roots with no sign
change between them is found), and finally a search is started from the point that came
closest to a root. The search stops once ``num_roots`` roots are found or all of this
is exhausted. Pass options as a dict via the ``root_options`` argument:

.. code-block:: python

    Erho, sols, info = solve_dke_ambipolar(
        field, pitchgrid, speedgrid, species, num_roots=3,
        root_options={"ftol": 1e-4, "method": "secant", "interior_samples": 6},
    )

``ftol`` *(float, default 1e-4)*
   A search has converged when the absolute value of the normalized radial current
   (see ``scale``) is below ``ftol``.

``xrtol`` *(float, default 1e-6)*, ``xatol`` *(float, default 0.0)*
   A search has also converged when the step in ``Erho`` is smaller than
   ``xrtol * |Erho|`` or ``xatol`` (in Volts).

``maxiter`` *(int, default 20)*
   Maximum number of iterations per search. Each iteration costs at least one DKE
   solve.

``method`` *("secant" or "newton", default "secant")*
   How the derivative of the radial current with respect to ``Erho`` is found.
   ``"newton"`` differentiates the DKE solve at every iteration, which costs roughly
   another solve. ``"secant"`` estimates the derivative from the last two iterations
   while the search is making progress, and only differentiates otherwise. It usually
   needs a similar number of iterations for a much smaller cost, but is less reliable
   at finding roots that are close together; try ``"newton"`` if a pair of close roots
   is being missed.

``max_stall`` *(int, default 3)*
   A search is abandoned once this many consecutive iterations fail to reduce the
   radial current below the smallest value that search has reached, such as when the
   iterates cycle or are pushed against a bound. A search whose iterates bracket a
   root is never abandoned this way.

``probe_steps`` *(int, default 1)*
   Number of iterations allowed when sampling a bound, which is done to look for a
   sign change rather than to converge to a root.

``interior_samples`` *(int, default 6)*
   Number of points inside the bounds sampled once the guesses, sign changes and
   bounds are exhausted, each splitting the widest interval not yet explored. The
   samples are concentrated towards the first guess while still covering the whole
   interval. Increase this if roots are close together compared to the bounds and
   some are being missed.

``best_searches`` *(int, default 2)*
   Number of searches started from the point with the smallest radial current found
   so far, run after the interior samples.

``history_size`` *(int, default None — every point evaluated)*
   Number of evaluated points kept for detecting sign changes. The history is small
   compared to the DKE solutions, so there is rarely a reason to change this.

A search that does not find a new root still costs DKE solves, so when fewer than
``num_roots`` roots exist the whole fallback sequence above is run before the function
returns. Setting ``num_roots`` to the number of roots you expect, and narrowing
``bounds`` when you know where they are, is the most effective way to keep the cost
down.

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
