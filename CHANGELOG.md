Changelog
=========


Unreleased
----------
### New features
- New smoothers, including a theta/zeta plane smoother using surface averaged wind
  and fast diagonalization via a 2d Fourier transform, and theta/zeta line smoothers retaining coupling
  between the L0 and L1 Legendre modes in pitch to damp errors that are not smoothed
  by collisions.
- The multigrid option ``smooth_type`` is now a comma separated string listing the
  smoothers to apply and their order, for both the DKE and MDKE. The new defaults are
  ``"plane,s,x,a,l01t,l01z"`` for the DKE (the frozen plane, block-Jacobi lines in
  species, speed and pitch, and new ``"l01t"`` and ``"l01z"`` theta and zeta line
  smoothers acting on the lowest two Legendre moments in pitch) and ``"plane,a"`` for
  the MDKE (the frozen plane and pitch lines). The previous integer values are no
  longer accepted. See the tuning guide.
- The multigrid option ``smooth_weights`` can now be a dict mapping entries of
  ``smooth_type`` to their relaxation weight, with the other smoothers using their
  defaults. A single value now applies to every smoother.
- ``solve_dke`` and ``solve_dke_ambipolar`` take a new ``mesh`` argument, a
  ``jax.sharding.Mesh`` with axes ``"species"`` and/or ``"speed"``, to split a single
  solve across several devices and reduce the memory needed on each. See the
  "Multiple devices" section of the performance docs.
- The recycled Krylov subspace ``info["U"]``, ``info["C"]`` returned by ``solve_dke``
  and ``DKESolution.f1_krylov`` are now tuples of the parts for ``f1`` and for the
  source terms, rather than single stacked arrays. The ``U`` and ``f1`` warm start
  options of ``solve_dke`` accept either form.

### Performance improvements
- The direct solve on the coarsest multigrid level now applies a step of iterative
  refinement against the coarse operator, removing most of the error left by the LU
  factorization of badly scaled multispecies coarse matrices.
- All smoothers on a multigrid level are now built from a single shared operator,
  reducing the compile time of the preconditioner by about a third.


v0.0.2
------
### New features
- New function ``solve_dke_ambipolar`` to find the ambipolar radial electric field(s)
  where the radial current vanishes, along with the corresponding DKE solutions. Multiple
  roots are found using a Newton/secant iteration with deflation, reusing the Krylov
  solution and recycled subspace between solves. See the "Ambipolar root finding
  options" section of the advanced tuning docs for details.
- ``Field`` creation methods can now be ``vmap``ed over radial position ``rho``.
- MDKE smoothers now support ``smooth_solver="banded"`` and ``"cr"`` storage formats.


### Performance improvements
- Refactoring to reduce compile time, should be ~50% less
- ``RosenbluthPotentials`` with ``quad=False`` now use a fixed Gauss-Legendre quadrature
  rather than using incomplete gamma functions. This is both faster and more accurate
  over the ranges commonly in use (``nx <= 20``, ``nL<=16``). ``quad=True`` still uses
  adaptive quadrature which is the most robust but slower.
- ``solve_dke`` now measures the residual of multispecies problems in an entropy-weighted
  norm, so each species' residual is measured relative to its own Maxwellian rather than
  being dominated by the species with the largest distribution function. This changes
  the meaning of ``rtol`` and the reported residual in ``info["res"]``.
- The energy scattering collision operator now uses a Galerkin discretization in speed,
  which guarantees the discrete operator is definite. The previous collocation
  discretization could introduce spurious unstable modes that caused slow or failed
  convergence in some multispecies cases.
- Reduced GPU memory usage in the Krylov solvers, avoiding duplicate copies of the
  Krylov basis and recycled subspace.
- The direct solve on the coarsest multigrid level now equilibrates the coarse matrix
  before factoring it. This improves robustness for multispecies problems, where the
  coarse matrix can be badly scaled and previously could stall the outer Krylov solve.
- The dense coarse grid operator is now built a chunk of columns at a time, which
  significantly reduces peak memory. The chunk size can be set with the
  ``as_matrix_chunk`` key of ``multigrid_options``.

### Bug fixes
- Fix small indexing bug with monoenergetic distribution function. Moments such as
  ``D_ij`` are unaffected, but directly indexing the distribution function was
  mis-ordered.
- ``vmap`` over ``solve_dke`` now no longer prints on every iteration with
  ``verbose=True``. Now the krylov solvers only print when requested, and when
  ``vmap``ed they print the largest residual in the batch.
- Fix ordering of particle and heat sources in ``DKESolution`` for multispecies cases.
  Sources are now correctly paired per species; previously the particle and heat
  sources could be assigned to the wrong species when ``ns > 1``.
- Fix calculation of the covariant field components ``I`` and ``G`` for fields that are
  not in Boozer coordinates.
- Fix multigrid coarsening when one or more axes are clipped at their minimum
  resolution. The remaining axes now absorb the extra coarsening, so each coarse level
  is close to its target size rather than being under-coarsened.
- Fix Krylov subspace recycling for transposed solves (used for reverse mode AD), and
  for recycled subspaces containing zero or linearly dependent vectors, which could
  previously corrupt the residual.
- Krylov solvers no longer return NaN when the initial guess is already the exact
  solution.

### Misc
- New dependency on ``optimistix``. Added support for python 3.14, and extended the
  supported version ranges of ``jax``, ``numpy``, ``quadax`` and ``scipy``.

v0.0.1
------
- Initial release
