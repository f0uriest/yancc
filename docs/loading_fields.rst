============================
Loading Magnetic Fields
============================

Every yancc solve starts with a :class:`~yancc.field.Field` representing
:math:`B(\theta, \zeta)` on a single flux surface. The class has five
constructors covering the equilibrium formats yancc supports:

================================================== ================================================================
Constructor                                        Source
================================================== ================================================================
:meth:`~yancc.field.Field.from_desc`               In-memory ``desc.equilibrium.Equilibrium`` object
:meth:`~yancc.field.Field.from_vmec`               VMEC ``wout`` netCDF file
:meth:`~yancc.field.Field.from_booz_xform`         BOOZ_XFORM ``boozmn`` netCDF file
:meth:`~yancc.field.Field.from_ipp_bc`             IPP ``.bc`` (Boozer coordinate) file
:meth:`~yancc.field.Field.from_boozer`             Raw Boozer-coordinate data, no file
================================================== ================================================================

The shared invocation is::

    field = Field.from_<format>(source, rho, ntheta, nzeta, ...)

Conventions common to every constructor
=======================================

Radial coordinate
-----------------

Every constructor takes ``rho``, the square root of normalized toroidal
flux:

.. math::

    \rho = \sqrt{\psi_t / \psi_{t,\mathrm{LCFS}}} = \sqrt{s}.

If your input is :math:`s`, pass ``rho = np.sqrt(s)``. If it is the
dimensional minor radius :math:`r`, pass ``rho = r / a_minor``. yancc never
takes :math:`s` or :math:`r` directly. See :ref:`radial-coordinate` for
why this matters for gradients as well.

Surface resolution
------------------

``ntheta`` and ``nzeta`` must both be **odd**.

``nzeta`` covers a single field period :math:`[0, 2\pi / \mathrm{NFP})`,
not the full torus. The number of field periods is read from the file.

Axisymmetric (tokamak) fields
-----------------------------

For an axisymmetric device (a tokamak) the field has no toroidal variation,
so pass ``nzeta=1`` to any constructor. yancc then treats the surface as
zeta-independent: the toroidal derivative drops out
(:math:`\partial / \partial \zeta = 0`) and the value of ``NFP`` is
irrelevant. The poloidal resolution ``ntheta`` is chosen exactly as for a
stellarator. For example::

    field = Field.from_desc(eq, rho=0.5, ntheta=17, nzeta=1)

Stellarator symmetry
--------------------

``from_vmec`` and ``from_booz_xform`` accept only stellarator-symmetric
equilibria; they raise on load if asymmetric modes are present.
Asymmetric equilibria must be loaded through ``from_desc`` or constructed
manually with ``from_boozer``.

By format
=========

DESC (``from_desc``)
--------------------

The most direct path if your equilibrium is already in DESC:

.. code-block:: python

    import desc
    from yancc.field import Field

    eq = desc.examples.get("NCSX")
    field = Field.from_desc(eq, rho=0.5, ntheta=17, nzeta=37)

``eq`` must be a single ``desc.equilibrium.Equilibrium`` instance. Non-stellarator
symmetric and stellarator symmetric equilibria are both supported.

VMEC ``wout`` (``from_vmec``)
-----------------------------

.. code-block:: python

    field = Field.from_vmec("wout_NCSX.nc", rho=0.5, ntheta=17, nzeta=37)

Reads from a VMEC netCDF ``wout`` file. Stellarator-symmetric only.

BOOZ_XFORM ``boozmn`` (``from_booz_xform``)
-------------------------------------------

.. code-block:: python

    field = Field.from_booz_xform(
        "boozmn_wout_NCSX.nc",
        rho=jnp.sqrt(0.2),
        ntheta=17,
        nzeta=33,
        cutoff=1e-5,
    )

The ``cutoff`` argument drops Fourier modes with
:math:`|b_{mn}| < \mathrm{cutoff} \cdot |b_{00}|` before evaluating
:math:`B(\theta, \zeta)`. ``cutoff = 1e-5`` is a conservative trim that
removes numerically tiny modes without affecting physical structure; useful
for convergence at low collisionality. To do a spectral-content study
independent of grid resolution, vary ``cutoff`` with ``ntheta``/``nzeta``
held fixed.

The requested ``rho`` must lie within the range of surfaces on which the
Boozer transform was computed in the input file.

Stellarator-symmetric only.

IPP ``.bc`` (``from_ipp_bc``)
-----------------------------

.. code-block:: python

    field = Field.from_ipp_bc("NCSX.bc", rho=0.5, ntheta=17, nzeta=37, cutoff=1e-5)

Reads the IPP Boozer-coordinate ``.bc`` text format. The ``cutoff``
argument behaves identically to ``from_booz_xform``.

Direct Boozer construction (``from_boozer``)
--------------------------------------------

If your data is already in Boozer coordinates but not in a format yancc
reads, use :meth:`~yancc.field.Field.from_boozer` directly:

.. code-block:: python

    field = Field.from_boozer(
        rho=0.5,
        Bmag=Bmag,        # shape (ntheta, nzeta), uniform Boozer angles
        I=I,              # Boozer toroidal current  [T*m]
        G=G,              # Boozer poloidal current  [T*m]
        iota=iota,        # rotational transform     [dimensionless]
        Psi=Psi,          # total LCFS toroidal flux [Wb]
        R_major=R0,       # [m]
        a_minor=a,        # [m]
        NFP=NFP,
    )

``Bmag`` must be sampled at uniformly spaced angles on
:math:`(\theta_\mathrm{Boozer}, \zeta_\mathrm{Boozer}) \in [0, 2\pi) \times [0, 2\pi / \mathrm{NFP})`.
The endpoints should not be included (ie, ``theta=np.linspace(0, 2*np.pi, ntheta, endpoint=False)``)
Optional ``dBdt`` and ``dBdz`` can be supplied if you have analytic
derivatives; otherwise yancc computes them from ``Bmag``. ``B0`` defaults
to the surface average of :math:`|B|`.

This is also the cleanest way to construct synthetic / test fields (e.g.
analytic tokamak limits, single-helicity reductions).

Tips
====

- **Multi-surface scans.** Call the constructor once per surface; there is
  no built-in iterator. Cost per Field is small, so caching all surfaces
  upfront is fine.
- **Resampling.** :meth:`~yancc.field.Field.resample` returns a new Field
  on a different ``ntheta, nzeta`` without re-reading the source file —
  use this for spatial-grid convergence studies.
