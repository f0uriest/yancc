======
Theory
======

This page describes what yancc actually computes: the equations being solved,
the coordinates they are written in, and how the inputs of the public API map
onto terms in those equations. It is meant as a reference, not a derivation -
for the latter, see e.g. Helander & Sigmar [1]_ or Beidler et al. [2]_.

yancc solves two related problems on a single flux surface of a given
magnetic equilibrium:

- The **drift kinetic equation** (DKE), linearized about a local Maxwellian,
  for one or more species coupled through their collision operators. This
  gives the perturbed distribution function :math:`f_s` from which radial
  fluxes, parallel flows, and the bootstrap current follow. The formulation
  is the same as in SFINCS [3]_.
- The **monoenergetic drift kinetic equation** (MDKE), a single-species
  reduction at fixed particle speed, which yields the 3×3 monoenergetic
  transport matrix :math:`D_{ij}`. This is the same problem solved by DKES [4]_ and,
  more recently, MONKES [5]_;

Both problems are linear and time-independent: each call to
:func:`~yancc.solve.solve_dke` or :func:`~yancc.solve.solve_mdke` is a single
sparse linear solve.

Coordinates
===========

Configuration space
-------------------

A flux surface is parameterized by arbitrary poloidal and toroidal angles
:math:`(\theta, \zeta)` on the surface labeled by the radial coordinate

.. math::

    \rho = \sqrt{\psi_t / \psi_{t,\mathrm{LCFS}}},

i.e. the square root of normalized toroidal flux. **Every** radial input and
output in yancc is in :math:`\rho` (see :ref:`radial-coordinate` in the
quickstart for conversions from :math:`s` or :math:`r`).

The on-surface grid sizes are ``ntheta`` and ``nzeta``, set when the
:class:`~yancc.field.Field` is constructed.

Velocity space
--------------

Velocity space is parameterized by the speed coordinate

.. math::

    x_s = v / v_{th,s}, \qquad v_{th,s} = \sqrt{2 T_s / m_s},

and the pitch-angle :math:`a`:

.. math::

    a = -\arccos{v_\parallel / v}

(Other codes use the cosine of the pitch angle :math:`\xi = v_\parallel / v`, we choose
the angle itself because it simplifies the boundary conditions.)

With these conventions the local Maxwellian for species :math:`s` is

.. math::

    F_{M,s}(\rho, x_s) = n_s \left( \frac{m_s}{2 \pi T_s} \right)^{3/2}
                        e^{-x_s^2}.

The pitch-angle grid (:class:`~yancc.velocity_grids.UniformPitchAngleGrid`)
has ``na`` uniformly spaced points on :math:`a \in [0, \pi]`. The speed
grid (:class:`~yancc.velocity_grids.MaxwellSpeedGrid`) has ``nx`` collocation
nodes chosen as the roots of polynomials orthogonal under the Maxwellian
weight :math:`e^{-x^2}` on :math:`[0, \infty)`; this is what makes a small
``nx`` (typically 5–8) sufficient to resolve thermal moments.

The full DKE
============

Following SFINCS [3]_, yancc writes the perturbed distribution as

.. math::

    f_{1,s} = f_s - F_{M,s},

and solves the linearized drift kinetic equation for :math:`f_{1,s}` on the
flux surface:

.. math::
   :label: dke

    \big( v_\parallel \mathbf{b} + \mathbf{v}_E \big) \cdot \nabla f_{1,s}
    \;-\; C_s[f_{1,s}]
    \;=\; S_s,

where :math:`\mathbf{b} = \mathbf{B}/B`, :math:`\mathbf{v}_E` is the
flux-surface :math:`\mathbf{E} \times \mathbf{B}` drift driven by
:math:`E_\rho`, and :math:`C_s` is the linearized Fokker–Planck collision
operator including both test-particle and field-particle pieces, summed over
all kinetic species and any additional :class:`~yancc.species.LocalMaxwellian`
``background`` species.

In terms of our chosen coordinates and field components this takes the form

.. math::
    :label: dke_full

    \dot{x}_s \frac{\partial f_{1,s}}{\partial x_s}
    + \dot{a} \frac{\partial f_{1,s}}{\partial a}
    + \dot{\theta} \frac{\partial f_{1,s}}{\partial \theta}
    + \dot{\zeta} \frac{\partial f_{1,s}}{\partial \zeta}
    + C_L[f_{1,s}, f_{1,s'}] + C_E[f_{1,s}, f_{1,s'}] + C_F[f_{1,s}, f_{1,s'}]
    = -\mathbf{v}_\mathrm{drift} \cdot \nabla \rho \frac{\partial F_{M,s}}{\partial \rho}

Where the particle trajectories are given by:

.. math::
    :label: sfincs_trajectories

    \begin{align}
    \dot{\theta} &= -\frac{v_{th,s} x_s \cos{(a)} B^\theta}{B} + \frac{B_\zeta}{B^2\sqrt{g}}E_\rho \\
    \dot{\zeta} &= -\frac{v_{th,s} x_s \cos{(a)} B^\zeta}{B} - \frac{B_\theta}{B^2\sqrt{g}}E_\rho \\
    \dot{a} &= - \frac{\sin{(a)}}{2 B^2} v_{th,s} x_s \Bigg( B^\theta \frac{\partial B}{\partial \theta}
    + B^\zeta \frac{\partial B}{\partial \zeta} \Bigg)
    + \cos{(a)} \sin{(a)} \frac{1}{2B^3 \sqrt{g}} E_\rho \Bigg( B_\zeta \frac{\partial B}{\partial \theta}
    - B_\theta \frac{\partial B}{\partial \zeta} \Bigg) \\
    \dot{x}_s &= (1 + \cos^2{a}) \frac{x_s}{2B^3 \sqrt{g}} E_\rho \Bigg( B_\zeta \frac{\partial B}{\partial \theta} - B_\theta \frac{\partial B}{\partial \zeta} \Bigg)
    \end{align}

And the collision operator terms are:

.. math::
    :label: fp_collision_operator

    \begin{align}
    C_{L,ss'} &= \frac{\nu_{D,ss'}}{2 \sin a} \frac{\partial}{\partial a} \Bigg[\sin{a} \frac{\partial f_{1,s}}{\partial a}\Bigg] \\
    C_{E,ss'} &= \nu_{||,ss'}\Big[ \frac{v^2}{2} \frac{\partial^2 f_{1,s}}{\partial v^2} - \frac{v^2}{v_{th,s'}^2} \Big(1 - \frac{m_s}{m_{s'}} \Big) v \frac{\partial f_{1,s}}{\partial v}\Big] + \nu_{D,ss'} v \frac{\partial f_{1,s}}{\partial v} + 4\pi \Gamma_{ss'}\frac{m_s}{m_s'} F_{M,s'} f_{1,s} \\
    C_{F,ss'} &= \Gamma_{ss'} F_{M,s} \Big[\frac{2v^2}{v_{th,s}^4} \frac{\partial^2 G_{s'}}{\partial v^2} - \frac{2v}{v_{th,s}^2} \Big(1 - \frac{m_s}{m_{s'}} \Big) \frac{\partial H_{s'}}{\partial v} - \frac{2}{v_{th,s}^2} H_{s'} + 4\pi \frac{m_s}{m_{s'}} f_{1,s'} \Big] \\
    \nabla^2_v H_{s'} &= -4 \pi f_{s'} \\
    \nabla^2_v G_{s'} &= 2 H_{s'} \\
    \end{align}

The collision frequencies are given by:

.. math::
    :label: collisionalities

    \begin{align}
    \nu_{D,ss'} &= \frac{\Gamma_{ss'} n_{s'}}{v^3} [\mathrm{erf}(v/v_{th,s'}) - \Psi(v/v_{th,s'})] \\
    \nu_{||,ss'} &= 2 \frac{\Gamma_{ss'} n_{s'}}{v^3} \Psi(v/v_{th,s'}) \\
    \Gamma_{ss'} &= \frac{4\pi q_s^2 q_{s'}^2 \ln \Lambda_{ss'}}{(4\pi \epsilon_0)^2 m_s^2} \\
    \Psi(x) &= \frac{1}{2x^2}\Big[\mathrm{erf}(x) - \frac{2x}{\sqrt{\pi}} \exp(-x^2) \Big]
    \end{align}

where :math:`\ln \Lambda_{ss'}` is the Coulomb logarithm.

Drive term
----------

The right-hand side :math:`S_s` is a sum of three contributions weighted by
the species **thermodynamic forces**:

.. math::
   :label: forces

    \begin{aligned}
    A_{1,s} &= \frac{1}{n_s}\frac{\partial n_s}{\partial \rho}
              - \frac{q_s E_\rho}{T_s}
              - \frac{3}{2} \frac{1}{T_s} \frac{\partial T_s}{\partial \rho}, \\
    A_{2,s} &= \frac{1}{T_s}\frac{\partial T_s}{\partial \rho}, \\
    A_{3,s} &= \frac{q_s}{T_s} \frac{\langle E_\parallel B \rangle}{\langle B^2 \rangle}.
    \end{aligned}

Each is multiplied by a known geometric/Maxwellian factor:

.. math::

    S_s = \big( A_{1,s} + x_s^2 A_{2,s} \big)
          \big( -\mathbf{v}_{m,s} \cdot \nabla \rho \big) F_{M,s}
        \;+\; A_{3,s}\, B v_\parallel F_{M,s},

where :math:`\mathbf{v}_{m,s}` is the magnetic (curvature + grad-B) drift.
The :math:`A_1` term encodes the density gradient and the radial electric
field, :math:`A_2` the temperature gradient, and :math:`A_3` the inductive
parallel electric field (usually zero in stellarators).

Mapping onto the API
--------------------

The public inputs to :func:`~yancc.solve.solve_dke` correspond to
equation :eq:`dke` and :eq:`forces` as follows:

================================================ ==============================================================
API input                                        Term in the equations
================================================ ==============================================================
``species[i].temperature``, ``.density``         :math:`T_s,\, n_s` in :math:`F_{M,s}` and the forces
``species[i].dTdrho``, ``.dndrho``               :math:`\partial T_s / \partial \rho,\,
                                                 \partial n_s / \partial \rho` in :math:`A_{1,s}, A_{2,s}`
``Erho``                                         :math:`E_\rho = -\partial \Phi / \partial \rho` in :math:`A_{1,s}`
``EparB``                                        :math:`\langle E_\parallel B \rangle` in :math:`A_{3,s}`
``background``                                   Extra Maxwellian species in :math:`C_s` only
``field``                                        :math:`B,\, \mathbf{b},\, \mathbf{v}_{m,s},\, \mathbf{v}_E`
================================================ ==============================================================

The returned :class:`~yancc.solution.DKESolution` stores
:math:`f_{1,s}` on the full ``(ns, nx, na, nt, nz)`` grid; flux-surface
moments are computed on demand via ``sol.get(name)``.

The Monoenergetic DKE
=====================

The MDKE is the same equation reduced to a single species at fixed particle
speed :math:`v`, with the collision operator replaced by pure pitch-angle
scattering at frequency :math:`\nu(v)`. This is the formulation introduced
by DKES [4]_ and used by MONKES [5]_:

.. math::
   :label: mdke

    \big( v\, \xi\, \mathbf{b} + \mathbf{v}_E \big) \cdot \nabla f_1
    \;-\; \frac{\nu}{2 \sin a}
          \frac{\partial}{\partial a}
          \left[ \sin a \frac{\partial f_1}{\partial a} \right]
    \;=\; S.

After dividing through by :math:`v`, only two scalar drives remain:

.. math::

    \widehat{E_\rho} \;=\; E_\rho / v
    \quad [\mathrm{V\, s\, m^{-1}}],
    \qquad
    \widehat{\nu} \;=\; \nu / v
    \quad [\mathrm{m^{-1}}],

passed as ``erhohat`` and ``nuhat`` to
:func:`~yancc.solve.solve_mdke`. yancc solves :eq:`mdke` two times - once
per unique drive (the first and second are the same when speed is ignored) - and
assembles the 3×3 monoenergetic transport matrix

.. math::

    D_{ij} = \langle s_i^{(j)} \cdot f^{(j)} \rangle,

returned by ``sol.get("Dij")``. A DKES-normalized form using the conventions
of [4]_ is also available (``"Dij_DKES"``); see :doc:`variables`.

Outputs
=======

For the full DKE, ``sol.get(name)`` computes radial particle and heat fluxes,
parallel flows, the bootstrap current, and SFINCS-normalized variants of
each. For the MDKE only the transport matrix is computed. The complete list
of names and the array shape returned by each is in :doc:`variables`.

References
==========

.. [1] P. Helander and D. J. Sigmar, *Collisional Transport in Magnetized
       Plasmas*, Cambridge University Press (2002).
.. [2] Beidler, C. D., et al. "Benchmarking of the mono-energetic transport
       coefficients—results from the International Collaboration on Neoclassical
       Transport in Stellarators (ICNTS)." Nuclear Fusion 51.7 (2011): 076001.
.. [3] Landreman, Matt, et al. "Comparison of particle trajectories and collision
       operators for collisional transport in nonaxisymmetric plasmas." Physics of
       Plasmas 21.4 (2014).
       SFINCS code: https://github.com/landreman/sfincs.
.. [4] Hirshman, S. P., et al. Plasma transport coefficients for nonsymmetric toroidal
       confinement systems. No. ORNL/TM--9925. Oak Ridge National Lab., TN (USA), 1986.
.. [5] Escoto, F. J., et al. "MONKES: a fast neoclassical code for the evaluation of
       monoenergetic transport coefficients in stellarator plasmas." Nuclear Fusion
       64.7 (2024): 076030.
       MONKES code: https://github.com/JavierEscoto/MONKES.
