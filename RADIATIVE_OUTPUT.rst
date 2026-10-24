Radiation and user output
=========================

The current driver writes daily hydro fields to ``uranus.out1.*.nc`` and
user variables to ``uranus.out2.*.nc``. Both use Snapy's stitched six-face
NetCDF layout. Cell-centered fields have dimensions ``(time, x1, x3, x2)``.
For the production grid this is ``(1, 64, 192, 288)`` per snapshot.

``out1`` contains density, pressure, all three local velocity components,
and the CH4/H2S vapor, cloud, and precipitation mass fractions.

``out2`` contains:

* ``temp``: temperature in K.
* ``qtol``: total non-dry mass fraction, kg/kg.
* ``rt_heating``: volumetric radiative heating, W/m^3.
* ``net_flux``: total upward-minus-downward radiative flux, W/m^2, on
  the ``CCF`` grid (horizontal cell centers and radial faces).

Following the Jupiter case, the ``path`` output writes vapor and condensate
paths separately for every species:

* Vapor: ``path_CH4``, ``path_H2S``.
* Cloud condensate: ``path_CH4_s_``, ``path_H2S_s_``.
* Precipitating condensate: ``path_CH4_s_p_``, ``path_H2S_s_p_``.

These fields have units ``kg/m^2`` and dimensions ``(time, x3, x2)``.
Snapy integrates species mass over active radial cells and divides by the
bottom face area, including the spherical geometry:

.. math::

   P_s = \frac{\sum_i \rho_i q_{s,i} V_i}{A_\mathrm{bottom}}.

Net flux includes both shortwave and longwave bands. Its dimensions are
``(time, x1f, x3, x2)``, or ``(1, 65, 192, 288)`` for the production grid.
Each radial interface is stored once, including the bottom and top
boundaries. The heating can be reconstructed from adjacent faces as

.. math::

   q_i=-\frac{A_{i+1/2}F_{i+1/2}-A_{i-1/2}F_{i-1/2}}{V_i}.

The step is clipped to hourly radiation deadlines and daily output times.
Radiation is refreshed after moist chemistry at those deadlines, before
writing the output. Heating remains fixed between solves. Thermal flux can
be nonzero on the night side; only incident shortwave radiation vanishes.
Snapy assigns the ``CCF`` grid and units ``W/m^2`` to names ending in
``_flux``; this suffix is reserved for energy fluxes. This driver requires
these suffix rules in the installed Snapy library.

Numbered restart bundles are written every ten days, and a final restart is
written when the driver finishes. A short ``--max-cycles`` test always writes
a final restart; NetCDF files follow their configured output cadence.
