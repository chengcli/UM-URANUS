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
Snapy registers ``net_flux`` as ``CCF`` with units ``W/m^2``; this driver
requires that metadata entry in the installed Snapy library.

Numbered restart bundles are written every ten days, and a final restart is
written when the driver finishes. A short ``--max-cycles`` test always writes
a final restart; NetCDF files follow their configured output cadence.
