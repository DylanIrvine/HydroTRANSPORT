# HydroTRANSPORT

Solute transport in groundwater, using analytical solutions with a source that
can change in steps (including being switched off).

Current version: 0.1.0

A Streamlit app that shows how a plume grows in steady, uniform flow, and how
it clears once the source is reduced or removed. It is part of the same suite
as HydroSILO, HydroPOINT, HydroSTITCH, HydroCAPTURE and the Hydrochem Workbench.

## Using it

1. Choose a solution. The page says what each one assumes and when to use it.
2. Enter the aquifer and transport values: porosity, velocity (or hydraulic
   conductivity and gradient), dispersivities, and optionally sorption and decay.
3. Describe the source: its size where needed, and a history table with a start
   time and a value on each row. A row with a value of zero switches it off.
4. Add observation points, the length of the calculation, and up to five times
   for the profiles and plume maps.
5. Run, then download the workbook, single figures, or everything as a zip.
   The workbook holds the concentration through time on its first sheet and
   the input settings on its second, followed by the other tables.

## Solutions

| Solution | Source | Source value | Wexler (1992) |
|---|---|---|---|
| 1D, constant concentration at the inlet | inlet at x = 0 | concentration | equation 60 |
| 1D, flux at the inlet | inlet at x = 0 | concentration of the inflow | equations 67, 68 |
| 2D, point source | point, mixed over the aquifer thickness | mass rate | equation 76 |
| 2D, line source across the flow | line of width W at x = 0 | concentration | equation 91b |
| 3D, point source | point at a chosen depth | mass rate | equation 105 |
| 3D, patch source across the flow | rectangle W by H at x = 0, at a chosen depth | concentration | equation 121b |

All assume a uniform aquifer, steady flow in the x direction, linear sorption
and first-order decay.

## Method notes

- **Units.** Calculations are in metres, days and g/m³ (the same as mg/L).
  Times can be entered and shown in years.
- **Point sources** take a mass rate (g/d), which is the injection rate times
  the injected concentration in Wexler's notation. The 2D point source divides
  it by the aquifer thickness.
- **Source that changes.** The 1D solutions without decay and the 3D point
  source have closed forms, and step changes are added by superposition. The
  others are a time integral with the source history inside it, so a step change
  does not involve subtracting two nearly equal numbers.
- **Closed forms.** Products of exp() and erfc() are evaluated with the scaled
  complementary error function, so they do not overflow when the distance is
  many times the dispersivity.
- **Time integrals.** 12-point Gauss-Legendre on panels evenly spaced in ln(τ),
  with extra panel edges around the advective arrival time and at each change in
  the source. This holds its accuracy from very early times to long after
  steady state, and close to the source.
- **Flux inlet with decay.** Wexler's equation 67 subtracts two very large
  terms when decay is weak, so the time integral is used whenever decay is on.
- **Decay** applies to dissolved and sorbed solute at the same rate.
- **Switching off.** For the constant concentration sources a value of zero
  holds the source at zero concentration. For the flux inlet and the point
  sources it stops solute entering.
- **Water table.** In the 3D solutions z is depth below a flat water table at
  z = 0, and the source sits at a chosen depth. Solute cannot cross the water
  table. This is done with a mirror image of the source above it, which is
  exact for horizontal flow. A point source on the water table gives twice the
  concentration of the same source in an aquifer with no top. There is no
  recharge and no aquifer base.
- **3D view.** Three slices (from above, from the side, across the flow) with
  the slice positions marked, and an optional interactive view of shells of
  equal concentration (Plotly).

## Checks

The test suite compares the code with references that do not share its
method:

- numerical Laplace inversion of the 1D solutions at 50 digits (mpmath), with
  and without decay and sorption;
- the point sources as an integral of instantaneous releases, by adaptive
  quadrature, and their closed-form steady states (the 2D one uses the Bessel
  function K0);
- adaptive quadrature of Wexler's equations 91b and 121b from 0.01 to 36,500
  days, including 0.01 m from the source, and for a source with step changes;
- limits (a very tall patch is a strip, a very wide strip is the 1D inlet),
  the mass balance of the flux inlet with a changing source, and zero vertical
  gradient at the water table.

```bash
pip install pytest
pytest tests/
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository layout

```
HydroTRANSPORT/
  app.py                    Streamlit front end
  hydrotransport/
    solutions.py            the six solutions and the time integral
    engine.py               source history, grids, summary tables, plot extents
    plots.py                set-up diagram and result figures
    workbook.py             xlsx writer
    run_counter.py          shared run total (Upstash Redis over REST)
  logo/                     header logo, browser-tab icon, figure badge and svg source
  tests/                    pytest suite (no network needed)
```

## Not yet included

- Unit choices beyond metres, mg/L, and days or years.
- Uncertainty ranges on the inputs.
- Finite-length columns and the Gaussian source from Wexler (1992), and
  instantaneous (slug) sources.
- More than one source.
- An aquifer base, recharge, or flow with a vertical component.

## Licence

Copyright (c) 2026 Charles Darwin University. All rights reserved. See
`LICENSE`.

## References

Wexler, E.J. (1992). Analytical solutions for one-, two-, and three-dimensional
solute transport in ground-water systems with uniform flow. U.S. Geological
Survey Techniques of Water-Resources Investigations 03-B7.
https://doi.org/10.3133/twri03B7

Ogata, A., and Banks, R.B. (1961). A solution of the differential equation of
longitudinal dispersion in porous media. U.S. Geological Survey Professional
Paper 411-A.
