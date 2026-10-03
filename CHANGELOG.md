# Changelog

All notable changes to this code are listed here, newest first.

## v2.0.3 (3 October 2026)

Documentation only; every number is unchanged.

- Article title changed again, to "Diamond Spin Sensors Measure the Microwave
  Loss of Each Trapped Vortex in Tantalum Qubit Films" (README.md and
  CITATION.cff).

## v2.0.2 (3 October 2026)

Figure labels and documentation only; every number is unchanged.

- New article title: "Thermal Noise Measures the Microwave Loss of Each
  Trapped Vortex in Tantalum Qubit Films".
- `scripts/07_figures.py`: legends moved so that no text overlaps the data
  (Figs. 2b, 2e, 3d, 6a, S2); plain-language labels in Figs. 3d, S1, and S2.
- Reference to Bahrami et al. updated to the published version
  (Phys. Rev. B 113, 054505, 2026).

## v2.0.1 (29 September 2026)

Documentation and organisation only; every number and figure is unchanged.

- README rewritten as a step-by-step guide (installation, three ways to use
  the code, a table of scripts with run times, and a table of which script
  makes which figure).
- Scripts renumbered so the numbers match the running order:
  `04_nv_maps.py` is now `05_nv_maps.py`, and `05_observables.py` is now
  `04_observables.py`.
- `run_all.sh` now creates the `data/` folder first.
- Added this CHANGELOG.

## v2.0.0 (29 September 2026)

Code for the article "Listening to Vortex Noise: Diamond Spin Sensors Reveal
Which Trapped Vortices Cause Microwave Loss in Tantalum Qubit Films".
Data archive: https://doi.org/10.5281/zenodo.23022201

**New**
- `physics.py`: exact magnetic field of a vortex in a film of finite
  thickness; noise from thermal vortex motion; sensor (NV center) relaxation
  and dephasing rates; the noise-to-loss law (fluctuation-dissipation
  theorem); Mattis-Bardeen conductivity; noise of the film itself; loss added
  to a resonator.
- `tests.py`: eleven self-tests against exact results.
- `scripts/01_fieldcool_sweep.py`: 16 field-cooling simulations (plain films,
  150 nm and 200 nm holes, cooling fields from 0.5 to 2.5 times the matching field, two protocol checks).
- `scripts/02_tdgl_ac_loss.py`: 5 GHz drive simulations that compare flux
  held in holes with vortices between holes.
- `scripts/03_langevin_fdt.py`: independent random-motion checks.
- Single-vortex signals and parameter scans, including the temperature
  dependence of the penetration depth (`scripts/04_observables.py` since
  v2.0.1).
- Field and noise maps above the simulated films (`scripts/05_nv_maps.py`
  since v2.0.1).
- `scripts/06_inference.py`: recovery of vortex properties and loss from noisy
  fake data.
- `scripts/07_figures.py`: all figures, in Times New Roman.
- `run_all.sh` and `requirements.txt`.

**Changed**
- All tantalum values now come from one film (sample D2 of Bahrami et al.,
  arXiv:2503.03168) instead of a mix of films.
- The penetration depth (not reported for that film) is set to 50 nm and
  scanned from 35 to 100 nm; version 1 used an unsourced 100 nm.
- Microwave power is now the average of current times voltage over whole
  periods; the version 1 article used the average of the absolute value,
  which mixed in stored (reactive) power.
- The vortex drag is described as a formula-based estimate (Bardeen-Stephen),
  not as an independent measurement.

**Removed**
- The version 1 scripts for image reconstruction, covariance magnetometry,
  current-voltage staircases, and calibration, which belonged to the earlier
  manuscript.

## v1 (July 2026)

Code for the earlier manuscript on end-to-end imaging of vortices with a
diamond membrane sensor. Data archive: https://doi.org/10.5281/zenodo.21498663
