# NV Micromembrane Vortex Sensing: Simulation Codebase

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21498663.svg)](https://doi.org/10.5281/zenodo.21498663)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Complete open-source simulation framework for the manuscript
"Simulation Co-Design of Free-Standing Diamond Micromembrane Quantum Sensors
for Nanoscale Imaging of Vortex Matter in Tantalum Superconducting Circuits".

## Dependencies

Python 3.11 with `tdgl` (pyTDGL 0.9), `numpy`, `scipy`, `matplotlib`,
`scikit-learn` (optional). Install: `pip install tdgl`.
LaTeX with IEEEtran for the manuscript.

## Layout

- `params.py` - all material and device constants with sources
- `tdgl_common.py` - device builders, field-cool protocols, census loader
  (converts pyTDGL's coherence-length mesh units to um)
- `figstyle.py` - shared publication figure style (colorblind-safe)
- `scripts/00_calibrate.py` - current-scale calibration of the bare strip
- `scripts/01b_targets_seeded.py` - field-cooled vortex configurations
  (nucleation-pulse protocol; production data)
- `scripts/01c_targets_epscool.py` - near-Tc protocol (documents vortex
  expulsion in clean mesoscopic strips; not used for production data)
- `scripts/02_pinning_iv.py` - staircase IV, bare vs antidot, flux-flow
  suppression and the depinning-current lower bound
- `scripts/06_microwave_drive.py` - direct 2.87 GHz TDGL drive
- `scripts/03_nv_observables.py` - Pearl kernel, Biot-Savart NV-plane maps,
  sensitivity model, two-channel Langevin noise spectroscopy
- `scripts/04_reconstruction.py` - physics-informed inversion (joint NNLS +
  local fits), CRB, occupancy classification
- `scripts/05_covariance_dynamics.py` - two-NV covariance magnetometry of
  vortex telegraph hopping
- `scripts/07_analysis.py` - consolidated results -> `data/results.json`
- `scripts/08_figures.py` - all manuscript figures

## Reproduction

Run from the repository root, in order:

```
python3 scripts/00_calibrate.py
python3 scripts/01b_targets_seeded.py
python3 scripts/02_pinning_iv.py
python3 scripts/06_microwave_drive.py
python3 scripts/03_nv_observables.py
python3 scripts/04_reconstruction.py
python3 scripts/05_covariance_dynamics.py
python3 scripts/07_analysis.py
python3 scripts/08_figures.py
```

Total compute is about two CPU-hours; every figure regenerates from
`data/results.json` and the raw `.npz` outputs.

## Availability

- Code (this repository): https://github.com/Tanvir-Mahmud-Mahim/nv-membrane-vortex-sensing
- Simulation database and derived model outputs: Zenodo,
  doi:10.5281/zenodo.21498663 (replace with the DOI assigned at deposit;
  see `zenodo/INSTRUCTIONS.md` in the project folder). The `data/*.npz`
  files are distributed through Zenodo rather than git.

## Material anchors (all openly published)

- Ta type-A films: Bahrami et al., arXiv:2503.03168 (Tc, Hc2, rho_n, eta)
- Al surface losses: Hedrick et al., arXiv:2603.13183
- Diamond micromembranes: Pakpour-Tabrizi et al., Adv. Opt. Mater. 2026,
  doi:10.1002/adom.202503864 (NV depth, T2, x7 collection gain)
- Covariance magnetometry: Rovny et al., Science 378, 1301 (2022)

## License

Code is released under the Apache License 2.0 (see `LICENSE`).
The Zenodo dataset is released under CC BY 4.0.

## How to cite

If you use this code or data, please cite:

> T. M. Mahim, M. M. Rahman, and A.S.M. Mohsin, "Imaging Single Vortices
> and Their Losses in Tantalum Superconducting Circuits With a
> Pick-and-Place Diamond Quantum Sensor: An End-to-End Simulation Study,"
> 2026. Code: https://github.com/Tanvir-Mahmud-Mahim/nv-membrane-vortex-sensing
> Data: doi:10.5281/zenodo.21498663

A machine-readable citation is in `CITATION.cff`.
