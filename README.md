# Vortex noise as a loss meter: NV spin relaxometry of trapped vortices in tantalum films

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022201.svg)](https://doi.org/10.5281/zenodo.23022201)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Open-source code for the article
"Listening to Vortex Noise: Diamond Spin Sensors Reveal Which Trapped
Vortices Cause Microwave Loss in Tantalum Qubit Films"
(T. M. Mahim, M. M. Rahman, A. S. M. Mohsin, BRAC University).

The code computes the thermal magnetic noise of a trapped vortex at a
nitrogen-vacancy (NV) spin, converts it into the microwave loss of the same
vortex through the fluctuation-dissipation theorem, simulates field-cooled
tantalum films with hole lattices (pyTDGL), tests the key relations with
Brownian dynamics, and regenerates every number and figure of the article.

## Install

    pip install -r requirements.txt      # Python 3.11

Figures use Times New Roman when its TrueType files are found in `~/.fonts`
(or in `fonts/` next to `figstyle.py`); otherwise a Times-like fallback is used.

## Files

| File | Purpose |
|---|---|
| `params.py` | every material, sensor and device parameter, with its source or marked as assumed |
| `physics.py` | finite-thickness London vortex field, displacement noise, NV rates, FDT loss, Mattis-Bardeen conductivity, slab (quasiparticle) noise, resonator loss |
| `tests.py` | closed-form self-tests of `physics.py` (all must print PASS) |
| `tdgl_common.py` | pyTDGL device builders and vortex detection |
| `figstyle.py` | figure style (Times New Roman, colorblind-safe colors) |
| `scripts/01_fieldcool_sweep.py` | TDGL field-cooled census versus cooling field and hole size |
| `scripts/02_tdgl_ac_loss.py` | TDGL 5 GHz drive: dissipation of hole-held versus interstitial flux |
| `scripts/03_langevin_fdt.py` | Brownian-dynamics checks of the FDT, the T1 formula and echo dephasing |
| `scripts/04_nv_maps.py` | static field and relaxation maps above the TDGL census |
| `scripts/05_observables.py` | single-vortex observables, temperature and sensitivity scans |
| `scripts/06_inference.py` | synthetic-measurement test (recovering drag, pinning and loss) |
| `scripts/07_figures.py` | all figures of the article and its supplementary material |
| `run_all.sh` | full reproduction (about 8 CPU hours) |

All scripts are run from this folder and write to `data/`; figures go to
`figures/` (or `$FIG_DIR`).

## Reproduce

    ./run_all.sh

To redraw the figures from the archived results instead, unzip the Zenodo
archive, copy the contents of its `data_database/` and `data_models/` folders
into `data/`, and run `python3 scripts/07_figures.py`.

## Data

The simulation database and model outputs of this article are archived on
Zenodo: https://doi.org/10.5281/zenodo.23022201 (version 2.0). The concept DOI
https://doi.org/10.5281/zenodo.21498662 always resolves to the latest version.

## Citation

See `CITATION.cff`. License: Apache-2.0.
