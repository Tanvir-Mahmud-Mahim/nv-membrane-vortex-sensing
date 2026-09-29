# Listening to Vortex Noise: Simulation Code

[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022201.svg)](https://doi.org/10.5281/zenodo.23022201)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Release](https://img.shields.io/badge/release-v2.0.1-green.svg)](https://github.com/Tanvir-Mahmud-Mahim/nv-membrane-vortex-sensing/releases/latest)

Code for the article **"Listening to Vortex Noise: Diamond Spin Sensors Reveal
Which Trapped Vortices Cause Microwave Loss in Tantalum Qubit Films"**
by Tanvir M. Mahim, M. Mosaddequr Rahman, and A.S.M. Mohsin
(Department of Electrical and Electronic Engineering, BRAC University).

- Repository: https://github.com/Tanvir-Mahmud-Mahim/nv-membrane-vortex-sensing
- Archived data (all simulation results used in the article): https://doi.org/10.5281/zenodo.23022201

Every number and figure in the article and its supplementary material is
produced by the scripts in this repository.

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](#5-the-scripts-step-by-step)
6. [Which script makes which figure](#6-which-script-makes-which-figure)
7. [The Python modules](#7-the-python-modules)
8. [Where the numbers come from](#8-where-the-numbers-come-from)
9. [Built-in checks](#9-built-in-checks)
10. [Notes on the calculations](#10-notes-on-the-calculations)
11. [Version history](#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

---

## 1. The idea in one minute

Superconducting qubits made of tantalum lose energy when tiny whirls of
magnetic flux, called **vortices**, get trapped in the metal and wiggle under
the microwave current. Some trapped flux sits harmlessly in small holes
patterned in the film; the rest sits between the holes and causes loss.
A normal magnetic image cannot tell the two apart.

This project shows that the **magnetic noise** of each vortex can. A basic law
of physics (the fluctuation-dissipation theorem) says that anything that
absorbs energy must also jiggle by itself at any temperature, by an exactly
related amount. A lossy vortex therefore makes magnetic noise, while flux in a
hole stays quiet. A tiny magnetic sensor in diamond (an **NV center**) placed
about 50 nm above the film hears this noise. The code computes:

- the magnetic field and noise of one vortex at the sensor,
- how many vortices end up in holes or between holes after cooling the film,
- how much microwave energy each kind of trapped flux absorbs,
- how the sensor signal converts into the energy loss of each vortex,
- independent checks of every step.

---

## 2. What is in this repository

```
nv-membrane-vortex-sensing/
|-- README.md             this guide
|-- CHANGELOG.md          what changed between versions
|-- LICENSE               Apache-2.0 license
|-- CITATION.cff          citation details (drives the "Cite this repository" button)
|-- requirements.txt      Python packages to install
|-- run_all.sh            runs every step in order (Linux and macOS)
|-- params.py             every material, sensor and device value, with its source
|-- physics.py            the physics formulas (field, noise, sensor rates, loss)
|-- tests.py              quick self-tests of physics.py (about 1 minute)
|-- tdgl_common.py        helpers for the superconductor simulations
|-- figstyle.py           figure style (Times New Roman, colour-blind-safe colours)
`-- scripts/
    |-- 01_fieldcool_sweep.py   cool films in a magnetic field; count trapped flux
    |-- 02_tdgl_ac_loss.py      drive the film at 5 GHz; measure the energy loss
    |-- 03_langevin_fdt.py      independent random-motion checks of the key formulas
    |-- 04_observables.py       single-vortex signals, temperature and parameter scans
    |-- 05_nv_maps.py           field and noise maps above the simulated films
    |-- 06_inference.py         test: recover vortex properties from noisy fake data
    `-- 07_figures.py           draw every figure of the article
```

The scripts are numbered in the order they run. Each one writes its results to
a folder called `data/`, and the figure script writes to `figures/`. These two
folders are created automatically and are not stored on GitHub; the archived
copy of `data/` is on Zenodo.

---

## 3. Installation

You need **Python 3.10 or newer** (tested with Python 3.11).

```
pip install -r requirements.txt
```

This installs `tdgl` (pyTDGL 0.9.0, the superconductor simulator), `numpy`,
`scipy`, `numba`, `matplotlib`, and `h5py`.

**Fonts (optional).** Figures use Times New Roman when its font files
(`times.ttf`, `timesbd.ttf`, `timesi.ttf`, `timesbi.ttf`) are in `~/.fonts`
or in a folder named `fonts/` next to `figstyle.py`. Otherwise a similar
Times-style font is used automatically.

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder.

### Way A: check that everything works (about 1 minute)

```
python tests.py
```

Every line must start with `PASS`, and the last line must read `ALL PASS`.

### Way B: redraw every figure from the archived results (about 2 minutes)

1. Download `NV_vortex_noise_database_and_models_v2.zip` from
   https://doi.org/10.5281/zenodo.23022201 and unzip it.
2. Create a folder named `data` inside this repository.
3. Copy the **contents** of the unzipped `data_database/` and `data_models/`
   folders into `data/`.
4. Run:

```
python scripts/07_figures.py
```

The figures appear in `figures/` (supplementary figures in
`figures/supplementary/`).

### Way C: recompute everything from scratch (about 4 hours on one CPU core)

On Linux or macOS:

```
./run_all.sh
```

On Windows, run the same commands one by one in PowerShell, in the order given
in [Section 5](#5-the-scripts-step-by-step).

---

## 5. The scripts, step by step

| Step | Command | What it does | Time* | Results |
|---|---|---|---|---|
| 0 | `python tests.py` | Checks the physics formulas against known exact results | 1 min | printed on screen |
| 1 | `python scripts/01_fieldcool_sweep.py` | Simulates 16 films cooled in a magnetic field (plain films and films with 150 nm or 200 nm holes, at 0.5 to 2.5 times the matching field, plus two protocol checks) and counts where the flux ends up | 80 min | `data/fieldcool_sweep/` |
| 2 | `python scripts/02_tdgl_ac_loss.py CASE 1000` | Drives a film with a 5 GHz current and measures the absorbed power. Run it for each CASE: `anti15_B0`, `anti15_B8`, `anti15_B12`, `anti15_B8zfc`, `bare_B0`, `bare_B4`; then run `python scripts/02_tdgl_ac_loss.py anti15_B12 500` | 7 to 21 min per case | `data/ac_loss/` |
| 3 | `python scripts/03_langevin_fdt.py` | Random-motion simulations that check the noise-to-loss law, the sensor relaxation formula, and the dephasing formula | 20 min | `data/langevin_checks.json` |
| 4 | `python scripts/04_observables.py` | Signals of one vortex at the sensor; scans of height, penetration depth, thickness, temperature, and pinning | 1 min | `data/observables.json` |
| 5 | `python scripts/05_nv_maps.py` | Field and noise maps above every simulated film (needs steps 1 and 4) | 1 min | `data/nv_maps.json`, `data/nv_maps_*.npz` |
| 6 | `python scripts/06_inference.py` | Makes noisy fake measurements and checks that the vortex properties and loss can be recovered | 10 min | `data/inference.json` |
| 7 | `python scripts/07_figures.py` | Draws all figures | 2 min | `figures/` |

\*Times measured on a shared two-core computer; a free modern core is usually faster.

Steps 1 and 2 are the slow simulations. Steps 3 to 7 are fast.
All random numbers use fixed starting values (seeds), so the results repeat
exactly on the same computer.

---

## 6. Which script makes which figure

| Figure in the article | Content | Data from | Drawn by |
|---|---|---|---|
| Fig. 1 | Overview (graphical abstract) | steps 1, 5 | `07_figures.py abstract` |
| Fig. 2 | One vortex: field, noise, temperature, loss | step 4 | `07_figures.py vortex` |
| Fig. 3 | Field and noise maps; flux count versus cooling field | steps 1, 5 | `07_figures.py maps` |
| Fig. 4 | 5 GHz drive: holes versus interstitial vortices | step 2 | `07_figures.py tdgl` |
| Fig. 5 | Independent random-motion checks | step 3 | `07_figures.py checks` |
| Fig. 6 | Reading out the pinning; recovering the loss | steps 4, 6 | `07_figures.py readout` |
| Fig. S1 to S3 | Supplementary: all simulated films, drive traces, noise spectra | steps 1 to 3 | `07_figures.py supp` |

---

## 7. The Python modules

| File | What it contains |
|---|---|
| `params.py` | All values used in the calculations, each with its source or marked as assumed |
| `physics.py` | Magnetic field of a vortex in a film of finite thickness; vortex motion noise; sensor relaxation and dephasing rates; the noise-to-loss law; superconductor conductivity (Mattis-Bardeen); noise of the film itself; loss added to a resonator |
| `tdgl_common.py` | Builds the simulated films (strips with or without a lattice of holes) and finds vortices in the results |
| `figstyle.py` | Shared figure style |
| `tests.py` | Eleven self-tests of `physics.py` against exact results |

---

## 8. Where the numbers come from

Each value in `params.py` is marked with its source.

**Measured tantalum film values** come from sample D2 in Table I of
Bahrami et al., *Vortex Motion Induced Losses in Tantalum Resonators*,
arXiv:2503.03168 (2025), https://doi.org/10.48550/arXiv.2503.03168:
critical temperature 4.39 K, upper critical field 0.132 T,
coherence length 49.9 nm, mean free path 142.3 nm,
normal resistivity 0.55 microohm cm, and activation temperature 0.37 K.

**Derived values** follow from these with standard formulas, for example the
vortex drag (Bardeen-Stephen formula) 4.97e-8 kg/(m s).

**Assumed values** are not reported for this film or depend on the device.
Each is scanned in the article to show how much it matters:

| Assumed value | Used | Scanned |
|---|---|---|
| Penetration depth | 50 nm | 35 to 100 nm |
| Film thickness | 150 nm | 50 to 300 nm |
| Sensor height above the film | 50 nm | 20 to 150 nm |
| Sensor temperature | 2 K | 0.6 to 4.2 K |
| Intrinsic sensor relaxation rate | 1000 per second | reference line only |
| Resonator | 5 GHz, 50 ohm, 10 micrometre wide | fixed example |

---

## 9. Built-in checks

- **`tests.py`** compares the formulas with exact results: flux quantization,
  the thin-film (Pearl) and far-field limits, analytic derivatives, the
  Johnson-noise formula for a normal metal, the Mattis-Bardeen limits, and the
  noise-to-loss identity.
- **`03_langevin_fdt.py`** checks the key relations with independent random
  simulations: absorbed power versus noise (agreement within a few percent),
  the sensor relaxation formula, and the dephasing formula.
- **`06_inference.py`** checks that the vortex properties and the loss can be
  recovered from noisy fake data.

---

## 10. Notes on the calculations

- **Superconductor simulations.** `01` and `02` use pyTDGL, which solves the
  time-dependent Ginzburg-Landau equations. Lengths are in units of the
  coherence length (49.9 nm) and time in units of 0.571 ps; the scripts convert
  everything to micrometres, seconds and watts before saving.
- **Cooling protocol.** A clean simulated film at fixed temperature keeps flux
  out, so cooling through the critical temperature is mimicked with a short
  30 mT pulse followed by relaxation at the target field. A weaker pulse gives
  a different count of trapped flux (this is reported in the supplementary
  material), which is why the article recommends counting vortices in each
  real device.
- **Measured power.** The absorbed power is the average of current times
  voltage over two full drive periods, so the stored (reactive) part cancels.
  All comparisons are differences between runs of the same film, which removes
  the constant loss at the current contacts of the simulation.
- **Storage.** The Brownian-motion noise spectra are stored in 300 averaged
  frequency bins to keep the files small; the comparisons in the article use
  the full-resolution spectra computed during the run.

---

## 11. Version history

| Version | Date | Article | Data |
|---|---|---|---|
| **v2.0.1** (this version) | 29 Sep 2026 | Same as v2.0.0; clearer guide and script order, results unchanged | https://doi.org/10.5281/zenodo.23022201 |
| v2.0.0 | 29 Sep 2026 | Listening to Vortex Noise ... (current manuscript) | https://doi.org/10.5281/zenodo.23022201 |
| v1 | Jul 2026 | Earlier end-to-end imaging manuscript | https://doi.org/10.5281/zenodo.21498663 |

Details are in [CHANGELOG.md](CHANGELOG.md).

---

## 12. How to cite

Please cite the article and the data archive. GitHub also shows a
**"Cite this repository"** button in the right-hand column, which reads
`CITATION.cff`.

> T. M. Mahim, M. M. Rahman, and A.S.M. Mohsin, "Listening to Vortex Noise:
> Diamond Spin Sensors Reveal Which Trapped Vortices Cause Microwave Loss in
> Tantalum Qubit Films" (2026).
>
> Data: T. M. Mahim, M. M. Rahman, and A.S.M. Mohsin, Simulation database and
> model outputs, Zenodo, version 2.0 (2026), https://doi.org/10.5281/zenodo.23022201

---

## 13. License and contact

Code: Apache License 2.0 (see `LICENSE`). Data on Zenodo: CC BY 4.0.

Questions and bug reports: please open an issue on this repository, or
contact Tanvir M. Mahim, BRAC University (tanvir.mahim@bracu.ac.bd).
