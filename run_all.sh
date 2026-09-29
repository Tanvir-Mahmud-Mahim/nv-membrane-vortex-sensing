#!/bin/bash
# Full reproduction of every number and figure (about 4 hours on one CPU core).
# Run from this folder. Outputs go to data/ and figures/.
set -e
mkdir -p data
export OMP_NUM_THREADS=1
python3 tests.py | tee data/tests_output.txt
python3 params.py
python3 -c "import params; params.dump()"       # writes data/params.json
python3 scripts/01_fieldcool_sweep.py            # TDGL census (16 runs)
for c in anti15_B0 anti15_B8 anti15_B12 anti15_B8zfc bare_B0 bare_B4; do
  python3 scripts/02_tdgl_ac_loss.py $c 1000     # TDGL 5 GHz drive
done
python3 scripts/02_tdgl_ac_loss.py anti15_B12 500 # linearity check
python3 scripts/03_langevin_fdt.py               # Brownian-dynamics checks
python3 scripts/04_observables.py                # single-vortex observables
python3 scripts/05_nv_maps.py                    # field and noise maps
python3 scripts/06_inference.py                  # synthetic-measurement test
python3 scripts/07_figures.py abstract vortex maps tdgl checks readout supp
