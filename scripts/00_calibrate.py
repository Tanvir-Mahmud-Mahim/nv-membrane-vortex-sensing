"""Calibration: find the current scale (onset of dissipation) for the bare
strip at zero field and at B_phi, so the staircase ranges are well placed."""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
from tdgl_common import make_device, staircase_iv
import params as P

t0 = time.time()
dev, _ = make_device("cal", hole_diam=0.0)
print("sites:", dev.mesh.sites.shape[0], flush=True)

out = {}
for B in [0.0, P.B_PHI_MT]:
    I = np.linspace(20, 400, 12)
    I_, V, sol = staircase_iv(dev, B, I, hold=30, skip=10)
    out[f"B={B:.2f}"] = {"I_uA": I_.tolist(), "V": V.tolist()}
    print(f"B={B:.2f} mT:", np.round(V, 4).tolist(), flush=True)

with open("data/calibration.json", "w") as f:
    json.dump(out, f, indent=2)
print("elapsed %.1f s" % (time.time() - t0))
