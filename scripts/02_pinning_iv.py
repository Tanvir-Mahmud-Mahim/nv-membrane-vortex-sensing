"""Transport contrast of the sensing target: flux-flow suppression by the
antidot lattice at the matching field. Staircase IV for bare vs antidot strip
at B = B_phi; the vortex voltage is V(B) - V(0) per device."""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
from tdgl_common import make_device, staircase_iv
import params as P

I = np.linspace(20, 400, 12)
out = {"I_uA": I.tolist()}
for tag, D, pitch in [("bare", 0.0, None), ("anti15", 0.15, 0.5)]:
    dev, centers = make_device(tag + "_iv", hole_diam=D, pitch=pitch,
                               transport=True)
    for B in [0.0, P.B_PHI_MT]:
        t0 = time.time()
        _, V, sol = staircase_iv(dev, B, I, hold=40, skip=15)
        out[f"{tag}_B{B:.2f}"] = V.tolist()
        print(f"{tag} B={B:.2f}: {np.round(V,5).tolist()} "
              f"({time.time()-t0:.0f} s)", flush=True)
        if D > 0 and B > 0:
            occ = [abs(sum(sol.hole_fluxoid(f"hole{i}")).to("Phi_0").magnitude)
                   for i in range(len(centers))]
            out[f"{tag}_B{B:.2f}_occ"] = occ
with open("data/pinning_iv.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE iv")
