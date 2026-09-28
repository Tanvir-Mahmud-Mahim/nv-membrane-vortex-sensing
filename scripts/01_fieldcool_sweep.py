"""Field-cooled vortex census versus cooling field (pyTDGL).

A clean mesoscopic film at fixed temperature keeps a Bean-Livingston edge
barrier, so flux does not enter at millitesla fields. Field cooling through
Tc is emulated by a nucleation pulse at B_SEED (order parameter suppressed,
flux floods in) followed by relaxation at the target field; excess vortices
leave and the trapped state equilibrates.

For every case we record: hole fluxoid numbers (pyTDGL hole_fluxoid),
interstitial vortex cores (phase winding on mesh triangles; holes carry no
triangles), and the applied flux quanta B*A/Phi0.
Usage: python scripts/01_fieldcool_sweep.py [case ...]   (default: all)
"""
import sys, time, json, os
sys.path.insert(0, ".")
import numpy as np
import tdgl
from tdgl_common import make_device, vortices_on_mesh
import params as P

L = 2.0                                  # um, square film
B_SEED = 30.0                            # mT
RATIOS = [0.5, 1.0, 1.5, 2.0, 2.5]
CASES = {}
for r in RATIOS:
    CASES[f"anti15_r{r:.1f}"] = (0.15, r, B_SEED, 250)
    CASES[f"anti20_r{r:.1f}"] = (0.20, r, B_SEED, 250)
for r in RATIOS[:4]:
    CASES[f"bare_r{r:.1f}"] = (0.0, r, B_SEED, 250)
# protocol robustness: weaker pulse, longer relaxation
CASES["anti15_r1.5_seed20"] = (0.15, 1.5, 20.0, 250)
CASES["anti15_r1.5_long"] = (0.15, 1.5, B_SEED, 600)

os.makedirs("data/fieldcool_sweep", exist_ok=True)
todo = sys.argv[1:] or list(CASES)
for name in todo:
    D, r, bseed, trelax = CASES[name]
    B = r * P.B_PHI_MT
    t0 = time.time()
    dev, centers = make_device(name, hole_diam=D, pitch=P.PITCH if D else None,
                               w=L, l=L, transport=False)
    opt = lambda t: tdgl.SolverOptions(solve_time=t, field_units="mT",
                                       current_units="uA", save_every=1000,
                                       progress_interval=0)
    s1 = tdgl.solve(dev, opt(60), applied_vector_potential=bseed)
    sol = tdgl.solve(dev, opt(trelax), applied_vector_potential=B,
                     seed_solution=s1)
    psi = sol.tdgl_data.psi
    occ = [float(sum(sol.hole_fluxoid(f"hole{i}")).to("Phi_0").magnitude)
           for i in range(len(centers))]
    cen, wsg = vortices_on_mesh(dev.mesh.sites, dev.mesh.elements, psi)
    xi_um = P.XI_0_NM * 1e-3
    # integer fluxoid per hole: phase winding on a ring 40 nm outside the edge
    from scipy.interpolate import LinearNDInterpolator
    Ic = LinearNDInterpolator(dev.mesh.sites * xi_um, np.exp(1j * np.angle(psi)))
    th = np.linspace(0, 2 * np.pi, 361)
    wind = []
    for (hx, hy) in centers:
        rr = D / 2 + 0.04
        ring = Ic(hx + rr * np.cos(th), hy + rr * np.sin(th))
        wind.append(int(np.rint(abs(np.sum(np.angle(ring[1:] / ring[:-1])))
                                / (2 * np.pi))))
    out = dict(name=name, hole_diam_um=D, B_over_Bphi=r, B_mT=B,
               seed_mT=bseed, relax_tau=trelax, L_um=L,
               applied_quanta=B * 1e-3 * (L * 1e-6) ** 2 / P.PHI0,
               hole_centers_um=np.array(centers).reshape(-1, 2).tolist(),
               hole_fluxoid=occ, hole_quanta=wind,
               interstitial_um=(cen * xi_um).tolist(),
               interstitial_sign=wsg.tolist(), runtime_s=time.time() - t0)
    np.savez(f"data/fieldcool_sweep/{name}.npz", sites=dev.mesh.sites * xi_um,
             psi=psi, elements=dev.mesh.elements)
    with open(f"data/fieldcool_sweep/{name}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{name}: B={B:.2f} mT holes={sum(out['hole_quanta'])} quanta in "
          f"{len(centers)} holes, interstitial={len(cen)}, applied="
          f"{out['applied_quanta']:.1f} ({out['runtime_s']:.0f} s)", flush=True)
