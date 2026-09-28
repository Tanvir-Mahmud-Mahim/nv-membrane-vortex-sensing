"""TDGL microwave-loss test: do hole-trapped fluxoids dissipate?

For each case the transport strip is field cooled (nucleation pulse, then
relaxation at the target field), then driven by I(t) = I0 sin(2 pi f t) at
f = F_RES_HZ. The dissipated power is the cycle average of I(t) V(t) over an
integer number of periods (reactive, inductive power averages to zero).
The zero-field (Meissner) run of the same device is the baseline, so the
excess power belongs to the trapped flux. Usage:
    python scripts/02_tdgl_ac_loss.py <case> [I0_uA]
"""
import sys, time, json, os
sys.path.insert(0, ".")
import numpy as np
import tdgl
from tdgl_common import make_device, vortices_on_mesh
import params as P

CASES = {  # name: (hole diameter um, pitch um, field mT)
    "bare_B0": (0.0, None, 0.0),
    "bare_B4": (0.0, None, 0.5 * P.B_PHI_MT),
    "bare_B8": (0.0, None, 1.0 * P.B_PHI_MT),
    "anti15_B0": (0.15, 0.5, 0.0),
    "anti15_B8": (0.15, 0.5, 1.0 * P.B_PHI_MT),
    "anti15_B8zfc": (0.15, 0.5, 1.0 * P.B_PHI_MT),   # zero-field cooled control
    "anti15_B12": (0.15, 0.5, 1.5 * P.B_PHI_MT),
}
case = sys.argv[1]
I0 = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
D, pitch, B = CASES[case]
PERIOD = 1.0 / P.F_RES_HZ / P.TAU_0_S          # in tau0 units
N_SKIP, N_AVG = 0.5, 2                         # cycles
os.makedirs("data/ac_loss", exist_ok=True)

t0 = time.time()
dev, centers = make_device(case, hole_diam=D, pitch=pitch, transport=True)
def run(t, Bf, seed=None, **kw):
    o = tdgl.SolverOptions(solve_time=t, field_units="mT", current_units="uA",
                           save_every=kw.pop("save_every", 500),
                           progress_interval=0, **kw)
    return tdgl.solve(dev, o, applied_vector_potential=Bf,
                      seed_solution=seed, **{k: v for k, v in kw.items()})
if B > 0 and not case.endswith("zfc"):
    seed = run(60, 30.0)
    seed = run(250, B, seed)
elif B > 0:   # zero-field cooled: field applied to the superconducting state
    seed = run(100, 0.0)
    seed = run(250, B, seed)
else:
    seed = run(100, 0.0)

def drive(t):
    I = I0 * np.sin(2 * np.pi * t / PERIOD)
    return {"source": I, "drain": -I}

H5 = f"/tmp/claude-0/ac_{case}_{int(I0)}.h5"
if os.path.exists(H5):
    os.remove(H5)
o3 = tdgl.SolverOptions(solve_time=(N_SKIP + N_AVG) * PERIOD, field_units="mT",
                        current_units="uA", save_every=40, output_file=H5,
                        progress_interval=0)
sol = tdgl.solve(dev, o3, applied_vector_potential=B,
                 terminal_currents=drive, seed_solution=seed)
t = sol.dynamics.time
V = sol.dynamics.voltage()
It = I0 * np.sin(2 * np.pi * t / PERIOD)
m = (t >= N_SKIP * PERIOD) & (t <= (N_SKIP + N_AVG) * PERIOD)
Pred = np.trapezoid(It[m] * V[m], t[m]) / (t[m][-1] - t[m][0])   # uA * V0
# in-phase / quadrature voltage amplitudes
s = np.sin(2 * np.pi * t[m] / PERIOD); c = np.cos(2 * np.pi * t[m] / PERIOD)
T_ = t[m][-1] - t[m][0]
V_in = 2 * np.trapezoid(V[m] * s, t[m]) / T_
V_q = 2 * np.trapezoid(V[m] * c, t[m]) / T_
V0 = dev.V0().to("V").magnitude
P_W = Pred * 1e-6 * V0
# ---- diagnostics (stored, not used in the paper): time-averaged |J_n|^2 and
# J.E per mesh edge. Locally, J.E contains the divergence of the energy flux,
# so only the probe power P = <I V> is used for the dissipation results. ----
em = dev.mesh.edge_mesh
xi_um = P.XI_0_NM * 1e-3
area_e = em.edge_lengths * em.dual_edge_lengths / 2 * xi_um**2      # um^2
cen_e = em.centers * xi_um
import h5py
H5 = str(sol.path) if getattr(sol, "path", None) else H5   # pyTDGL may rename
acc = np.zeros(len(area_e)); accJE = np.zeros(len(area_e)); nacc = 0
ei, ej = em.edges[:, 0], em.edges[:, 1]
with h5py.File(H5, "r") as h5:
    for key in sorted(h5["data"].keys(), key=int):
        grp = h5["data"][key]
        tt = float(grp.attrs["time"]) if "time" in grp.attrs else -1.0
        if tt < N_SKIP * PERIOD:
            continue
        jn = np.asarray(grp["normal_current"])
        js = np.asarray(grp["supercurrent"])
        mu = np.asarray(grp["mu"])
        Ee = -(mu[ej] - mu[ei]) / em.edge_lengths     # no screening: dA/dt = 0
        acc += jn**2; accJE += (js + jn) * Ee; nacc += 1
jn2 = acc / max(nacc, 1)                     # <|J_n|^2> in K0^2 units
pJE = accJE / max(nacc, 1)                   # <J.E> per edge, reduced units
# ---- census of the driven state (end of run) ----
psi = sol.tdgl_data.psi
cen, wsgn = vortices_on_mesh(dev.mesh.sites, dev.mesh.elements, psi)
cen_um = cen * xi_um
inside = np.abs(cen_um[:, 0]) < 0.75          # drop terminal-edge artefacts
cen_um, wsgn = cen_um[inside], wsgn[inside]
cen = cen_um
xp = 0.4 * P.FILM_L
n_between = int(np.sum(np.abs(cen_um[:, 0]) < xp))
from scipy.interpolate import LinearNDInterpolator
Ic = LinearNDInterpolator(dev.mesh.sites * xi_um, np.exp(1j * np.angle(psi)))
th = np.linspace(0, 2 * np.pi, 361)
wind = []
for (hx, hy) in centers:
    ring = Ic(hx + (D / 2 + 0.04) * np.cos(th), hy + (D / 2 + 0.04) * np.sin(th))
    wind.append(int(np.rint(abs(np.sum(np.angle(ring[1:] / ring[:-1]))) / (2 * np.pi))))
occ = wind
def region_power(cx, cy, r, q=None):
    q = pJE if q is None else q
    m = np.hypot(cen_e[:, 0] - cx, cen_e[:, 1] - cy) < r
    return float(np.sum(q[m] * area_e[m]))
hole_P = [region_power(hx, hy, D / 2 + 0.1) for hx, hy in centers]
vort_P = [region_power(vx, vy, 0.15) for vx, vy in cen_um]
tot_P = float(np.sum(pJE * area_e))
probe_P = float(np.sum((pJE * area_e)[np.abs(cen_e[:, 0]) < 0.4 * P.FILM_L]))
out = dict(case=case, B_mT=B, I0_uA=I0, f_Hz=P.F_RES_HZ, period_tau=PERIOD,
           P_red=Pred, P_W=P_W, V0_V=V0, V_inphase_red=V_in, V_quad_red=V_q,
           n_interstitial=int(len(cen)), n_between_probes=n_between,
           hole_quanta=occ, interstitial_um=cen_um.tolist(),
           JE_total=tot_P, JE_between_probes=probe_P, joule_hole_regions=hole_P,
           joule_vortex_regions=vort_P, n_snapshots=nacc,
           runtime_s=time.time() - t0)
np.savez(f"data/ac_loss/{case}_I{int(I0)}.npz", t=t, V=V, I=It, jn2=jn2, pJE=pJE,
         edge_centers_um=cen_e, edge_area_um2=area_e, psi=psi,
         sites_um=dev.mesh.sites * xi_um, elements=dev.mesh.elements)
os.remove(H5)
with open(f"data/ac_loss/{case}_I{int(I0)}.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: out[k] for k in ["case", "I0_uA", "P_red", "P_W",
      "V_inphase_red", "V_quad_red", "n_interstitial", "n_between_probes",
      "hole_quanta", "JE_total", "JE_between_probes", "n_snapshots", "runtime_s"]}), flush=True)
