"""Field and noise maps above TDGL field-cooled vortex configurations.

For each census (scripts/01_fieldcool_sweep.py) we superpose
  * the static stray field of every trapped flux quantum, projected on the
    NV axis: hole fluxoids use a London vortex spread over the hole disk
    (approximation), interstitial vortices use the exact finite-thickness
    London field;
  * the excess NV relaxation rate. By the fluctuation-dissipation theorem
    only dissipative degrees of freedom produce thermal field noise. An
    interstitial vortex has a normal core and viscous drag eta_v; a fluxoid
    held by a hole has no core, so its only noise is the film's own
    quasiparticle (Johnson) noise, which is included as a uniform background
    together with an assumed intrinsic NV floor.
Output: data/nv_maps_<case>.npz and data/nv_maps.json (census summary).
"""
import sys, json, glob, os
sys.path.insert(0, ".")
import numpy as np
import params as P
import physics as ph

lam, d, h = P.LAMBDA_0_NM * 1e-9, P.THICKNESS_NM * 1e-9, P.NV_HEIGHT_NM * 1e-9
T = P.T_MEAS_K
wnv = 2 * np.pi * P.F_NV_HZ
n = ph.nv_axis(P.NV_THETA_DEG, P.NV_PHI_DEG)
obs = json.load(open("data/observables.json"))   # from scripts/04_observables.py
QP = obs["qp_invT1_at_Tmeas"]
FLOOR = P.NV_INTRINSIC_RATE
KP_MAP = 1e-6          # N/m, representative weak pinning (T1 is kp-independent
                       # for kp << 2 pi f_NV eta_v, see observables.json)
Sx = ph.Sx_harmonic(wnv, T, P.ETA_V, KP_MAP)

rr = np.linspace(0, 3.2e-6, 4001)
RAD_V = ph.vortex_radial(rr, h, lam, d)
RAD_H = {D: ph.vortex_radial(rr, h, lam, d, form=ph.disk_form(D / 2 * 1e-6))
         for D in (0.15, 0.20)}


def maps_for(rad, X, Y):
    R = np.hypot(X, Y)
    Bz, Br, dBz, dBr = (np.interp(R, rr, rad[k]) for k in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(R > 0, X / np.where(R > 0, R, 1), 1.0)
        s = np.where(R > 0, Y / np.where(R > 0, R, 1), 0.0)
        Bro = np.where(R > 1e-12, Br / np.where(R > 1e-12, R, 1), dBr)
    B = np.stack([Br * c, Br * s, Bz])
    G = np.array([[dBr * c * c + Bro * s * s, (dBr - Bro) * c * s],
                  [(dBr - Bro) * c * s, dBr * s * s + Bro * c * c],
                  [dBz * c, dBz * s]])
    return B, G


summary = {}
files = sorted(glob.glob("data/fieldcool_sweep/*.json"))
for fn in files:
    cz = json.load(open(fn))
    name = cz["name"]
    L = cz["L_um"] * 1e-6
    x = np.linspace(-L / 2, L / 2, 201)
    X, Y = np.meshgrid(x, x)
    Bpar = np.zeros_like(X)
    rate = np.zeros_like(X)
    D = cz["hole_diam_um"]
    for (hx, hy), q in zip(cz["hole_centers_um"], cz["hole_quanta"]):
        if q == 0:
            continue
        B, _ = maps_for(RAD_H[D], X - hx * 1e-6, Y - hy * 1e-6)
        Bpar += q * np.einsum("i,i...->...", n, B)
    for (vx, vy), sg in zip(cz["interstitial_um"], cz["interstitial_sign"]):
        B, G = maps_for(RAD_V, X - vx * 1e-6, Y - vy * 1e-6)
        Bpar += sg * np.einsum("i,i...->...", n, B)
        gperp, _ = ph.geometric_factors(G, n)
        rate += 3 * P.GAMMA_E**2 / 4 * Sx * gperp
    total = rate + QP + FLOOR
    np.savez(f"data/nv_maps_{name}.npz", x=x, Bpar=Bpar, rate=total,
             rate_vortex=rate, holes=np.array(cz["hole_centers_um"]),
             hole_quanta=np.array(cz["hole_quanta"]),
             inter=np.array(cz["interstitial_um"]).reshape(-1, 2))
    # contrast at each trapped flux quantum position
    def at(px, py):
        i = np.argmin(abs(x - px * 1e-6)); j = np.argmin(abs(x - py * 1e-6))
        return float(total[j, i]), float(Bpar[j, i])
    hole_rates = [at(*c)[0] for c, q in zip(cz["hole_centers_um"], cz["hole_quanta"]) if q > 0]
    hole_B = [at(*c)[1] for c, q in zip(cz["hole_centers_um"], cz["hole_quanta"]) if q > 0]
    int_rates = [at(*v)[0] for v in cz["interstitial_um"]]
    summary[name] = dict(B_over_Bphi=cz["B_over_Bphi"], hole_diam_um=D,
                         applied_quanta=cz["applied_quanta"],
                         hole_quanta_total=int(sum(cz["hole_quanta"])),
                         holes_occupied=int(sum(q > 0 for q in cz["hole_quanta"])),
                         n_holes=len(cz["hole_quanta"]),
                         n_interstitial=len(cz["interstitial_um"]),
                         max_quanta_per_hole=int(max(cz["hole_quanta"] or [0])),
                         hole_rate_median=float(np.median(hole_rates)) if hole_rates else None,
                         hole_Bpar_median_mT=float(np.median(hole_B) * 1e3) if hole_B else None,
                         inter_rate_median=float(np.median(int_rates)) if int_rates else None,
                         background_rate=QP + FLOOR)
    print(name, {k: v for k, v in summary[name].items() if k not in ("hole_diam_um",)}, flush=True)

with open("data/nv_maps.json", "w") as f:
    json.dump(summary, f, indent=1)
print("DONE")
