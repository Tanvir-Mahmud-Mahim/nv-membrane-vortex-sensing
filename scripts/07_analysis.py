"""Consolidated analysis: reads all simulation outputs, computes the derived
quantities quoted in the manuscript, and writes data/results.json."""
import sys, json, glob
sys.path.insert(0, ".")
import numpy as np
import params as P

exec(open("scripts/03_nv_observables.py").read().split('if __name__')[0])

R = {}

# ---- material anchors ----
R["xi_nm"] = P.XI_0_NM
R["eta_BS"] = P.ETA_BS
R["Lambda_pearl_nm"] = LAMBDA_PEARL_UM * 1e3
R["tau0_ps"] = P.TAU_0_S * 1e12
R["B_phi_mT"] = P.B_PHI_MT
R["z_nv_surface_nm"] = Z_SURF_UM * 1e3

# ---- field-cool census (unit-corrected loader, winding occupancy) ----
from tdgl_common import load_fieldcool
census = {}
for f in sorted(glob.glob("data/fieldcool_*.npz")):
    tag = f.split("fieldcool_")[1][:-4]
    d = load_fieldcool(tag)
    occ = d["occ_winding"]
    census[tag] = dict(
        B_mT=float(d["B_mT"]),
        n_holes=int(len(d["hole_centers"])),
        occupied=int(np.sum(occ > 0)) if occ.size else 0,
        quanta_trapped=int(np.sum(occ)) if occ.size else 0,
        interstitial=int(len(d["vortex_pos"])),
        expected_quanta=float(d["B_mT"] * 1e-3 * (P.FILM_W * P.FILM_L * 1e-12)
                              / P.PHI0),
    )
R["census"] = census

# ---- IV / depinning ----
try:
    iv = json.load(open("data/pinning_iv.json"))
    I = np.array(iv["I_uA"])
    dv_bare = np.array(iv["bare_B8.27"]) - np.array(iv["bare_B0.00"])
    dv_anti = np.array(iv["anti15_B8.27"]) - np.array(iv["anti15_B0.00"])
    R["iv"] = dict(
        I_uA=I.tolist(), dv_bare=dv_bare.tolist(), dv_anti=dv_anti.tolist(),
        suppression_at_max=float(dv_bare[-1] / dv_anti[-1]),
        J_d_lower_A_per_m=float(I[-1] * 1e-6 / (P.FILM_W * 1e-6)),
    )
    J_d = R["iv"]["J_d_lower_A_per_m"]
except Exception as e:
    print("iv missing:", e); J_d = 250.0

# ---- microwave drive ----
try:
    mw = json.load(open("data/microwave_drive.json"))
    R["microwave"] = mw
    R["mw_suppression"] = mw["bare"]["P_absavg_red"] / mw["anti15"]["P_absavg_red"]
except Exception as e:
    print("mw missing:", e)

# ---- Pearl peaks (sheet-referenced heights) ----
peaks = {}
for so in [10, 25, 50, 100]:
    peaks[so] = float(pearl_bz(np.array([0.0]), z_sheet_um(so))[0])
R["Bpk_mT"] = peaks
R["odmr_shift_25nm_MHz"] = float(GAMMA_NV * PROJ * peaks[25] * 1e-3 / 1e6)

# ---- sensitivity ----
sens = dict(nv_observables(np.zeros((2, 2))))
R["sensitivity_uT_sqrtHz"] = sens
# SNR=1 pixel time at 25 nm standoff, membrane
sigma_needed = peaks[25] * 1e3  # uT signal
R["t_snr1_ns_membrane"] = float((sens["membrane"] / sigma_needed) ** 2 * 1e9)
# full frame at 0.1 ms pixels
R["frame_s_101px_0p1ms"] = float(101 * 101 * 1e-4)

# ---- two-channel noise spectroscopy ----
R["noise"] = {k: (v if not isinstance(v, dict) else v)
              for k, v in relaxometry(J_d).items()}
R["T2phi_contrast"] = (R["noise"]["natural"]["Gammaphi_per_s"]
                       / R["noise"]["antidot"]["Gammaphi_per_s"])

# ---- reconstruction summary ----
try:
    rec = json.load(open("data/reconstruction.json"))
    best = [r for r in rec["scan"]
            if r["standoff_nm"] == 25 and r["t_px_ms"] == 1.0
            and r["mode"] == "membrane"][0]
    R["recon_25nm_1ms"] = best
    R["occupancy_curve"] = rec["occupancy"]
except Exception as e:
    print("recon missing:", e)

# ---- covariance ----
try:
    cov = json.load(open("data/covariance.json"))
    R["covariance"] = cov
except Exception as e:
    print("cov missing:", e)

with open("data/results.json", "w") as f:
    json.dump(R, f, indent=2, default=float)
print(json.dumps(R, indent=1, default=float)[:3500])
