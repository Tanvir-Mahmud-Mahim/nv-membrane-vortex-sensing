"""Material, device and sensor parameters, with provenance.

Every number below is either (i) taken from a cited measurement, (ii) derived
from those numbers with a stated formula, or (iii) marked ASSUMED and varied
in a sensitivity analysis (scripts/07_sensitivity.py).

Primary material source
  [B25] Bahrami et al., arXiv:2503.03168, Table I, alpha-Ta film "D2" on
        sapphire (type-A, clean limit):
        Tc = 4.39 K, Hc2(0) = 0.132 T, xi = 49.9 nm, l = 142.3 nm,
        RRR = 65.1, rho_n(5 K) = 0.55 uOhm cm, eta = 4.99e-8 kg/(m s)
        (eta computed in [B25] from eta = Phi0^2 / (2 pi xi^2 rho_n)),
        activation temperature of the loss T_act = 0.37 K.
  [M25] Marcaud et al., Commun. Mater. 6, 182 (2025): room-temperature
        grown Ta films; kinetic-inductance data used only as a consistency
        check of the penetration-depth range.
"""
import json
import os
import numpy as np

os.makedirs("data", exist_ok=True)   # all scripts write their outputs here

# ---------------- constants (CODATA) ----------------
PHI0 = 2.067833848e-15    # Wb
MU0 = 4e-7 * np.pi        # T m / A (to 1e-10 relative)
KB = 1.380649e-23         # J/K
HBAR = 1.054571817e-34    # J s
GAMMA_E = 1.76085963e11   # rad/(s T), electron gyromagnetic ratio (NV)

# ---------------- tantalum film, [B25] sample D2 ----------------
TC_K = 4.39
HC2_0_T = 0.132
XI_0_NM = 49.9
MFP_NM = 142.3
RHO_N_OHM_M = 0.55e-8
ETA_B25 = 4.99e-8                     # kg/(m s), as tabulated in [B25]
T_ACT_K = 0.37                        # K, [B25] Table I
# Bardeen-Stephen drag recomputed from the tabulated xi and rho_n
ETA_BS = PHI0**2 / (2 * np.pi * (XI_0_NM * 1e-9) ** 2 * RHO_N_OHM_M)

# ASSUMED: London penetration depth. [B25] does not report lambda. Estimates
# from the D2 data span ~35-65 nm (dirty-limit BCS with rho_n and Delta,
# clean-limit interpolation); we take 50 nm and scan 35-100 nm.
LAMBDA_0_NM = 50.0                    # at T_MEAS_K (2 K)
LAMBDA_SCAN_NM = (35.0, 50.0, 70.0, 100.0)
# ASSUMED: film thickness ([B25] does not state it). The stray field is
# insensitive to d once d > 2 lambda (checked in 07_sensitivity.py).
THICKNESS_NM = 150.0
GAMMA_TDGL = 10.0                     # pyTDGL inelastic-scattering parameter

# ---------------- derived vortex mechanics ----------------
ETA_V = ETA_BS * THICKNESS_NM * 1e-9  # kg/s, drag of one vortex line (rigid)
# Pinning stiffness is NOT known for Ta films. Scan range and one anchor:
KP_SCAN = np.logspace(-9, -3, 61)     # N/m
# Anchor from [B25] T_act if read as a barrier U = kB*T_act acting over a
# length xi: kp ~ 2U/xi^2 (order of magnitude only).
KP_TACT = 2 * KB * T_ACT_K / (XI_0_NM * 1e-9) ** 2

# ---------------- TDGL units ----------------
SIGMA_N = 1.0 / RHO_N_OHM_M
TAU_0_S = MU0 * SIGMA_N * (LAMBDA_0_NM * 1e-9) ** 2   # pyTDGL time unit

# ---------------- geometry (um) ----------------
PITCH = 0.5                           # antidot pitch
B_PHI_MT = PHI0 / (PITCH * 1e-6) ** 2 * 1e3            # matching field 8.27 mT
MESH_EDGE = 0.030                     # um (0.6 xi)
FILM_W = 1.6                          # um, TDGL transport strip width
FILM_L = 1.6                          # um, TDGL transport strip length

# ---------------- NV sensor (ASSUMED geometry, scanned) ----------------
NV_HEIGHT_NM = 50.0                   # NV height above the Ta surface
NV_HEIGHT_SCAN_NM = (20.0, 35.0, 50.0, 75.0, 100.0, 150.0)
NV_THETA_DEG = 54.7356                # NV axis tilt, (100) diamond
NV_PHI_DEG = 0.0                      # in-plane azimuth of the NV axis
F_NV_HZ = 2.87e9                      # zero-field NV transition
T_MEAS_K = 2.0                        # NV measurement temperature
NV_INTRINSIC_RATE = 1e3               # 1/s, ASSUMED intrinsic 1/T1 floor

# ---------------- resonator (ASSUMED, typical CPW) ----------------
F_RES_HZ = 5.0e9
CPW_W_UM = 10.0                       # center-conductor width
CPW_Z0 = 50.0                         # Ohm
N_PHOTON_REF = 1.0                    # loss is linear, reported per photon


def dump(path="data/params.json"):
    d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in globals().items()
         if k.isupper() and isinstance(v, (int, float, tuple, np.ndarray))}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return d


if __name__ == "__main__":
    print(f"eta_BS (recomputed) = {ETA_BS:.3e} kg/(m s)  [B25 table: 4.99e-8]")
    print(f"eta_v = eta*d       = {ETA_V:.3e} kg/s")
    print(f"kp(T_act anchor)    = {KP_TACT:.2e} N/m")
    print(f"B_phi               = {B_PHI_MT:.3f} mT")
    print(f"tau0                = {TAU_0_S*1e12:.3f} ps")
