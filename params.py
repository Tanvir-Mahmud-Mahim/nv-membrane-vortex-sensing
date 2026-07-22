"""Shared material and device parameters for the vortex-pinning co-design study.

All material anchors are taken from published, open-access measurements:
  [Bahrami et al., arXiv:2503.03168] type-A (clean-limit) alpha-Ta films on sapphire:
      Tc ~ 4.3 K, Hc2(0) ~ 0.10-0.17 T, xi ~ 44-57 nm, rho_n(5 K) ~ 0.55-0.87 uOhm cm,
      eta (viscous drag) ~ 2.9-5.3e-8 kg/(m s), mean free path l ~ 110-142 nm (clean limit).
  [Hedrick et al., arXiv:2603.13183] Al resonator surface loss tangents:
      tan(delta_s) = (1.77-5.43)e-3 for a ~3 nm dielectric layer (fab dependent).
  [Pakpour-Tabrizi et al., Adv. Opt. Mater. 2026, 14, e03864] diamond micromembranes:
      shallow NV centers at ~4-20 nm depth, Hahn-echo T2 up to ~100 us, pick-and-place
      integration on arbitrary targets.
"""
import json
import numpy as np

PHI0 = 2.067833848e-15  # Wb, flux quantum
MU0 = 4e-7 * np.pi

# ---------------- Tantalum film (type-A, clean limit) ----------------
TC_K = 4.3               # K
HC2_0_T = 0.132          # T (mid of type-A range 0.108-0.166)
XI_0_NM = float(np.sqrt(PHI0 / (2 * np.pi * HC2_0_T)) * 1e9)  # ~50 nm
RHO_N_OHM_M = 0.65e-8    # 0.65 uOhm cm, type-A average
LAMBDA_0_NM = 100.0      # nm, alpha-Ta thin-film London depth (85-120 nm reported)
THICKNESS_NM = 150.0     # nm, film thickness
GAMMA_TDGL = 10.0        # inelastic scattering parameter (gapful dynamics)

# Bardeen-Stephen viscous drag coefficient (validated against measured eta)
ETA_BS = PHI0**2 / (2 * np.pi * (XI_0_NM * 1e-9) ** 2 * RHO_N_OHM_M)  # kg/(m s)

# TDGL characteristic scales
SIGMA_N = 1.0 / RHO_N_OHM_M
TAU_0_S = MU0 * SIGMA_N * (LAMBDA_0_NM * 1e-9) ** 2   # ~1.9 ps
# TDGL time unit used by pyTDGL: tau = mu0 * sigma * lambda^2

# ---------------- Simulation geometry (units: um) ----------------
FILM_W = 1.6             # um, transport strip width  (32 xi)
FILM_L = 1.6             # um, transport strip length
PITCH = 0.5              # um, antidot lattice pitch -> B_phi = PHI0/PITCH^2
B_PHI_MT = PHI0 / (PITCH * 1e-6) ** 2 * 1e3  # mT, matching field (~8.27 mT)
MESH_EDGE = 0.030        # um, max mesh edge (~0.6 xi)

# ---------------- Surface-loss (TLS) model anchors ----------------
TAN_DELTA_S = 3.19e-3    # wet-etch surface loss tangent (Al paper, Ta-like value)
T_DIEL_NM = 3.0          # nm, assumed surface dielectric thickness
P_EDGE_CPW = 2.2e-4      # baseline edge participation of a 6-um-gap CPW (Wenner 2011 scale)
CPW_EDGE_UM_PER_UM2 = 2.0 / 60.0  # two gap edges per ~60 um transverse cell

# ---------------- NV magnetometry anchors ----------------
NV_DEPTH_NM = 6.0        # nm, average implantation depth (membrane paper)
MEMBRANE_STANDOFF_NM = 25.0  # nm, membrane-target gap after pick-and-place
NV_T2_US = 100.0         # us, Hahn-echo T2 ceiling for shallow NV
NV_ETA_UT_SQHZ = 1.0     # uT/sqrt(Hz), DC (relaxometry-limited) sensitivity scale


def dump(path="data/params.json"):
    d = {k: v for k, v in globals().items()
         if k.isupper() and isinstance(v, (int, float))}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return d


if __name__ == "__main__":
    print(f"xi(0)        = {XI_0_NM:.1f} nm")
    print(f"B_phi(0.5um) = {B_PHI_MT:.2f} mT")
    print(f"eta_BS       = {ETA_BS:.2e} kg/(m s)  [measured type-A: 2.9-5.3e-8]")
    print(f"tau_0        = {TAU_0_S*1e12:.2f} ps")
