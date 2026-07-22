"""Covariance magnetometry of vortex dynamics (protocol of Rovny et al.,
Science 378, 1301 (2022), applied to vortex hopping).

A weakly pinned vortex performs thermally activated telegraph hopping between
two pinning sites. Two NV centers in the same membrane sample the resulting
field. Single-NV averaging washes the jumps out; the two-point covariance
isolates the correlated hop signal and yields the hop rate.

Also computes the single-NV Langevin noise spectrum S_B(f) for pinned vs
weakly pinned vortices (the relaxometry contrast of script 03, now spectrally
resolved from kHz to 10 GHz).
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import params as P

exec(open("scripts/03_nv_observables.py").read().split('if __name__')[0])

rng = np.random.default_rng(3)
out = {}

# ---- geometry: vortex hops between two sites 100 nm apart ----
z = z_sheet_um(25)  # um, sheet-referenced
kern_r = np.linspace(0, 1.2, 500)
kern_b = pearl_bz(kern_r, z)
def kern(r): return np.interp(r, kern_r, kern_b)

siteA = np.array([-0.05, 0.0]); siteB = np.array([0.05, 0.0])
nv1 = np.array([-0.08, 0.02]); nv2 = np.array([0.10, -0.03])
B1 = [kern(np.linalg.norm(nv1 - s)) for s in (siteA, siteB)]
B2 = [kern(np.linalg.norm(nv2 - s)) for s in (siteA, siteB)]
out["dB1_mT"] = float(B1[0] - B1[1]); out["dB2_mT"] = float(B2[0] - B2[1])

# ---- telegraph process ----
rate = 2e3  # Hz hop rate
T = 2.0     # s
n = 200000
dt = T / n
state = np.zeros(n, dtype=int)
s = 0
p = rate * dt
for i in range(1, n):
    if rng.random() < p:
        s = 1 - s
    state[i] = s
b1 = np.where(state == 0, B1[0], B1[1])
b2 = np.where(state == 0, B2[0], B2[1])
# NV phase readout shots: Ramsey phase ~ gamma*B*tau with photon shot noise
tau = 10e-6
sig1 = GAMMA_NV * PROJ * b1 * 1e-3 * tau  # radians-ish scaled
sig2 = GAMMA_NV * PROJ * b2 * 1e-3 * tau
shot = 8.0 * np.std(sig1 - sig1.mean())   # SNR_single ~ 0.12 per shot
m1 = sig1 + rng.normal(0, shot, n)
m2 = sig2 + rng.normal(0, shot, n)
cov_t = []
lags = np.arange(0, 400, 4)
d1 = m1 - m1.mean(); d2 = m2 - m2.mean()
for L in lags:
    if L == 0:
        cov_t.append(np.mean(d1 * d2))
    else:
        cov_t.append(np.mean(d1[:-L] * d2[L:]))
cov_t = np.array(cov_t)
theory = np.mean((sig1 - sig1.mean()) * (sig2 - sig2.mean())) * np.exp(-2 * rate * lags * dt)
out["hop_rate_Hz"] = rate
out["snr_single_shot"] = float(np.std(sig1 - sig1.mean()) / shot)
np.savez("data/covariance.npz", lags_s=lags * dt, cov=cov_t, theory=theory,
         b1_mT=b1[:2000], t_short=np.arange(2000) * dt)

# ---- Langevin spectra, pinned vs natural pinning ----
eta_v = P.ETA_BS * (P.THICKNESS_NM * 1e-9)
J_d = json.load(open("data/nv_observables.json"))["J_d_A_per_m"]
F_dep = J_d * P.PHI0
xi_m = P.XI_0_NM * 1e-9
gradB = (kern(0.0) * 1e-3) / ((z * 1e-6) + xi_m)
f = np.logspace(3, 10.3, 400)
w = 2 * np.pi * f
spec = {}
for tag, kp_frac in [("antidot", 1.0), ("natural", 0.02)]:
    kp = kp_frac * F_dep / xi_m
    Sx = 4 * KB * 2.1 * eta_v / (kp ** 2 + (w * eta_v) ** 2)
    SB = gradB ** 2 * Sx
    spec[tag] = SB
    out[f"f_p_{tag}_GHz"] = float(kp / eta_v / 2 / np.pi / 1e9)
np.savez("data/langevin_spectra.npz", f_Hz=f, **{k: v for k, v in spec.items()})

with open("data/covariance.json", "w") as f_:
    json.dump(out, f_, indent=2, default=float)
print(json.dumps(out, indent=1, default=float))
