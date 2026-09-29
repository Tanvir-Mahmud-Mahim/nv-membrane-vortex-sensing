"""NV observables of one interstitial vortex versus pinning stiffness,
temperature, sensor height, penetration depth and film thickness.

Outputs data/observables.json and data/observables.npz used by the figures.
Three observables are computed for a vortex of drag eta_v and pinning
stiffness kp (harmonic Langevin model), for an NV at height h:
  (1) 1/T1 excess, maximum over lateral NV position (GHz noise),
  (2) Hahn-echo decay rate from the Gaussian (Anderson-Kubo) formula for
      Ornstein-Uhlenbeck field noise, maximum over lateral position,
  (3) thermal blur of the time-averaged field image, sigma_x = sqrt(kT/kp),
      quantified as the fractional drop of the image peak.
Per-vortex resonator loss at F_RES follows from the same (eta_v, kp).
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import brentq
import params as P
import physics as ph

lam, d, h = P.LAMBDA_0_NM * 1e-9, P.THICKNESS_NM * 1e-9, P.NV_HEIGHT_NM * 1e-9
T = P.T_MEAS_K
eta = P.ETA_V
wnv = 2 * np.pi * P.F_NV_HZ
wres = 2 * np.pi * P.F_RES_HZ
n = ph.nv_axis(P.NV_THETA_DEG, P.NV_PHI_DEG)
kT = P.KB * T
out = {}

# ---------------- geometry of one vortex at the NV plane ----------------
x = np.linspace(-500e-9, 500e-9, 201)
X, Y = np.meshgrid(x, x)
B, G = ph.field_and_gradient_maps(X, Y, h, lam, d)
gperp, gpar = ph.geometric_factors(G, n)
Bpar = np.einsum("i,i...->...", n, B)
out["gperp_max"] = float(gperp.max())
out["gpar_max"] = float(gpar.max())
out["Bpar_max_mT"] = float(Bpar.max() * 1e3)


def echo_rate(sigB2, tc, gamma=P.GAMMA_E):
    """1/t where the Hahn-echo attenuation exponent of OU noise reaches 1:
    chi(t) = g^2 s^2 tc^2 [t/tc - 3 + 4 exp(-t/2tc) - exp(-t/tc)]."""
    a = gamma**2 * sigB2 * tc**2
    f = lambda t: a * (t / tc - 3 + 4 * np.exp(-t / (2 * tc)) - np.exp(-t / tc)) - 1
    hi = tc
    while f(hi) < 0:
        hi *= 4
        if hi > 1e3:
            return 0.0
    return 1.0 / brentq(f, 1e-18, hi)


# Gauss-Hermite quadrature for the thermal blur of the static image
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(24)
gh_w = gh_w / gh_w.sum()
rr = np.linspace(0, 1.5e-6, 3001)
Bz_r, Br_r, _, _ = ph.vortex_radial(rr, h, lam, d)


def blurred_peak(sig):
    """Peak of the time-averaged B.n image (maximised over position)."""
    xs = np.linspace(-150e-9, 150e-9, 121)[:, None, None]
    dx = xs - sig * gh_x[None, :, None]
    dy = -sig * gh_x[None, None, :] + 0 * dx
    r = np.hypot(dx, dy)
    bz = np.interp(r, rr, Bz_r)
    br = np.interp(r, rr, Br_r)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(r > 0, dx / r, 0.0)
        cy = np.where(r > 0, dy / r, 0.0)
    b = n[0] * br * cx + n[1] * br * cy + n[2] * bz
    W = gh_w[:, None] * gh_w[None, :]
    return float(np.max(np.sum(b * W, axis=(1, 2))))


peak0 = blurred_peak(1e-12)
kps = P.KP_SCAN
rows = []
for kp in kps:
    Snv = ph.Sx_harmonic(wnv, T, eta, kp)
    S0 = ph.Sx_harmonic(0.0, T, eta, kp)
    r1 = 3 * P.GAMMA_E**2 / 4 * Snv * out["gperp_max"]
    tc = eta / kp
    sig = np.sqrt(kT / kp)
    # echo: evaluated with the linearised field variance where it is valid
    # (sigma_x well below the field length scale h + lambda)
    sigB2 = out["gpar_max"] * kT / kp
    er = echo_rate(sigB2, tc) if sig < 0.3 * (h + lam) else np.nan
    blur = 1 - blurred_peak(sig) / peak0 if sig > 0.3e-9 else 0.0
    fp = kp / (2 * np.pi * eta)
    iq_c = ph.per_vortex_inverse_Q(wres, eta, kp, 2 / np.pi, 1.0,
                                   P.CPW_W_UM * 1e-6, P.CPW_Z0)
    iq_e = ph.per_vortex_inverse_Q(wres, eta, kp,
                                   ph.cpw_current_factor(4.8e-6, P.CPW_W_UM * 1e-6),
                                   1.0, P.CPW_W_UM * 1e-6, P.CPW_Z0)
    rows.append(dict(kp=kp, fp_Hz=fp, invT1=r1, echo_rate=er,
                     blur_frac=blur, sigma_x_nm=sig * 1e9, tau_c=tc,
                     invQ_center=iq_c, invQ_edge=iq_e))
out["vs_kp"] = rows
out["kp_tact"] = P.KP_TACT
out["fp_tact_Hz"] = P.KP_TACT / (2 * np.pi * eta)
out["kp_equal_fres"] = wres * eta          # f_p = f_res
out["kp_equal_fnv"] = wnv * eta

# ---------------- quasiparticle background vs temperature ----------------
# lambda = LAMBDA_0_NM is taken at the measurement temperature T_MEAS; its
# temperature dependence follows the Mattis-Bardeen sigma2 (lambda ~ sigma2^-1/2).
s1m, s2m = ph.mattis_bardeen(T, P.TC_K, P.F_NV_HZ)
Ts = np.linspace(0.6, 4.2, 37)
qp, vort, lamTs = [], [], []
for Tq in Ts:
    s1, s2 = ph.mattis_bardeen(Tq, P.TC_K, P.F_NV_HZ)
    lamT = lam * np.sqrt(s2m / s2)
    sig = s1 * P.SIGMA_N + 1j / (P.MU0 * wnv * lamT**2)
    Szz = ph.sheet_noise_Szz(h, Tq, wnv, sig, d)
    qp.append(ph.nv_T1_from_sheet(Szz, n))
    # vortex (viscous limit, eta taken T-independent; see text), with the
    # stray-field geometry recomputed for lambda(T)
    _, GT = ph.field_and_gradient_maps(X, Y, h, lamT, d)
    gpT, _ = ph.geometric_factors(GT, n)
    vort.append(3 * P.GAMMA_E**2 / 4 * 4 * P.KB * Tq / (eta * wnv**2) * gpT.max())
    lamTs.append(lamT * 1e9)
out["lambda_T_nm"] = lamTs
out["T_K"] = Ts.tolist()
out["qp_invT1"] = qp
out["vortex_invT1_viscous"] = vort
i2 = int(np.argmin(abs(Ts - T)))
out["qp_invT1_at_Tmeas"] = qp[i2]
out["vortex_invT1_at_Tmeas"] = vort[i2]
out["T_of_vortex_max"] = float(Ts[int(np.argmax(vort))])
out["sigma1_over_sn_Tmeas"] = s1m
# normal-state reference just above Tc (same slab model)
Szz_n = ph.sheet_noise_Szz(h, P.TC_K, wnv, P.SIGMA_N + 0j, d)
out["normal_invT1_at_Tc"] = ph.nv_T1_from_sheet(Szz_n, n)
Szz_n2 = ph.sheet_noise_Szz(h, T, wnv, P.SIGMA_N + 0j, d)
out["normal_invT1_at_Tmeas"] = ph.nv_T1_from_sheet(Szz_n2, n)

# ---------------- height, lambda and thickness sensitivity ----------------
sens = {"h_nm": [], "lam_nm": [], "d_nm": []}
def peak_rate(hh, ll, dd):
    Bq, Gq = ph.field_and_gradient_maps(X, Y, hh, ll, dd)
    gp, _ = ph.geometric_factors(Gq, n)
    return 3 * P.GAMMA_E**2 / 4 * 4 * kT / (eta * wnv**2) * gp.max()
for hh in P.NV_HEIGHT_SCAN_NM:
    sens["h_nm"].append([hh, peak_rate(hh * 1e-9, lam, d)])
for ll in P.LAMBDA_SCAN_NM:
    sens["lam_nm"].append([ll, peak_rate(h, ll * 1e-9, d)])
for dd in (50.0, 100.0, 150.0, 200.0, 300.0):
    sens["d_nm"].append([dd, peak_rate(h, lam, dd * 1e-9)])
out["sensitivity"] = sens

# NV axis orientation: peak rate for the four (100) NV families and for an
# NV along the film normal
orient = {}
for name, th_, ph_ in [("normal", 0.0, 0.0), ("tilted_phi0", P.NV_THETA_DEG, 0.0),
                       ("tilted_phi45", P.NV_THETA_DEG, 45.0)]:
    nn = ph.nv_axis(th_, ph_)
    gp, gq = ph.geometric_factors(G, nn)
    orient[name] = dict(gperp_max=float(gp.max()), gpar_max=float(gq.max()))
out["orientation"] = orient

np.savez("data/observables_maps.npz", x=x, Bpar=Bpar, gperp=gperp, gpar=gpar,
         Bz=B[2])
with open("data/observables.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ["gperp_max", "gpar_max", "Bpar_max_mT",
      "kp_tact", "fp_tact_Hz", "kp_equal_fres", "qp_invT1_at_Tmeas",
      "vortex_invT1_at_Tmeas", "sigma1_over_sn_Tmeas", "normal_invT1_at_Tc",
      "normal_invT1_at_Tmeas"]},
      indent=1))
for r in rows[::6]:
    print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items()})
print(sens)
print(orient)
