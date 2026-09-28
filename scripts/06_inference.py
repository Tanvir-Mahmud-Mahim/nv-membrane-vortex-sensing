"""Synthetic-measurement test: can NV data recover the vortex drag eta_v,
the pinning stiffness kp and hence the per-vortex resonator loss?

For each true kp we generate noisy synthetic data along a line scan across
one interstitial vortex (NV height h_true unknown to the fit):
  DC image  : time-averaged B.n (thermal blur included), sigma = 20 uT/pixel
  relaxation: excess 1/T1 (thermal average of g_perp), 10 % error/point
  echo      : Hahn-echo decay rate at 8 points, 10 % error, used as a value
              when it lies in the measurable window 1e4-2e7 1/s and as a
              one-sided bound (censored point) otherwise.
The fit has three free parameters (h, eta_v, kp); lambda and d are fixed at
the model values. A second run uses a true lambda of 70 nm to expose the
systematic error of a wrong lambda. Assumed noise levels are stated in the
paper. Output: data/inference.json
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import least_squares, brentq
import params as P
import physics as ph

T = P.T_MEAS_K
kT = P.KB * T
wnv = 2 * np.pi * P.F_NV_HZ
wres = 2 * np.pi * P.F_RES_HZ
n = ph.nv_axis(P.NV_THETA_DEG, P.NV_PHI_DEG)
d = P.THICKNESS_NM * 1e-9
G2 = 3 * P.GAMMA_E**2 / 4
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(8)
gh_w = gh_w / gh_w.sum()
rr = np.linspace(0, 1.6e-6, 2401)
H_GRID = np.arange(30e-9, 90.1e-9, 2e-9)


def tables(lam):
    tab = np.array([ph.vortex_radial(rr, hh, lam, d) for hh in H_GRID])
    return tab  # (nh, 4, nr)


def radial_at(tab, h):
    i = np.clip(np.searchsorted(H_GRID, h) - 1, 0, len(H_GRID) - 2)
    f = (h - H_GRID[i]) / (H_GRID[i + 1] - H_GRID[i])
    return (1 - f) * tab[i] + f * tab[i + 1]


def fields(rad, dx, dy):
    """B.n, g_perp, g_par at relative positions (arrays)."""
    r = np.hypot(dx, dy)
    Bz, Br, dBz, dBr = (np.interp(r, rr, rad[k]) for k in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(r > 0, dx / r, 1.0); s = np.where(r > 0, dy / r, 0.0)
        Bro = np.where(r > 1e-12, Br / np.where(r > 1e-12, r, 1), dBr)
    Bx, By = Br * c, Br * s
    Gm = np.array([[dBr * c * c + Bro * s * s, (dBr - Bro) * c * s],
                   [(dBr - Bro) * c * s, dBr * s * s + Bro * c * c],
                   [dBz * c, dBz * s]])
    par = np.einsum("i,ij...->j...", n, Gm)
    gpar = (par**2).sum(0)
    gperp = (Gm**2).sum((0, 1)) - gpar
    return n[0] * Bx + n[1] * By + n[2] * Bz, gperp, gpar


def echo_rate(sigB2, tc):
    a = P.GAMMA_E**2 * sigB2 * tc**2
    f = lambda t: a * (t / tc - 3 + 4 * np.exp(-t / (2 * tc)) - np.exp(-t / tc)) - 1
    hi = tc
    while f(hi) < 0:
        hi *= 4
    return 1.0 / brentq(f, 1e-18, hi)


XD = np.linspace(-400e-9, 400e-9, 81)      # DC image points
XT = np.linspace(-300e-9, 300e-9, 31)      # T1 points
XE = np.array([-40e-9, -15e-9, 0.0, 20e-9, 80e-9, 120e-9, 200e-9, 300e-9])   # echo points


def model(theta, tab):
    h, leta, lkp = theta
    eta, kp = np.exp(leta), np.exp(lkp)
    rad = radial_at(tab, h)
    sig = min(np.sqrt(kT / kp), 250e-9)
    ux = sig * gh_x[:, None]; uy = sig * gh_x[None, :]
    W = gh_w[:, None] * gh_w[None, :]
    def avg(xs):
        b, gp, gq = fields(rad, xs[:, None, None] - ux, 0 * xs[:, None, None] - uy)
        return (b * W).sum((1, 2)), (gp * W).sum((1, 2)), (gq * W).sum((1, 2))
    bdc, _, _ = avg(XD)
    _, gpT, _ = avg(XT)
    r1 = G2 * ph.Sx_harmonic(wnv, T, eta, kp) * gpT
    _, _, gqE = fields(rad, XE, 0 * XE)
    er = np.array([echo_rate(g * kT / kp, eta / kp) for g in gqE])
    return bdc, r1, er


def run(kp_true, lam_true, lam_fit, ntrial, seed):
    rng = np.random.default_rng(seed)
    tab_t, tab_f = tables(lam_true), (tables(lam_fit) if lam_fit != lam_true else None)
    tab_f = tab_t if tab_f is None else tab_f
    th_true = np.array([P.NV_HEIGHT_NM * 1e-9, np.log(P.ETA_V), np.log(kp_true)])
    b0, r0, e0 = model(th_true, tab_t)
    use_e = (e0 > 1e4) & (e0 < 2e7)
    res = []
    for k in range(ntrial):
        bd = b0 + rng.normal(0, 20e-6, b0.size)
        rd = r0 * (1 + 0.1 * rng.standard_normal(r0.size))
        ed = e0 * (1 + 0.1 * rng.standard_normal(e0.size))
        def resid(th):
            b, r, e = model(th, tab_f)
            out = [(b - bd) / 20e-6, (np.log(np.abs(r) + 1e-30) - np.log(np.abs(rd))) / 0.1]
            if use_e.any():
                out.append((np.log(e[use_e]) - np.log(ed[use_e])) / 0.1)
            # censored echo points: below the intrinsic floor (not detected)
            # or faster than the shortest echo time (fully dephased)
            lo, hi = ~use_e & (e0 <= 1e4), ~use_e & (e0 >= 2e7)
            if lo.any():
                out.append(np.clip(np.log(e[lo] / 1e4), 0, None) / 0.1)
            if hi.any():
                out.append(np.clip(np.log(2e7 / e[hi]), 0, None) / 0.1)
            return np.concatenate(out)
        best = None
        for lk0 in (np.log(1e-7), np.log(1e-4)):
            x0 = np.array([60e-9, np.log(P.ETA_V) + 0.7, lk0])
            try:
                s = least_squares(resid, x0, x_scale=[1e-8, 1, 1],
                                  bounds=([32e-9, np.log(1e-17), np.log(1e-10)],
                                          [88e-9, np.log(1e-12), np.log(1e-1)]))
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
        h, le, lk = best.x
        eta, kp = np.exp(le), np.exp(lk)
        iq = ph.per_vortex_inverse_Q(wres, eta, kp, 2 / np.pi, 1, P.CPW_W_UM * 1e-6, P.CPW_Z0)
        res.append(dict(h=h, eta=eta, kp=kp, invQ=iq, chi2=2 * best.cost / best.fun.size))
    iq_true = ph.per_vortex_inverse_Q(wres, P.ETA_V, kp_true, 2 / np.pi, 1, P.CPW_W_UM * 1e-6, P.CPW_Z0)
    q = lambda key: np.percentile([r[key] for r in res], [16, 50, 84]).tolist()
    return dict(kp_true=kp_true, lam_true=lam_true, lam_fit=lam_fit,
                echo_points_used=int(use_e.sum()), eta_true=P.ETA_V,
                invQ_true=iq_true, h=q("h"), eta=q("eta"), kp=q("kp"),
                invQ=q("invQ"), trials=res)


t0 = time.time()
KPS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3]
out = {"baseline": [], "wrong_lambda": []}
for i, kp in enumerate(KPS):
    r = run(kp, 50e-9, 50e-9, 20, 100 + i)
    out["baseline"].append(r)
    print(f"kp={kp:.0e}: kp_fit={r['kp'][1]:.2e} [{r['kp'][0]:.2e},{r['kp'][2]:.2e}] "
          f"eta_fit/eta={r['eta'][1]/P.ETA_V:.3f} invQ_fit/true={r['invQ'][1]/r['invQ_true']:.3f} "
          f"h={r['h'][1]*1e9:.1f} nm echo_pts={r['echo_points_used']} ({time.time()-t0:.0f}s)", flush=True)
for i, kp in enumerate([1e-7, 1e-5, 3e-4]):
    r = run(kp, 70e-9, 50e-9, 10, 300 + i)
    out["wrong_lambda"].append(r)
    print(f"[lam 70 fit 50] kp={kp:.0e}: kp_fit/true={r['kp'][1]/kp:.2f} "
          f"eta_fit/eta={r['eta'][1]/P.ETA_V:.3f} invQ_fit/true={r['invQ'][1]/r['invQ_true']:.3f} "
          f"h={r['h'][1]*1e9:.1f} nm", flush=True)
with open("data/inference.json", "w") as f:
    json.dump(out, f, indent=1)
print("DONE")
