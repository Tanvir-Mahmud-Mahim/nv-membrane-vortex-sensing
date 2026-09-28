"""Independent stochastic checks of the three relations the paper relies on.

A. Fluctuation-dissipation (FDT) for a pinned vortex. Brownian dynamics in
   (i) a harmonic well and (ii) an anharmonic double well (thermal hopping).
   The absorbed power measured in DRIVEN runs is compared with
   w^2 F^2 S_x(w) / (8 kB T) computed from the NOISE of UNDRIVEN runs.
   No analytic spectrum enters the comparison.
B. NV relaxation. The field at an NV above a thermally moving vortex is
   computed with the exact London stray field (no linearisation). Its
   spectral density at the NV frequency is compared with
   g_perp S_x(w) (small motion) and with <g_perp> over the thermal cloud
   (large motion, "blur" regime).
C. NV dephasing. Ramsey coherence under the exact vortex field is compared
   with the Gaussian (Anderson-Kubo) prediction built from the linearised
   field variance and the correlation time eta_v/kp.
Outputs: data/langevin_checks.npz and data/langevin_checks.json
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from numba import njit
from scipy.signal import welch
import params as P
import physics as ph

rng = np.random.default_rng(20260928)
out = {}


# ---------------------------------------------------------------- A: FDT
@njit(cache=True)
def bd_run(model, n, steps, dt, f, w, seed, x0, nkeep):
    """Overdamped Brownian dynamics in reduced units (kT = 1, eta = 1).
    model 0: U = x^2/2 ; model 1: U = Ub ((x/a)^2 - 1)^2 with Ub = 4, a = 2.
    Returns the particle-averaged trajectory and one raw trajectory set."""
    np.random.seed(seed)
    x = x0.copy()
    mean = np.zeros(steps)
    keep = np.zeros((nkeep, steps), dtype=np.float32)
    s = np.sqrt(2 * dt)
    for i in range(steps):
        t = i * dt
        drive = f * np.cos(w * t)
        for j in range(n):
            if model == 0:
                force = -x[j]
            else:
                force = -4.0 * 4.0 * x[j] / 4.0 * ((x[j] / 2.0) ** 2 - 1.0)
            x[j] += (force + drive) * dt + s * np.random.randn()
        mean[i] = x.mean()
        for j in range(keep.shape[0]):
            keep[j, i] = x[j]
    return mean, keep


def logbin(w, S, n=300):
    """Average a spectrum in n logarithmic frequency bins (storage only)."""
    edges = np.logspace(np.log10(w.min()), np.log10(w.max()), n + 1)
    idx = np.digitize(w, edges) - 1
    wb, Sb = [], []
    for k in range(n):
        sel = idx == k
        if sel.any():
            wb.append(float(np.exp(np.mean(np.log(w[sel])))))
            Sb.append(float(np.mean(S[sel])))
    return wb, Sb


def spectrum_from_noise(model, dt, steps, n, seed, nperseg):
    x0 = np.zeros(n) if model == 0 else np.sign(rng.standard_normal(n)) * 2.0
    _, keep = bd_run(model, n, steps, dt, 0.0, 0.0, seed, x0, n)
    fr, S = welch(keep, fs=1 / dt, nperseg=nperseg, axis=1)
    return 2 * np.pi * fr, S.mean(axis=0)          # one-sided, per unit f


def power_from_drive(model, w, f, dt, n, seed):
    period = 2 * np.pi / w
    n_skip = max(3, int(np.ceil(200 / period)))      # >= 200 relaxation units
    n_avg = max(10, int(np.ceil(1500 / period)))
    steps = int((n_skip + n_avg) * period / dt)
    x0 = np.zeros(n) if model == 0 else np.sign(rng.standard_normal(n)) * 2.0
    mean, _ = bd_run(model, n, steps, dt, f, w, seed, x0, 1)
    t = np.arange(steps) * dt
    m = (t > n_skip * period) & (t <= (n_skip + n_avg) * period)
    # in-phase (with force) and quadrature parts of <x>
    c = np.cos(w * t[m]); s = np.sin(w * t[m])
    xs = 2 * np.mean(mean[m] * s)
    return 0.5 * w * f * xs          # P = (w/2) F^2 Im(alpha), Im(alpha)=xs/F


t0 = time.time()
import os
A_CACHE = "data/langevin_fdt_A.json"
resA = json.load(open(A_CACHE)) if os.path.exists(A_CACHE) else {}
# harmonic: well relaxation time 1; double well: barrier 4 kT, hop rate ~1e-2
SET = {0: dict(dt=2e-3, steps=2**20, n=16, nperseg=2**17, f=0.3, nd=2000,
               ws=np.logspace(-1.0, 1.3, 8)),
       1: dict(dt=5e-3, steps=2**22, n=16, nperseg=2**20, f=0.1, nd=3000,
               ws=np.logspace(-2.0, 0.7, 8))}
for model, name in [(0, "harmonic"), (1, "double_well")]:
    if name in resA:
        continue
    c = SET[model]
    dt = c["dt"]
    wS, S = spectrum_from_noise(model, dt, c["steps"], c["n"], 11 + model,
                                c["nperseg"])
    ws = c["ws"]
    P_drive, P_fdt = [], []
    for k, w in enumerate(ws):
        f = c["f"]
        P_drive.append(power_from_drive(model, w, f, dt, c["nd"], 100 + k))
        # smooth the noise spectrum over +-15% in w before sampling it
        band = (wS > w / 1.15) & (wS < w * 1.15)
        Sw = float(np.mean(S[band])) if band.sum() > 2 else float(np.interp(w, wS, S))
        P_fdt.append(w**2 * f**2 * Sw / 8.0)
    wb, Sb = logbin(wS[1:], S[1:])        # compact storage of the spectrum
    resA[name] = dict(w=ws.tolist(), P_drive=P_drive, P_fdt=P_fdt,
                      wS=wb, S=Sb)
    rel = np.abs(np.array(P_drive) / np.array(P_fdt) - 1)
    print(f"A/{name}: median |P_drive/P_fdt - 1| = {np.median(rel):.3f}, "
          f"max = {rel.max():.3f}  ({time.time()-t0:.0f} s)", flush=True)
out["fdt"] = resA
with open(A_CACHE, "w") as f:
    json.dump(resA, f)


# ------------------------------------------------ B, C: NV above a vortex
lam, d, h = P.LAMBDA_0_NM * 1e-9, P.THICKNESS_NM * 1e-9, P.NV_HEIGHT_NM * 1e-9
T, eta = P.T_MEAS_K, P.ETA_V
n_nv = ph.nv_axis(P.NV_THETA_DEG, P.NV_PHI_DEG)
rr = np.linspace(0, 1.2e-6, 6001)
Bz_r, Br_r, dBz_r, dBr_r = ph.vortex_radial(rr, h, lam, d)


@njit(cache=True)
def field_at(dx, dy, rr, Bz_r, Br_r, nx, ny, nz):
    """Exact field components at the NV for vortex displacement (dx, dy) when
    the NV sits at (X0, 0) relative to the vortex rest position: the caller
    passes relative coordinates."""
    r = np.sqrt(dx * dx + dy * dy)
    bz = np.interp(r, rr, Bz_r)
    br = np.interp(r, rr, Br_r)
    if r > 0:
        bx = br * dx / r; by = br * dy / r
    else:
        bx = 0.0; by = 0.0
    bpar = nx * bx + ny * by + nz * bz
    return bx, by, bz, bpar


@njit(cache=True)
def nv_traj(n, steps, dt, eta, kp, kT, X0, rr, Bz_r, Br_r, nx, ny, nz, seed,
            store, gam):
    """2D OU vortex; returns B components at the NV for `store` particles
    and the Ramsey coherence |<exp(i phase)>| over all particles."""
    np.random.seed(seed)
    sig = np.sqrt(kT / kp)
    ux = np.random.randn(n) * sig
    uy = np.random.randn(n) * sig
    amp = np.sqrt(2 * kT * dt / eta)      # eta dx = -kp x dt + sqrt(2 kT eta) dW
    a = kp / eta * dt
    Bst = np.zeros((store, 4, steps))
    cr = np.zeros(steps)
    ci = np.zeros(steps)
    ph_ = np.zeros(n)
    for i in range(steps):
        for j in range(n):
            ux[j] += -a * ux[j] + amp * np.random.randn()
            uy[j] += -a * uy[j] + amp * np.random.randn()
            bx, by, bz, bp = field_at(X0 - ux[j], -uy[j], rr, Bz_r, Br_r,
                                      nx, ny, nz)
            ph_[j] += gam * bp * dt
            cr[i] += np.cos(ph_[j]) / n
            ci[i] += np.sin(ph_[j]) / n
            if j < store:
                Bst[j, 0, i] = bx; Bst[j, 1, i] = by
                Bst[j, 2, i] = bz; Bst[j, 3, i] = bp
    return Bst, np.sqrt(cr**2 + ci**2)


kT = P.KB * T
wnv = 2 * np.pi * P.F_NV_HZ
X0 = 100e-9          # NV lateral offset from the vortex rest position
Xg = np.array([[X0]]); Yg = np.array([[0.0]])
_, Gm = ph.field_and_gradient_maps(Xg, Yg, h, lam, d)
g_perp_lin, g_par_lin = [float(np.ravel(v)[0]) for v in ph.geometric_factors(Gm, n_nv)]

resB = []
for kp in [1e-5, 1e-7, 1e-8]:
    tc = eta / kp
    dt = min(tc / 20, 1 / (P.F_NV_HZ * 20))
    steps = 2**15
    Bst, _ = nv_traj(64, steps, dt, eta, kp, kT, X0, rr, Bz_r, Br_r,
                     *n_nv, 7, 64, P.GAMMA_E)
    # transverse field power: sum over components minus the parallel one
    fr, Sxx = welch(Bst[:, 0], fs=1 / dt, nperseg=2**12, axis=1)
    _, Syy = welch(Bst[:, 1], fs=1 / dt, nperseg=2**12, axis=1)
    _, Szz = welch(Bst[:, 2], fs=1 / dt, nperseg=2**12, axis=1)
    _, Spp = welch(Bst[:, 3], fs=1 / dt, nperseg=2**12, axis=1)
    Sperp = (Sxx + Syy + Szz - Spp).mean(axis=0)
    S_sim = float(np.interp(P.F_NV_HZ, fr, Sperp))
    S_lin = g_perp_lin * ph.Sx_harmonic(wnv, T, eta, kp)
    # thermal average of g_perp over the displacement cloud (blur regime)
    sig = np.sqrt(kT / kp)
    u = rng.standard_normal((4000, 2)) * sig
    Xs = (X0 - u[:, 0])[:, None]; Ys = (-u[:, 1])[:, None]
    _, Gs = ph.field_and_gradient_maps(Xs, Ys, h, lam, d)
    gp_avg = float(np.mean(ph.geometric_factors(Gs, n_nv)[0]))
    S_avg = gp_avg * ph.Sx_harmonic(wnv, T, eta, kp)
    resB.append(dict(kp=kp, sigma_x_nm=sig * 1e9, S_sim=S_sim, S_lin=S_lin,
                     S_thermal_avg=S_avg, ratio_lin=S_sim / S_lin,
                     ratio_avg=S_sim / S_avg))
    print(f"B/kp={kp:.0e}: sigma_x={sig*1e9:.1f} nm  S_sim/S_lin="
          f"{S_sim/S_lin:.3f}  S_sim/S_avg={S_sim/S_avg:.3f} "
          f"({time.time()-t0:.0f} s)", flush=True)
out["t1"] = resB

# C: Ramsey coherence, motional-narrowing and intermediate regimes
resC = []
for kp in [3e-6, 3e-7, 1e-7]:
    tc = eta / kp
    sigB2 = g_par_lin * kT / kp
    G_mn = P.GAMMA_E**2 * sigB2 * tc
    tmax = 4.0 / G_mn if G_mn * tc < 0.3 else 6 * np.sqrt(2 / (P.GAMMA_E**2 * sigB2))
    dt = min(tc / 10, tmax / 4000)
    steps = int(tmax / dt)
    _, coh = nv_traj(3000, steps, dt, eta, kp, kT, X0, rr, Bz_r, Br_r,
                     *n_nv, 21, 1, P.GAMMA_E)
    tt = np.arange(1, steps + 1) * dt
    x = tt / tc
    kubo = np.exp(-P.GAMMA_E**2 * sigB2 * tc**2 * (np.exp(-x) + x - 1))
    sel = np.linspace(0, steps - 1, 200).astype(int)
    resC.append(dict(kp=kp, tau_c=tc, gamma_sigma_tau=float(np.sqrt(
        P.GAMMA_E**2 * sigB2) * tc), t=tt[sel].tolist(),
        coh_sim=coh[sel].tolist(), coh_kubo=kubo[sel].tolist(),
        max_abs_dev=float(np.max(np.abs(coh - kubo)))))
    print(f"C/kp={kp:.0e}: gamma*sigma*tau_c={resC[-1]['gamma_sigma_tau']:.2f}"
          f"  max|sim-kubo|={resC[-1]['max_abs_dev']:.3f} "
          f"({time.time()-t0:.0f} s)", flush=True)
out["ramsey"] = resC
out["X0_nm"] = X0 * 1e9
with open("data/langevin_checks.json", "w") as f:
    json.dump(out, f, indent=1)
print("DONE")
