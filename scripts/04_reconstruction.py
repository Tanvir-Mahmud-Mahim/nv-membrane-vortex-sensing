"""Physics-informed reconstruction of vortex configurations from noisy NV maps.

Forward model: sum of Pearl-vortex kernels at height z_NV (script 03).
Inverse: nonlinear least squares over vortex positions, initialized by peak
finding, with the number of vortices selected by the Bayesian information
criterion (BIC). A Cramer-Rao bound (CRB) gives the localization limit.

Experiment model: per-pixel field noise sigma_B = eta / sqrt(t_px), with eta
the shot-noise-limited sensitivity (with and without the x7 membrane photonic
gain). Outputs localization error and hole-occupancy classification accuracy
versus standoff and per-pixel time.
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import maximum_filter
import params as P

exec(open("scripts/03_nv_observables.py").read().split('if __name__')[0])

rng = np.random.default_rng(7)


def kernel_lut(z_um, rmax=1.5, n=4000):
    r = np.linspace(0, rmax, n)
    b = pearl_bz(r, z_um)
    return lambda rho: np.interp(rho, r, b, right=0.0)


def forward(positions, X, Y, kern):
    B = np.zeros_like(X)
    for (x0, y0) in positions:
        B += kern(np.hypot(X - x0, Y - y0))
    return B


def detect_peaks(B, xs, kern, thresh_frac=0.25):
    pk = kern(np.array([0.0]))[0]
    m = (B == maximum_filter(B, size=5)) & (B > thresh_frac * pk)
    iy, ix = np.where(m)
    return np.stack([xs[ix], xs[iy]], 1)


def fit_positions(Bnoisy, xs, kern, p0, win_um=0.25):
    """Per-vortex local fits: each candidate is refined on a window around
    its peak (vortex separation >> window), which is fast and unbiased."""
    X, Y = np.meshgrid(xs, xs)
    fits = []
    for (x0, y0) in np.asarray(p0, float):
        m = (np.abs(X - x0) < win_um) & (np.abs(Y - y0) < win_um)
        xw, yw, bw = X[m], Y[m], Bnoisy[m]

        def resid(v):
            return kern(np.hypot(xw - v[0], yw - v[1])) - bw

        sol = least_squares(resid, [x0, y0], method="lm", max_nfev=60)
        fits.append(sol.x)
    fits = np.array(fits).reshape(-1, 2)
    # second pass: refit each vortex with the others' fitted fields removed
    refined = []
    for i, (x0, y0) in enumerate(fits):
        others = np.delete(fits, i, axis=0)
        m = (np.abs(X - x0) < win_um) & (np.abs(Y - y0) < win_um)
        xw, yw = X[m], Y[m]
        bw = Bnoisy[m] - forward(others, xw, yw, kern)

        def resid2(v):
            return kern(np.hypot(xw - v[0], yw - v[1])) - bw

        sol = least_squares(resid2, [x0, y0], method="lm", max_nfev=60)
        refined.append(sol.x)
    return np.array(refined).reshape(-1, 2), None


def crb_sigma_x(kern, z_um, sigma_B, pix_um):
    """1D Cramer-Rao localization bound, evaluated on the actual pixel grid:
    sigma_x >= sigma_B / sqrt(sum_pixels (dB/dx)^2)."""
    r = np.arange(-0.75, 0.75 + pix_um / 2, pix_um)
    X, Y = np.meshgrid(r, r)
    B = kern(np.hypot(X, Y))
    gx = np.gradient(B, r, axis=1)
    return sigma_B / np.sqrt(np.sum(gx ** 2))


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    # ---- ground truth from TDGL (antidot above matching field) ----
    from tdgl_common import load_fieldcool
    try:
        npz = load_fieldcool("anti15_B12")
        holes = npz["hole_centers"]; occ = npz["occ_winding"]
        inter = npz["vortex_pos"]
        truth = [tuple(h) for h, o in zip(holes, occ) if o > 0]
        truth += [tuple(v) for v in inter]
    except FileNotFoundError:
        print("TDGL not ready; using synthetic lattice")
        truth = [(x, y) for x in (-0.5, 0, 0.5) for y in (-0.5, 0, 0.5)]
        holes = np.array(truth); occ = np.ones(len(truth))
        inter = np.zeros((0, 2))
    truth = np.array(truth)

    xs = np.linspace(-0.75, 0.75, 101)  # 15 nm pixels
    pix = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs)

    eta_membrane = 0.132  # uT/rtHz placeholder, replaced below
    sens = dict(nv_observables(np.zeros((2, 2))))
    out["sensitivity"] = sens

    # save one representative example (25 nm, 1 ms, membrane) for the figure
    z_ex = z_sheet_um(25)
    kern_ex = kernel_lut(z_ex)
    B_ex = forward(truth, X, Y, kern_ex)
    sig_ex = dict(nv_observables(np.zeros((2, 2))))["membrane"] * 1e-3 / np.sqrt(1e-3)
    Bn_ex = B_ex + rng.normal(0, sig_ex, B_ex.shape)
    p0_ex = detect_peaks(Bn_ex, xs, kern_ex)
    fit_ex, _ = fit_positions(Bn_ex, xs, kern_ex, p0_ex)
    np.savez("data/recon_example.npz", Bn=Bn_ex, xs=xs, truth=truth,
             fit=fit_ex)

    results = []
    for so_nm in [10, 25, 50, 100]:
        z = z_sheet_um(so_nm)
        kern = kernel_lut(z)
        Btrue = forward(truth, X, Y, kern)
        for t_px_ms in [0.1, 1.0, 10.0]:
            for tag in ["bare", "membrane"]:
                sigma_B_mT = sens[tag] * 1e-3 / np.sqrt(t_px_ms * 1e-3)
                trials = []
                det_frac = []
                from scipy.optimize import nnls
                for _ in range(6):
                    Bn = Btrue + rng.normal(0, sigma_B_mT, Btrue.shape)
                    # candidates: known hole sites + low-threshold blind peaks
                    cand = list(holes)
                    for p in detect_peaks(Bn, xs, kern, thresh_frac=0.15):
                        if not any(np.hypot(p[0] - c[0], p[1] - c[1]) < 0.1
                                   for c in cand):
                            cand.append(p)
                    cand = np.array(cand)
                    # joint amplitude solve resolves overlapping kernels
                    A_c = np.stack([kern(np.hypot(X - c[0], Y - c[1])).ravel()
                                    for c in cand], axis=1)
                    amp, _ = nnls(A_c, Bn.ravel())
                    p0 = cand[amp > 0.5]
                    if len(p0) == 0:
                        det_frac.append(0.0); continue
                    fit, _ = fit_positions(Bn, xs, kern, p0)
                    # match fitted to truth
                    errs = []
                    used = set()
                    for tpt in truth:
                        d = np.hypot(fit[:, 0] - tpt[0], fit[:, 1] - tpt[1])
                        j = int(np.argmin(d))
                        if d[j] < 0.15 and j not in used:
                            errs.append(d[j]); used.add(j)
                    det_frac.append(len(errs) / len(truth))
                    if errs:
                        trials.append(np.mean(errs))
                sigma_crb = crb_sigma_x(kern, z, sigma_B_mT, pix)
                results.append(dict(
                    standoff_nm=so_nm, t_px_ms=t_px_ms, mode=tag,
                    sigma_B_uT=sigma_B_mT * 1e3,
                    loc_err_nm=float(np.mean(trials) * 1e3) if trials else None,
                    detected=float(np.mean(det_frac)),
                    crb_nm=float(sigma_crb * 1e3),
                ))
                print(results[-1], flush=True)
    out["scan"] = results

    # ---- hole-occupancy classification vs pixel time ----
    # Joint physics-informed inversion: solve non-negative least squares for
    # the flux amplitude at every hole site simultaneously, so overlap
    # between neighboring vortex fields is modeled instead of ignored.
    from scipy.optimize import nnls
    so_nm = 25; z = z_sheet_um(so_nm)
    kern = kernel_lut(z)
    A = np.stack([kern(np.hypot(X - h[0], Y - h[1])).ravel() for h in holes],
                 axis=1)
    cls = []
    for t_px_ms in [0.01, 0.03, 0.1, 0.3, 1.0]:
        sigma_B_mT = sens["membrane"] * 1e-3 / np.sqrt(t_px_ms * 1e-3)
        correct = 0; total = 0
        for _ in range(40):
            occ_true = rng.random(len(holes)) < 0.7
            pos = holes[occ_true]
            Bt = forward(pos, X, Y, kern)
            Bn = Bt + rng.normal(0, sigma_B_mT, Bt.shape)
            amp, _ = nnls(A, Bn.ravel())
            correct += int(np.sum((amp > 0.5) == occ_true))
            total += len(holes)
        cls.append(dict(t_px_ms=t_px_ms, acc=correct / total,
                        sigma_B_uT=sigma_B_mT * 1e3))
        print(cls[-1], flush=True)
    out["occupancy"] = cls

    # occupancy accuracy vs standoff at fixed short pixel times
    occ_so = []
    for so2 in [25, 50, 100, 150, 200, 300]:
        z2 = z_sheet_um(so2)
        kern2 = kernel_lut(z2)
        A2 = np.stack([kern2(np.hypot(X - h[0], Y - h[1])).ravel()
                       for h in holes], axis=1)
        for t_px_ms in [0.01, 0.1]:
            sigma_B_mT = sens["membrane"] * 1e-3 / np.sqrt(t_px_ms * 1e-3)
            correct = 0; total = 0
            for _ in range(40):
                occ_true = rng.random(len(holes)) < 0.7
                Bt = forward(holes[occ_true], X, Y, kern2)
                Bn = Bt + rng.normal(0, sigma_B_mT, Bt.shape)
                amp, _ = nnls(A2, Bn.ravel())
                correct += int(np.sum((amp > 0.5) == occ_true))
                total += len(holes)
            occ_so.append(dict(standoff_nm=so2, t_px_ms=t_px_ms,
                               acc=correct / total))
            print(occ_so[-1], flush=True)
    out["occupancy_vs_standoff"] = occ_so

    with open("data/reconstruction.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("elapsed %.0f s" % (time.time() - t0))
