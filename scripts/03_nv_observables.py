"""NV-plane stray-field maps and NV observables.

Layers:
 1. Exact Biot-Savart field from the TDGL sheet-current solution, evaluated at
    the NV plane (film surface + membrane standoff + NV depth).
 2. Analytic Pearl-vortex kernel for the same plane (used later by the
    physics-informed reconstruction).
 3. NV observables: ODMR line shift, per-pixel sensing time at the shot-noise
    limit (with and without the x7 membrane photonic collection enhancement),
    and T1 relaxometry contrast from vortex Langevin noise.

Anchors:
  NV depth ~6 nm, membrane standoff ~25 nm, Hahn T2 ~100 us, collection
  enhancement x7  [Pakpour-Tabrizi 2026].
  eta (vortex drag), J_d from TDGL; gamma_NV = 28.024 GHz/T.
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
import params as P
from tdgl_common import load_fieldcool

GAMMA_NV = 28.024e9   # Hz/T
PROJ = 1 / np.sqrt(3)  # <111> NV axis projection of out-of-plane field ((100) membrane)
KB = 1.380649e-23

LAMBDA_PEARL_UM = 2 * (P.LAMBDA_0_NM * 1e-3) ** 2 / (P.THICKNESS_NM * 1e-3)


def z_sheet_um(standoff_nm):
    """NV height above the TDGL current sheet (film midplane): membrane
    standoff + NV implantation depth + half the film thickness. This is the
    conservative thin-film convention; all kernel and map evaluations use it."""
    return (P.THICKNESS_NM / 2 + standoff_nm + P.NV_DEPTH_NM) * 1e-3


Z_NV_UM = z_sheet_um(P.MEMBRANE_STANDOFF_NM)
Z_SURF_UM = Z_NV_UM  # legacy alias: all heights are sheet-referenced


def pearl_bz(rho_um, z_um, Lam=LAMBDA_PEARL_UM, xi_um=P.XI_0_NM * 1e-3):
    """B_z (mT) of a single Pearl vortex at radial distance rho, height z.
    Hankel-transform kernel with a Gaussian core cutoff at the coherence
    length:  B_z = (Phi0/2pi) Int k J0(k rho) e^{-k z - k^2 xi^2/2}/(1+k Lam) dk
    """
    from scipy.special import j0
    rho = np.atleast_1d(np.asarray(rho_um, float))
    kmax = 40.0 / max(z_um + 0.3 * xi_um, 0.01)
    k = np.linspace(1e-6, kmax, 6000)
    F = k * np.exp(-k * z_um - (k * xi_um) ** 2 / 2) / (1 + k * Lam)
    out = np.array([np.trapezoid(F * j0(k * r), k) for r in rho])
    PHI0_mT_um2 = P.PHI0 * 1e15  # 1 Wb = 1e3 mT * 1e12 um^2
    return PHI0_mT_um2 / (2 * np.pi) * out


def site_areas(sites):
    """Per-site quadrature weights (um^2) from the Delaunay dual: one third
    of each incident triangle's area."""
    from scipy.spatial import Delaunay
    tri = Delaunay(sites)
    A = np.zeros(len(sites))
    p = sites[tri.simplices]
    a2 = np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])) / 2
    for j in range(3):
        np.add.at(A, tri.simplices[:, j], a2 / 3)
    return A


def biot_savart_map(npz, z_um, half=0.6, n=121):
    """Exact B_z (mT) at height z above the film from the TDGL sheet current
    (uA/um = A/m) by direct Biot-Savart quadrature over the mesh, using
    Delaunay dual-cell weights. The applied uniform field is not included;
    the map is the film's response (vortices plus screening)."""
    sites = npz["sites"]; K = npz["K"]  # um; A/m
    areas = site_areas(sites)           # um^2
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    Bz = np.zeros(len(pts))
    step = 1500
    for i0 in range(0, len(pts), step):
        p = pts[i0:i0 + step]
        dx = (p[:, None, 0] - sites[None, :, 0]) * 1e-6      # m
        dy = (p[:, None, 1] - sites[None, :, 1]) * 1e-6      # m
        r3 = (dx ** 2 + dy ** 2 + (z_um * 1e-6) ** 2) ** 1.5
        # B_z = mu0/4pi * (K x r)_z / r^3, r from source to field point
        cross = K[None, :, 0] * dy - K[None, :, 1] * dx      # A/m * m
        Bz[i0:i0 + step] = 1e-7 * np.sum(cross / r3 * areas[None, :] * 1e-12,
                                         axis=1)
    return xs, Bz.reshape(n, n) * 1e3  # mT


def nv_observables(Bz_mT):
    """ODMR shift map and per-pixel imaging time."""
    df_MHz = GAMMA_NV * PROJ * (Bz_mT * 1e-3) / 1e6
    # Shot-noise-limited pulsed-ODMR sensitivity (Dreau 2011 scaling):
    # eta = P_F * dnu / (C * sqrt(R)) with typical shallow-NV numbers.
    C = 0.25          # contrast
    R0 = 60e3         # photons/s detected, confocal on bare membrane host
    dnu = 8e6         # ODMR linewidth, Hz
    for gain, tag in [(1.0, "bare"), (7.0, "membrane")]:
        R = R0 * gain
        eta_T = 0.7 * (dnu) / (GAMMA_NV * C * np.sqrt(R))  # T/sqrt(Hz)
        yield tag, eta_T * 1e6  # uT/sqrt(Hz)


def relaxometry(J_d_sheet_A_per_m, T_K=2.1):
    """Two-channel NV noise spectroscopy of a thermally fluctuating vortex.

    Overdamped Langevin dynamics in a pinning potential of stiffness k_p with
    Bardeen-Stephen drag eta_v gives the one-sided position noise
        S_x(w) = 4 k_B T eta_v / (k_p^2 + w^2 eta_v^2).
    The NV at standoff z sees S_B = |grad B|^2 S_x (per axis; x2 for 2D).
      Channel 1 (T1): at w_NV = 2 pi x 2.87 GHz the response is drag
        dominated (w eta >> k_p), so Gamma_1 reads the viscosity eta_v, the
        same parameter that sets vortex microwave loss in resonators.
      Channel 2 (T2/dephasing): the low-frequency plateau scales as 1/k_p^2,
        so Hahn-echo dephasing reads the pinning stiffness (the depinning
        current), with huge contrast between pinned and weakly pinned sites.
    """
    eta_v = P.ETA_BS * (P.THICKNESS_NM * 1e-9)          # kg/s per vortex
    F_dep = J_d_sheet_A_per_m * P.PHI0                   # N per vortex
    xi_m = P.XI_0_NM * 1e-9
    out = {}
    z_m = Z_SURF_UM * 1e-6
    Bpk = pearl_bz(np.array([0.0]), Z_SURF_UM)[0] * 1e-3  # T
    gradB = Bpk / (z_m + xi_m)                            # T/m scale
    w_nv = 2 * np.pi * 2.87e9
    gam = 2 * np.pi * GAMMA_NV                            # rad/s/T
    # lateral profile of the kernel gradient at the NV plane
    rho = np.linspace(0, 0.45, 120)                       # um
    bz = pearl_bz(rho, Z_SURF_UM) * 1e-3                  # T
    dbdr = np.gradient(bz, rho * 1e-6)                    # T/m
    prof = {}
    for tag, kp in [
        ("antidot", F_dep / xi_m),
        ("natural", 0.02 * F_dep / xi_m),
    ]:
        w_p = kp / eta_v
        tau_c = eta_v / kp
        var_x = KB * T_K / kp                              # m^2
        Sx_nv = 4 * KB * T_K * eta_v / (kp ** 2 + (w_nv * eta_v) ** 2)
        Sx_0 = 4 * KB * T_K * eta_v / kp ** 2
        # on-peak-gradient values
        SB_nv = 2 * gradB ** 2 * Sx_nv
        SB_0 = 2 * gradB ** 2 * Sx_0
        G1 = gam ** 2 * SB_nv / 2
        # dephasing: motional-narrowed rate, capped by the quasi-static limit
        var_B = 2 * gradB ** 2 * var_x                     # T^2
        G_mn = gam ** 2 * PROJ ** 2 * var_B * tau_c
        G_static = gam * PROJ * np.sqrt(var_B)
        Gphi = min(G_mn, G_static)
        # lateral profiles (per-axis gradient, both channels)
        g2 = dbdr ** 2
        G1_rho = gam ** 2 * (2 * g2 * Sx_nv) / 2
        varB_rho = 2 * g2 * var_x
        Gphi_rho = np.minimum(gam ** 2 * PROJ ** 2 * varB_rho * tau_c,
                              gam * PROJ * np.sqrt(varB_rho))
        prof[tag] = dict(G1_rho=G1_rho, Gphi_rho=Gphi_rho)
        out[tag] = dict(
            k_p_N_per_m=kp, f_p_GHz=w_p / 2 / np.pi / 1e9,
            tau_c_ns=tau_c * 1e9, x_rms_nm=np.sqrt(var_x) * 1e9,
            S_B_nv_T2Hz=SB_nv, S_B_0_T2Hz=SB_0,
            Gamma1_per_s=G1, T1_us=1e6 / G1,
            Gammaphi_per_s=Gphi, T2phi_us=1e6 / Gphi,
        )
    np.savez("data/noise_profiles.npz", rho_um=rho,
             G1_antidot=prof["antidot"]["G1_rho"],
             G1_natural=prof["natural"]["G1_rho"],
             Gphi_antidot=prof["antidot"]["Gphi_rho"],
             Gphi_natural=prof["natural"]["Gphi_rho"])
    out["Bpk_mT"] = Bpk * 1e3
    out["gradB_T_per_m"] = gradB
    out["eta_v_kg_per_s"] = eta_v
    out["F_dep_N"] = F_dep
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {"Lambda_pearl_um": LAMBDA_PEARL_UM, "z_nv_um": Z_NV_UM}
    # Pearl profile for several standoffs
    rho = np.linspace(0, 0.5, 200)
    prof = {}
    for so_nm in [10, 25, 50, 100]:
        z = z_sheet_um(so_nm)
        prof[str(so_nm)] = pearl_bz(rho, z).tolist()
    np.savez("data/pearl_profiles.npz", rho_um=rho,
             **{f"so{k}": np.array(v) for k, v in prof.items()})
    res["Bpk_25nm_mT"] = prof["25"][0]

    # Exact maps from TDGL solutions
    for tag in ["bare_B4", "anti15_B8"]:
        try:
            npz = load_fieldcool(tag)
        except FileNotFoundError:
            print("missing", tag); continue
        xs, Bz = biot_savart_map(npz, Z_NV_UM)
        np.savez(f"data/nvmap_{tag}.npz", xs=xs, Bz_mT=Bz,
                 B_applied=npz["B_mT"])
        res[f"map_{tag}_minmax"] = [float(Bz.min()), float(Bz.max())]
        print(tag, "map range", Bz.min(), Bz.max(), flush=True)

    res["sensitivity_uT_sqrtHz"] = dict(nv_observables(np.zeros((2, 2))))
    # J_d from IV data: the antidot film never depins in the simulated range
    # (its vortex voltage stays ~10x below the bare film), so the maximum
    # simulated current is a lower bound on the depinning current.
    try:
        iv = json.load(open("data/pinning_iv.json"))
        I = np.array(iv["I_uA"])
        V = np.array(iv["anti15_B8.27"]) - np.array(iv["anti15_B0.00"])
        Vb = np.array(iv["bare_B8.27"]) - np.array(iv["bare_B0.00"])
        depinned = V[3:] > 0.5 * Vb[3:]     # skip settle transient
        I_d = I[3:][np.argmax(depinned)] if depinned.any() else I[-1]
    except Exception as e:
        print("IV not ready:", e); I_d = 400.0
    J_d = I_d * 1e-6 / (P.FILM_W * 1e-6)  # A/m
    res["I_d_uA"] = float(I_d); res["J_d_A_per_m"] = float(J_d)
    res["relaxometry"] = {k: (v if not isinstance(v, dict) else
                              {kk: float(vv) for kk, vv in v.items()})
                          for k, v in relaxometry(J_d).items()}
    with open("data/nv_observables.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res["relaxometry"], indent=1, default=str)[:1200])
    print("elapsed", time.time() - t0)
