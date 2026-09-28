"""Publication figures (IEEE two-column, Times New Roman, colorblind-safe).

Usage: python scripts/07_figures.py [name ...]   (output: $FIG_DIR, default figures/)
names: abstract, vortex, maps, tdgl, checks, readout, supp
Layout rules: panel labels outside the axes (top-left, bold 9 pt), colorbars
in their own axes, legends placed away from data, no text outside a panel.
"""
import sys, json, glob, os
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, TwoSlopeNorm
from figstyle import C, FW_1COL, FW_2COL
import params as P
import physics as ph

OUT = os.environ.get("FIG_DIR", "figures")
OUTS = os.path.join(OUT, "supplementary")
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUTS, exist_ok=True)
lam, d, h = P.LAMBDA_0_NM * 1e-9, P.THICKNESS_NM * 1e-9, P.NV_HEIGHT_NM * 1e-9
n_nv = ph.nv_axis(P.NV_THETA_DEG, P.NV_PHI_DEG)


def plab(ax, s, dx=-0.02, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def save(fig, name, supp=False):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTS if supp else OUT}/{name}.{ext}")
    plt.close(fig)
    print("saved", name)


# =====================================================================
def fig_vortex():
    """Fig. 2: one vortex, its field, its noise, and why it outshines the film."""
    obs = json.load(open("data/observables.json"))
    fig = plt.figure(figsize=(FW_2COL, 4.35))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42, left=0.07,
                          right=0.975, top=0.94, bottom=0.10)
    # (a) field profiles at the NV height
    ax = fig.add_subplot(gs[0, 0])
    rr = np.linspace(0, 600e-9, 601)
    Bz, Br, _, _ = ph.vortex_radial(rr, h, lam, d)
    Bh2, _, _, _ = ph.vortex_radial(rr, h, lam, d, form=ph.disk_form(100e-9))
    Bh, _, _, _ = ph.vortex_radial(rr, h, lam, d, form=ph.disk_form(75e-9))
    ax.plot(rr * 1e9, Bz * 1e3, color=C["blue"], label="vortex core, $B_z$")
    ax.plot(rr * 1e9, Br * 1e3, color=C["blue"], ls="--", label=r"vortex core, $B_\rho$")
    ax.plot(rr * 1e9, Bh * 1e3, color=C["orange"], label="fluxoid in 150 nm hole")
    ax.plot(rr * 1e9, Bh2 * 1e3, color=C["orange"], ls=":",
            label="fluxoid in 200 nm hole")
    ax.set_xlabel("lateral distance (nm)")
    ax.set_ylabel("field at NV height (mT)")
    ax.set_xlim(0, 600); ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=6.4, handlelength=1.6)
    plab(ax, "(a)")
    # (b) displacement spectra
    ax = fig.add_subplot(gs[0, 1])
    f = np.logspace(3, 11, 400)
    w = 2 * np.pi * f
    cols = [C["verm"], C["orange"], C["green"], C["blue"]]
    for kp, c in zip([P.KP_TACT, 1e-6, 1e-4, 1e-3], cols):
        S = ph.Sx_harmonic(w, P.T_MEAS_K, P.ETA_V, kp)
        lab = (r"$10^{%d}$" % round(np.log10(kp)) if kp != P.KP_TACT
               else r"$T_{\rm act}$ est.")
        ax.loglog(f, S * 1e18, color=c, label=lab)
    ax.axvline(P.F_NV_HZ, color="k", lw=0.7, ls="--")
    ax.axvline(P.F_RES_HZ, color=C["pink"], lw=0.7, ls="-.")
    ax.text(P.F_NV_HZ / 1.3, 1e-4, "NV", fontsize=6.5, ha="right", va="center")
    ax.text(P.F_RES_HZ * 1.3, 1e-4, "res.", fontsize=6.5, ha="left", va="center",
            color=C["pink"])
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"$S_x$ (nm$^2$/Hz)")
    ax.set_ylim(1e-14, 1e5)
    ax.set_yticks([1e-14, 1e-10, 1e-6, 1e-2, 1e2])
    ax.set_xticks([1e3, 1e5, 1e7, 1e9, 1e11])
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.3, ncol=2,
              title=r"$k_p$ (N/m)", title_fontsize=6.2, columnspacing=0.8,
              borderaxespad=0.3)
    plab(ax, "(b)")
    # (c) relaxation halo around one vortex, tilted NV
    ax = fig.add_subplot(gs[0, 2])
    mp = np.load("data/observables_maps.npz")
    x = mp["x"] * 1e9
    Sx = ph.Sx_harmonic(2 * np.pi * P.F_NV_HZ, P.T_MEAS_K, P.ETA_V, 1e-6)
    R1 = 3 * P.GAMMA_E**2 / 4 * Sx * mp["gperp"]
    im = ax.imshow(R1, extent=[x[0], x[-1], x[0], x[-1]], origin="lower",
                   cmap="magma", norm=LogNorm(vmin=1e3, vmax=R1.max()))
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.annotate("", xy=(0.93, 0.12), xytext=(0.75, 0.12), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="w", lw=1))
    ax.text(0.84, 0.16, "NV axis\n(in-plane part)", color="w", fontsize=5.8,
            ha="center", va="bottom", transform=ax.transAxes)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"excess $1/T_1$ (s$^{-1}$)")
    plab(ax, "(c)")
    # (d) 1/T1 vs temperature
    ax = fig.add_subplot(gs[1, 0])
    Ts = np.array(obs["T_K"])
    ax.semilogy(Ts, obs["vortex_invT1_viscous"], color=C["blue"],
                label="one interstitial vortex")
    ax.semilogy(Ts, obs["qp_invT1"], color=C["green"],
                label="quasiparticles (superconducting film)")
    ax.axhline(obs["normal_invT1_at_Tc"], color=C["grey"], ls=":", lw=0.9)
    ax.text(0.7, obs["normal_invT1_at_Tc"] / 1.5, r"normal film at $T_c$",
            fontsize=6.3, color=C["grey"], va="top")
    ax.axhline(P.NV_INTRINSIC_RATE, color="k", ls="--", lw=0.7)
    ax.text(0.7, P.NV_INTRINSIC_RATE * 1.4, "assumed NV floor", fontsize=6.3)
    ax.set_xlabel("temperature (K)"); ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)")
    ax.set_xlim(0.6, 4.2); ax.set_ylim(1e-3, 1e6)
    ax.legend(loc="lower right", fontsize=6.3, bbox_to_anchor=(1.0, 0.0))
    plab(ax, "(d)")
    # (e) 1/T1 vs NV height and lambda
    ax = fig.add_subplot(gs[1, 1])
    sh = np.array(obs["sensitivity"]["h_nm"])
    ax.loglog(sh[:, 0], sh[:, 1], "o-", color=C["blue"], ms=3.5,
              label=r"vs height ($\lambda$ = 50 nm)")
    sl = np.array(obs["sensitivity"]["lam_nm"])
    ax2 = ax.twiny()
    ax2.loglog(sl[:, 0], sl[:, 1], "s--", color=C["orange"], ms=3.5)
    ax.plot([], [], "s--", color=C["orange"], ms=3.5, label=r"vs $\lambda$ ($h$ = 50 nm)")
    ax.legend(loc="lower left", fontsize=6.3)
    ax2.set_xlabel(r"penetration depth $\lambda$ (nm), $h$ = 50 nm",
                   color=C["orange"], fontsize=7)
    ax2.tick_params(axis="x", colors=C["orange"], labelsize=6.5)
    ax.axhline(P.NV_INTRINSIC_RATE, color="k", ls="--", lw=0.7)
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([20, 35, 50, 75, 100, 150]))
    ax.xaxis.set_major_formatter(FixedFormatter(["20", "35", "50", "75", "100", "150"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax2.xaxis.set_major_locator(FixedLocator([35, 50, 70, 100]))
    ax2.xaxis.set_major_formatter(FixedFormatter(["35", "50", "70", "100"]))
    ax2.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("NV height $h$ (nm)"); ax.set_ylabel(r"peak excess $1/T_1$ (s$^{-1}$)")
    ax.set_ylim(3e2, 5e6)
    plab(ax, "(e)", dy=1.12)
    # (f) per-vortex loss vs pinning
    ax = fig.add_subplot(gs[1, 2])
    rows = obs["vs_kp"]
    kp = np.array([r["kp"] for r in rows])
    ax.loglog(kp, [r["invQ_edge"] for r in rows], color=C["verm"],
              label=r"0.2 $\mu$m from strip edge")
    ax.loglog(kp, [r["invQ_center"] for r in rows], color=C["blue"],
              label="strip center")
    ax.axvline(obs["kp_equal_fres"], color=C["pink"], ls="-.", lw=0.8)
    ax.text(obs["kp_equal_fres"] / 1.3, 1.5e-6, r"$f_p$ = 5 GHz",
            fontsize=6.3, color=C["pink"], ha="right", va="top")
    ax.axvline(obs["kp_tact"], color=C["verm"], ls=":", lw=0.8)
    ax.text(obs["kp_tact"] * 1.3, 1.5e-6, r"$T_{\rm act}$ estimate",
            fontsize=6.3, color=C["verm"], ha="left", va="top")
    ax.set_xlabel(r"pinning stiffness $k_p$ (N/m)")
    ax.set_ylabel(r"loss per vortex, $1/Q_v$ at 5 GHz")
    ax.set_ylim(1e-10, 2e-6)
    ax.legend(loc="lower center", fontsize=6.3, bbox_to_anchor=(0.47, 0.12))
    plab(ax, "(f)")
    save(fig, "fig_vortex")



# =====================================================================
def fig_abstract():
    """Fig. 1 / graphical abstract: concept, measured-style maps, FDT bridge."""
    case = "anti15_r1.5"
    m = np.load(f"data/nv_maps_{case}.npz")
    fig = plt.figure(figsize=(FW_2COL, 2.75))
    # ---------- left: cartoon ----------
    ax = fig.add_axes([0.005, 0.03, 0.36, 0.94])
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    # film with two holes
    film_y, film_h = 1.2, 0.9
    ax.add_patch(mpatches.Rectangle((0.3, film_y), 9.4, film_h, fc="#b9c4cf",
                                    ec="#5b6b7a", lw=0.8))
    for hx in (2.3, 7.4):
        ax.add_patch(mpatches.Rectangle((hx - 0.45, film_y), 0.9, film_h,
                                        fc="white", ec="#5b6b7a", lw=0.8))
    ax.text(5.0, film_y - 0.45, r"tantalum film ($d$ = 150 nm) with holes",
            ha="center", fontsize=7)
    # field lines of the hole fluxoid (lossless)
    t = np.linspace(-1, 1, 50)
    for s_ in (-0.25, 0.0, 0.25):
        ax.plot(2.3 + s_ + 0.9 * s_ * (t + 1) ** 2 * 0.25, film_y - 0.2 + 1.7 * (t + 1) / 2 * 1.9,
                color=C["orange"], lw=1.0)
    ax.text(2.3, 4.55, "hole fluxoid\nno core: silent", ha="center", fontsize=7,
            color=C["orange"])
    # interstitial vortex (lossy) with jitter arrows
    vx = 5.0
    for s_ in (-0.25, 0.0, 0.25):
        ax.plot(vx + s_ + 0.9 * s_ * (t + 1) ** 2 * 0.25, film_y - 0.2 + 1.7 * (t + 1) / 2 * 1.9,
                color=C["blue"], lw=1.0)
    ax.add_patch(mpatches.Ellipse((vx, film_y + film_h / 2), 0.35, film_h * 0.95,
                                  fc=C["blue"], ec="none", alpha=0.85))
    ax.annotate("", xy=(vx + 0.9, film_y + film_h / 2), xytext=(vx + 0.3, film_y + film_h / 2),
                arrowprops=dict(arrowstyle="->", color=C["verm"], lw=1.2))
    ax.annotate("", xy=(vx - 0.9, film_y + film_h / 2), xytext=(vx - 0.3, film_y + film_h / 2),
                arrowprops=dict(arrowstyle="->", color=C["verm"], lw=1.2))
    ax.text(vx + 0.2, 4.55, "interstitial vortex\nmoving core: noisy", ha="center",
            fontsize=7, color=C["blue"])
    # membrane with NV
    ax.add_patch(mpatches.FancyBboxPatch((0.6, 3.35), 8.8, 0.55,
                                         boxstyle="round,pad=0.02,rounding_size=0.12",
                                         fc="#e8f3ea", ec="#3c7a4d", lw=0.8))
    ax.text(9.3, 3.62, "diamond membrane", ha="right", va="center", fontsize=6.6,
            color="#2d5f3b")
    for nx_, c_ in ((2.3, C["orange"]), (5.0, C["blue"])):
        ax.plot(nx_, 3.45, "o", ms=4.2, color=C["verm"], zorder=5)
        ax.annotate("", xy=(nx_ + 0.35, 3.45 + 0.5), xytext=(nx_, 3.45),
                    arrowprops=dict(arrowstyle="->", color=C["verm"], lw=1.0))
    ax.text(5.0, 5.75, "Field-cooled tantalum film + NV membrane", fontsize=7.6,
            fontweight="bold", ha="center")
    ax.text(7.4, 2.45, "empty hole", ha="center", fontsize=6.4, color="#5b6b7a")
    # ---------- middle: maps ----------
    x = m["x"] * 1e6
    ext = [x[0], x[-1], x[0], x[-1]]
    a1 = fig.add_axes([0.385, 0.16, 0.19, 0.66])
    lim = np.max(np.abs(m["Bpar"])) * 1e3
    a1.imshow(m["Bpar"] * 1e3, extent=ext, origin="lower", cmap="RdBu_r",
              vmin=-lim, vmax=lim)
    a1.set_title("static field (ODMR)", fontsize=7.4, pad=3)
    a2 = fig.add_axes([0.59, 0.16, 0.19, 0.66])
    a2.imshow(m["rate"], extent=ext, origin="lower", cmap="magma",
              norm=LogNorm(vmin=m["rate"].min(), vmax=m["rate"].max()))
    a2.set_title(r"spin relaxation $1/T_1$", fontsize=7.4, pad=3)
    for a_ in (a1, a2):
        a_.set_xticks([]); a_.set_yticks([])
        for (hx, hy), q in zip(m["holes"], m["hole_quanta"]):
            a_.add_patch(plt.Circle((hx, hy), 0.075, fill=False,
                                    ec="k" if a_ is a1 else "w", lw=0.5))
    fig.text(0.585, 0.06, "TDGL census of a $2\\times2~\\mu$m$^2$ film at 1.5$B_\\Phi$: all trapped flux is bright\n"
             "in the field map; only interstitial vortices are bright in the noise map",
             ha="center", fontsize=6.4)
    # ---------- right: FDT bridge ----------
    ax = fig.add_axes([0.79, 0.03, 0.205, 0.94])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    def box(y, txt, fc):
        ax.add_patch(mpatches.FancyBboxPatch((0.4, y), 9.2, 2.1,
                     boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc,
                     ec="#444", lw=0.7))
        ax.text(5.0, y + 1.05, txt, ha="center", va="center", fontsize=6.8)
    box(7.2, "NV noise at 2.87 GHz\n$1/T_1$ set by $S_x(\\omega_{\\rm NV})$", "#fde7d9")
    box(1.0, "microwave loss\nper vortex, $1/Q_v$", "#dbe9f6")
    ax.annotate("", xy=(5.0, 3.2), xytext=(5.0, 7.1),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
    ax.text(5.3, 5.15, "FDT\n$P=\\dfrac{\\omega^2F^2S_x}{8k_BT}$", ha="left",
            va="center", fontsize=6.8)
    ax.text(5.0, 9.65, "Noise measures loss", ha="center", fontsize=7.6,
            fontweight="bold")
    save(fig, "graphical_abstract")


# =====================================================================
def fig_checks():
    """Fig. 5: independent stochastic checks of FDT, T1 and echo formulas."""
    d = json.load(open("data/langevin_checks.json"))
    fig = plt.figure(figsize=(FW_2COL, 2.35))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.985,
                          top=0.90, bottom=0.19)
    ax = fig.add_subplot(gs[0, 0])
    for key, c, mk, lab in [("harmonic", C["blue"], "o", "harmonic well"),
                            ("double_well", C["verm"], "s", r"double well (4 $k_BT$)")]:
        a = d["fdt"][key]
        w = np.array(a["w"])
        ax.loglog(w, a["P_fdt"], "-", color=c, lw=1.0)
        ax.loglog(w, a["P_drive"], mk, color=c, mfc="none", ms=4, label=lab)
    ax.set_xlabel("drive frequency (reduced units)")
    ax.set_ylabel("absorbed power (reduced)")
    ax.legend(loc="lower right", fontsize=6.3)
    ax.text(0.03, 0.97, "symbols: driven runs\nlines: FDT from noise runs",
            transform=ax.transAxes, fontsize=6.2, va="top")
    plab(ax, "(a)")
    ax = fig.add_subplot(gs[0, 1])
    r = d["t1"]
    sx = [q["sigma_x_nm"] for q in r]
    ax.semilogx(sx, [q["ratio_lin"] for q in r], "o-", color=C["grey"], ms=4,
                label="linear prediction")
    ax.semilogx(sx, [q["ratio_avg"] for q in r], "s-", color=C["blue"], ms=4,
                label="thermal-average prediction")
    ax.axhline(1, color="k", lw=0.6, ls="--")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "2", "5", "10", "20", "50"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1.2, 70)
    ax.set_xlabel(r"thermal excursion $\sigma_x$ (nm)")
    ax.set_ylabel(r"simulated / predicted $1/T_1$")
    ax.set_ylim(0.8, 1.45)
    ax.legend(loc="upper left", fontsize=6.3)
    plab(ax, "(b)")
    ax = fig.add_subplot(gs[0, 2])
    cols = [C["blue"], C["green"], C["verm"]]
    for q, c in zip(d["ramsey"], cols):
        t = np.array(q["t"]) * 1e9
        ax.plot(t / t[-1], q["coh_sim"], color=c, lw=1.2,
                label=r"$\gamma\sigma_B\tau_c$ = %.2g" % q["gamma_sigma_tau"])
        ax.plot(t / t[-1], q["coh_kubo"], "--", color="k", lw=0.7)
    ax.set_xlabel(r"time / $t_{\max}$")
    ax.set_ylabel("Ramsey coherence")
    ax.legend(loc="upper right", fontsize=6.3)
    ax.text(0.97, 0.62, "dashed: Gaussian\n(Kubo) formula", transform=ax.transAxes,
            fontsize=6.2, ha="right", va="top")
    plab(ax, "(c)")
    save(fig, "fig_checks")


def fig_readout():
    """Fig. 6: reading out the pinning and predicting the loss."""
    obs = json.load(open("data/observables.json"))
    inf = json.load(open("data/inference.json"))
    rows = obs["vs_kp"]
    kp = np.array([r["kp"] for r in rows])
    fig = plt.figure(figsize=(FW_2COL, 2.45))
    gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.07, right=0.985,
                          top=0.90, bottom=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(kp, [r["invT1"] for r in rows], color=C["blue"], label=r"$1/T_1$ (2.87 GHz)")
    er = np.array([r["echo_rate"] for r in rows], float)
    ax.loglog(kp, er, color=C["green"], label="echo decay rate")
    ax.axhspan(1e4, 2e7, color=C["green"], alpha=0.08, lw=0)
    ax.axhline(P.NV_INTRINSIC_RATE, color="k", ls="--", lw=0.6)
    ax.set_xlabel(r"pinning stiffness $k_p$ (N/m)")
    ax.set_ylabel(r"rate (s$^{-1}$)")
    ax.set_ylim(1e2, 1e9)
    axb = ax.twinx()
    axb.semilogx(kp, 100 * np.array([r["blur_frac"] for r in rows]),
                 color=C["orange"], ls="-.")
    axb.set_ylabel("image blur (%)", color=C["orange"], labelpad=1)
    axb.tick_params(axis="y", colors=C["orange"])
    axb.set_ylim(0, 100)
    ax.plot([], [], color=C["orange"], ls="-.", label="image blur")
    ax.legend(loc="upper right", fontsize=6.0)
    plab(ax, "(a)")
    ax = fig.add_subplot(gs[0, 1])
    b = inf["baseline"]
    kt = np.array([q["kp_true"] for q in b])
    k50 = np.array([q["kp"][1] for q in b])
    klo = np.array([q["kp"][0] for q in b]); khi = np.array([q["kp"][2] for q in b])
    ax.errorbar(kt, k50, yerr=[k50 - klo, khi - k50], fmt="o", color=C["blue"],
                ms=4, capsize=2, label=r"$k_p$ recovered")
    ax.loglog([1e-9, 3e-3], [1e-9, 3e-3], "k--", lw=0.6)
    wl = inf["wrong_lambda"]
    ax.loglog([q["kp_true"] for q in wl], [q["kp"][1] for q in wl], "s",
              color=C["verm"], mfc="none", ms=4.5, label=r"fit with wrong $\lambda$")
    ax.set_xlabel(r"true $k_p$ (N/m)"); ax.set_ylabel(r"fitted $k_p$ (N/m)")
    ax.legend(loc="upper left", fontsize=6.3)
    plab(ax, "(b)")
    ax = fig.add_subplot(gs[0, 2])
    rq = np.array([q["invQ"][1] / q["invQ_true"] for q in b])
    rlo = np.array([q["invQ"][0] / q["invQ_true"] for q in b])
    rhi = np.array([q["invQ"][2] / q["invQ_true"] for q in b])
    ax.errorbar(kt, rq, yerr=[rq - rlo, rhi - rq], fmt="o", color=C["blue"],
                ms=4, capsize=2, label=r"correct $\lambda$")
    ax.semilogx([q["kp_true"] for q in wl],
                [q["invQ"][1] / q["invQ_true"] for q in wl], "s", color=C["verm"],
                mfc="none", ms=4.5, label=r"$\lambda$ off by 40%")
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"true $k_p$ (N/m)")
    ax.set_ylabel(r"predicted / true $1/Q_v$")
    ax.legend(loc="lower left", fontsize=6.3)
    plab(ax, "(c)")
    save(fig, "fig_readout")


# =====================================================================
def _psi_panel(ax, npz, holes, D, inter=None, title=None):
    z = np.load(npz)
    s = z["sites_um"] if "sites_um" in z else z["sites"]
    tri = z["elements"]
    ax.tripcolor(s[:, 0], s[:, 1], tri, np.abs(z["psi"]), cmap="viridis",
                 vmin=0, vmax=1, shading="gouraud", rasterized=True)
    for hx, hy in holes:
        ax.add_patch(plt.Circle((hx, hy), D / 2, fc="white", ec="k", lw=0.4))
    if inter is not None and len(inter):
        inter = np.asarray(inter)
        ax.plot(inter[:, 0], inter[:, 1], "o", mfc="none", mec=C["orange"],
                ms=6, mew=1.2)
    ax.set_xlim(s[:, 0].min(), s[:, 0].max()); ax.set_ylim(s[:, 1].min(), s[:, 1].max())
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.4, pad=3)


def fig_maps():
    """Fig. 3: TDGL census, field map, noise map, census vs cooling field."""
    case = "anti15_r1.5"
    cz = json.load(open(f"data/fieldcool_sweep/{case}.json"))
    m = np.load(f"data/nv_maps_{case}.npz")
    summ = json.load(open("data/nv_maps.json"))
    fig = plt.figure(figsize=(FW_2COL, 2.45))
    gs = fig.add_gridspec(1, 4, wspace=0.34, left=0.015, right=0.985,
                          top=0.86, bottom=0.18, width_ratios=[1, 1, 1, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    _psi_panel(ax, f"data/fieldcool_sweep/{case}.npz", cz["hole_centers_um"],
               cz["hole_diam_um"], cz["interstitial_um"],
               r"TDGL, $|\psi|$ at 1.5 $B_\Phi$")
    for (hx, hy), q in zip(cz["hole_centers_um"], cz["hole_quanta"]):
        if q > 0:
            ax.text(hx, hy, str(q), ha="center", va="center", fontsize=5.5)
    plab(ax, "(a)", dx=0.02)
    x = m["x"] * 1e6
    ext = [x[0], x[-1], x[0], x[-1]]
    ax = fig.add_subplot(gs[0, 1])
    lim = np.max(np.abs(m["Bpar"])) * 1e3
    im = ax.imshow(m["Bpar"] * 1e3, extent=ext, origin="lower", cmap="RdBu_r",
                   vmin=-lim, vmax=lim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("static field along NV axis", fontsize=7.4, pad=3)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.04)
    cb.set_label("mT", fontsize=6.5, labelpad=1); cb.ax.tick_params(labelsize=6)
    plab(ax, "(b)", dx=0.02)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(m["rate"], extent=ext, origin="lower", cmap="magma",
                   norm=LogNorm(vmin=m["rate"].min(), vmax=m["rate"].max()))
    for (hx, hy), q in zip(m["holes"], m["hole_quanta"]):
        ax.add_patch(plt.Circle((hx, hy), 0.075, fill=False, ec="w", lw=0.5,
                                ls="-" if q else ":"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"relaxation $1/T_1$ at 2 K", fontsize=7.4, pad=3)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.04)
    cb.set_label(r"s$^{-1}$", fontsize=6.5, labelpad=1); cb.ax.tick_params(labelsize=6)
    plab(ax, "(c)", dx=0.02)
    ax = fig.add_subplot(gs[0, 3])
    for pre, c, mk, lab in [("anti15", C["blue"], "o", "150 nm holes"),
                            ("anti20", C["green"], "s", "200 nm holes"),
                            ("bare", C["verm"], "^", "no holes")]:
        ks = sorted([k for k in summ if k.startswith(pre + "_r") and
                     k.count("_") == 1], key=lambda k: summ[k]["B_over_Bphi"])
        r = [summ[k]["B_over_Bphi"] for k in ks]
        ax.plot(r, [summ[k]["n_interstitial"] for k in ks], mk + "-", color=c,
                ms=4, label=lab)
        if pre != "bare":
            ax.plot(r, [summ[k]["hole_quanta_total"] for k in ks], mk + ":",
                    color=c, ms=3.5, mfc="none")
    ax.set_xlabel(r"cooling field $B/B_\Phi$")
    ax.set_ylabel("number in $2\\times2~\\mu$m$^2$")
    ax.legend(loc="upper left", fontsize=6.2)
    ax.text(0.03, 0.70, "solid: interstitial (lossy)\ndotted: in holes (silent)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0)
    plab(ax, "(d)")
    save(fig, "fig_maps")


def fig_tdgl():
    """Fig. 4: TDGL microwave drive, hole fluxoids versus interstitial vortices."""
    import glob as _g
    R = {os.path.basename(f)[:-5]: json.load(open(f))
         for f in _g.glob("data/ac_loss/*.json")}
    fig = plt.figure(figsize=(FW_2COL, 2.45))
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.02, right=0.985,
                          top=0.86, bottom=0.19, width_ratios=[1.0, 1.15, 1.3])
    ax = fig.add_subplot(gs[0, 0])
    k = "anti15_B12_I1000"
    z = np.load(f"data/ac_loss/{k}.npz")
    from tdgl_common import hole_centers
    _psi_panel(ax, f"data/ac_loss/{k}.npz", hole_centers(0.5), 0.15,
               R[k]["interstitial_um"], r"driven strip, 1.5 $B_\Phi$")
    for (hx, hy), q in zip(hole_centers(0.5), R[k]["hole_quanta"]):
        if q > 0:
            ax.text(hx, hy, str(q), ha="center", va="center", fontsize=5.5)
    ax.annotate("", xy=(0.35, 0.74), xytext=(-0.35, 0.74),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
    ax.text(0.0, 0.66, r"$I(t)$, 5 GHz", color="k", fontsize=6.3, ha="center", va="top")
    plab(ax, "(a)", dx=0.02)
    ax = fig.add_subplot(gs[0, 1])
    per = 1.0 / P.F_RES_HZ / P.TAU_0_S
    for kk, c, lab in [("anti15_B0_I1000", C["grey"], "no flux"),
                       ("anti15_B8_I1000", C["orange"], "holes filled"),
                       ("anti15_B12_I1000", C["blue"], "holes + 4 interstitial")]:
        if kk not in R:
            continue
        zz = np.load(f"data/ac_loss/{kk}.npz")
        t = zz["t"] / per
        m_ = (t >= 1.5) & (t <= 2.5)
        ax.plot(zz["I"][m_] / 1e3, zz["V"][m_] * R[kk]["V0_V"] * 1e6, color=c,
                lw=1.0, label=lab)
    ax.set_xlabel("drive current (mA)")
    ax.set_ylabel(r"probe voltage ($\mu$V)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=5.8, handlelength=1.0, columnspacing=0.8)
    plab(ax, "(b)")
    ax = fig.add_subplot(gs[0, 2])
    F = P.PHI0 * 1e-3 / (P.FILM_W * 1e-6)
    Pff = F**2 / (2 * P.ETA_V)
    g = lambda k: R[k + "_I1000"]["P_W"]
    rows = [("8 fluxoids in holes\n(1.0 $B_\\Phi$ minus 0)", g("anti15_B8") - g("anti15_B0"), 0,
             sum(R["anti15_B8_I1000"]["hole_quanta"])),
            ("4 interstitial added\n(1.5 $B_\\Phi$ minus 1.0 $B_\\Phi$)", g("anti15_B12") - g("anti15_B8"),
             R["anti15_B12_I1000"]["n_between_probes"], 0),
            ("plain film\n(0.5 $B_\\Phi$ minus 0)", g("bare_B4") - g("bare_B0"),
             R["bare_B4_I1000"]["n_between_probes"], 0)]
    xs = np.arange(len(rows))
    ax.bar(xs, [r[1] * 1e12 for r in rows], color=C["blue"], width=0.55,
           label="TDGL excess power")
    ax.plot(xs, [r[2] * Pff * 1e12 for r in rows], "_", color=C["verm"],
            ms=22, mew=2, label=r"$N_{\rm int}F^2/2\eta_v$ (Bardeen-Stephen)")
    for xi, r in zip(xs, rows):
        ax.text(xi, max(r[1], r[2] * Pff) * 1e12 + 20,
                f"{r[2]} interstitial" + (f"\n{r[3]} in holes" if r[3] else ""),
                ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_ylabel("excess dissipation (pW)")
    ax.set_ylim(0, max(max(r[1], r[2] * Pff) for r in rows) * 1e12 * 1.45)
    ax.legend(loc="upper left", fontsize=6.0, markerscale=0.45)
    plab(ax, "(c)")
    save(fig, "fig_tdgl")


# =====================================================================
def fig_supp():
    """Supplementary figures S1-S3."""
    # S1: census grid
    names = [[f"bare_r{r:.1f}" for r in (0.5, 1.0, 1.5, 2.0)] + [None],
             [f"anti15_r{r:.1f}" for r in (0.5, 1.0, 1.5, 2.0, 2.5)],
             [f"anti20_r{r:.1f}" for r in (0.5, 1.0, 1.5, 2.0, 2.5)]]
    fig, axs = plt.subplots(3, 5, figsize=(FW_2COL, 4.6))
    for i, row in enumerate(names):
        for j, nm in enumerate(row):
            ax = axs[i, j]
            if nm is None or not os.path.exists(f"data/fieldcool_sweep/{nm}.json"):
                ax.axis("off"); continue
            cz = json.load(open(f"data/fieldcool_sweep/{nm}.json"))
            _psi_panel(ax, f"data/fieldcool_sweep/{nm}.npz", cz["hole_centers_um"],
                       cz["hole_diam_um"] or 0.0, cz["interstitial_um"])
            for (hx, hy), q in zip(cz["hole_centers_um"], cz["hole_quanta"]):
                if q > 0:
                    ax.text(hx, hy, str(q), ha="center", va="center", fontsize=5)
            hd = {"bare": "no holes", "anti15": "150 nm", "anti20": "200 nm"}[nm.split("_")[0]]
            ax.set_title(f"{hd}, {cz['B_over_Bphi']:.1f}$B_\\Phi$: "
                         f"{len(cz['interstitial_um'])} int.", fontsize=6.3, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.08, hspace=0.22)
    save(fig, "figS1_census", supp=True)
    # S2: probe voltage traces
    import glob as _g
    fig, ax = plt.subplots(1, 2, figsize=(FW_2COL, 2.3))
    per = 1.0 / P.F_RES_HZ / P.TAU_0_S
    cols = [C["grey"], C["orange"], C["blue"], C["green"], C["verm"], C["pink"], C["sky"]]
    for c, f in zip(cols, sorted(_g.glob("data/ac_loss/*_I1000.npz"))):
        z = np.load(f); js = json.load(open(f[:-4] + ".json"))
        t = z["t"] / per
        m_ = t >= 0.5
        ax[0].plot(t[m_], z["V"][m_] * js["V0_V"] * 1e6, color=c, lw=0.8,
                   label=js["case"].replace("_", " "))
        ax[1].plot(z["I"][m_] / 1e3, z["V"][m_] * js["V0_V"] * 1e6, color=c, lw=0.8)
    ax[0].set_xlabel("time (drive periods)"); ax[0].set_ylabel(r"probe voltage ($\mu$V)")
    ax[0].legend(fontsize=5.8, ncol=2, loc="lower left")
    ax[1].set_xlabel("drive current (mA)"); ax[1].set_ylabel(r"probe voltage ($\mu$V)")
    plab(ax[0], "(a)"); plab(ax[1], "(b)")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.18, wspace=0.3)
    save(fig, "figS2_ac_traces", supp=True)
    # S3: noise spectra of the Brownian runs versus analytic forms
    d = json.load(open("data/langevin_checks.json"))
    fig, ax = plt.subplots(1, 2, figsize=(FW_2COL, 2.2))
    a = d["fdt"]["harmonic"]
    w = np.array(a["wS"]); S = np.array(a["S"])
    ax[0].loglog(w, S, color=C["blue"], lw=0.8, label="Brownian run (Welch)")
    ax[0].loglog(w, 4 / (1 + w**2), "k--", lw=0.8, label=r"$4/(1+\omega^2)$")
    ax[0].set_xlim(3e-2, 3e2); ax[0].set_ylim(1e-5, 10)
    ax[0].set_xlabel(r"$\omega$ (reduced)"); ax[0].set_ylabel(r"$S_x$ (reduced)")
    ax[0].legend(fontsize=6.3, loc="lower left"); plab(ax[0], "(a)")
    a = d["fdt"]["double_well"]
    ax[1].loglog(a["wS"], a["S"], color=C["verm"], lw=0.8, label=r"double well, barrier 4 $k_BT$")
    ax[1].set_xlim(3e-3, 3e1)
    ax[1].set_xlabel(r"$\omega$ (reduced)"); ax[1].set_ylabel(r"$S_x$ (reduced)")
    ax[1].legend(fontsize=6.3, loc="lower left"); plab(ax[1], "(b)")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.2, wspace=0.3)
    save(fig, "figS3_spectra", supp=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["abstract", "vortex", "maps", "tdgl", "checks", "readout"]
    for w_ in which:
        globals()[f"fig_{w_}"]()
