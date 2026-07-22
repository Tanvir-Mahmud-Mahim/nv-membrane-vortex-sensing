"""Publication figures for the manuscript (IEEE two-column).

Layout rules used throughout:
  - panel labels (a), (b), ... sit OUTSIDE the axes, top-left, bold 9 pt
  - colorbars live in dedicated gridspec slots, never overlapping neighbors
  - legends are placed where no data or labels can collide
  - fonts: 8 pt labels, 7 pt ticks, ~6.8 pt legends (IEEE column scale)
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from figstyle import C, ORDER, CMAP_SEQ, CMAP_DIV, FW_2COL, panel_label
import params as P

exec(open("scripts/03_nv_observables.py").read().split('if __name__')[0])

RES = json.load(open("data/results.json"))


def plab(ax, s, dx=-0.02, dy=1.06):
    """Bold panel label outside the axes, top-left."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right")


def psi_panel(ax, tag, title, scalebar=False):
    d = load_fieldcool(tag)
    gx, gy, A = d["grid_x"], d["grid_y"], d["psi_grid"]
    im = ax.imshow(A, extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower",
                   cmap=CMAP_SEQ, vmin=0, vmax=1)
    for (hx, hy) in d["hole_centers"]:
        ax.add_patch(plt.Circle((hx, hy), d["hole_diam"] / 2, fill=False,
                                ec="white", lw=0.8))
    for (hx, hy), o in zip(d["hole_centers"], d["occ_winding"]):
        if o > 0:
            ax.plot(hx, hy, marker="+", color="white", ms=5, mew=1.1)
            if o > 1:
                ax.text(hx + 0.06, hy + 0.05, str(int(o)), color="white",
                        fontsize=6.5)
    if len(d["vortex_pos"]):
        ax.plot(d["vortex_pos"][:, 0], d["vortex_pos"][:, 1], "o",
                mfc="none", mec=C["orange"], ms=7, mew=1.3)
    if scalebar:
        ax.plot([0.25, 0.75], [-0.68, -0.68], color="white", lw=2.0)
        ax.text(0.5, -0.60, "500 nm", color="white", fontsize=6.5,
                ha="center")
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8)
    ax.set_title(title, fontsize=7.6, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    return im


# ================= Fig. 2: sensing target =================
def fig_target():
    fig = plt.figure(figsize=(FW_2COL, 4.15))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05],
                          hspace=0.42, wspace=0.16,
                          left=0.055, right=0.985, top=0.92, bottom=0.10)
    tags = [("bare_B4", "bare, 4.14 mT"),
            ("bare_B8", "bare, 8.27 mT"),
            ("anti15_B8", "antidot, $B_\\Phi$ = 8.27 mT"),
            ("anti15_B12", "antidot, 12.4 mT (1.5$B_\\Phi$)")]
    for i, (tag, ti) in enumerate(tags):
        ax = fig.add_subplot(gs[0, i])
        im = psi_panel(ax, tag, ti, scalebar=(i == 0))
        panel_label(ax, f"({'abcd'[i]})")
    # dedicated horizontal colorbar centered below the map row
    cax = fig.add_axes([0.30, 0.545, 0.40, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"order parameter $|\psi|/\psi_0$", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6.5)

    gsb = gs[1, :].subgridspec(1, 2, wspace=0.30)
    # (e) IV
    ax = fig.add_subplot(gsb[0])
    iv = RES["iv"]
    I = np.array(iv["I_uA"])[2:]
    ax.plot(I, 1e3 * np.array(iv["dv_bare"])[2:], "o-", color=C["blue"],
            label="bare film", ms=3.5)
    ax.plot(I, 1e3 * np.array(iv["dv_anti"])[2:], "s-", color=C["orange"],
            label="antidot lattice", ms=3.5)
    ax.set_xlabel(r"drive current ($\mu$A)", labelpad=1.5)
    ax.set_ylabel(r"vortex voltage ($10^{-3}\,V_0$)")
    ax.set_ylim(-0.08, 2.25)
    ax.legend(loc="upper left", bbox_to_anchor=(0.075, 0.985),
              handlelength=1.6, borderaxespad=0.0)
    ax.annotate(f"{iv['suppression_at_max']:.0f}$\\times$ suppression",
                xy=(I[-1] - 4, 1e3 * iv["dv_anti"][-1] + 0.05),
                xytext=(235, 0.75), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C["ink"]))
    panel_label(ax, "(e)")

    # (f) microwave drive
    ax = fig.add_subplot(gsb[1])
    mwb = np.load("data/mw_bare.npz"); mwa = np.load("data/mw_anti15.npz")
    t0 = P.TAU_0_S * 1e12 * 1e-3  # tau -> ns
    ax.plot(mwb["t"] * t0, 1e3 * mwb["I"] * mwb["V"] / 150,
            color=C["blue"], lw=0.9, label="bare film")
    ax.plot(mwa["t"] * t0, 1e3 * mwa["I"] * mwa["V"] / 150,
            color=C["orange"], lw=0.9, label="antidot lattice", alpha=0.95)
    ax.set_xlabel("time (ns)", labelpad=1.5)
    ax.set_ylabel("inst. dissipation (norm.)")
    ax.set_ylim(-3.4, 5.6)
    ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0.4,
              ncols=2, columnspacing=0.8, fontsize=6.7)
    ax.text(0.09, 0.035,
            f"2.87 GHz drive; cycle-averaged ratio "
            f"{RES['mw_suppression']:.1f}$\\times$",
            transform=ax.transAxes, fontsize=6.8, color=C["ink"])
    panel_label(ax, "(f)")
    fig.savefig("manuscript/figures/fig_target.pdf")
    plt.close(fig)


# ================= Fig. 3: stray field =================
def fig_field():
    fig = plt.figure(figsize=(FW_2COL, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1.0, 1.0, 1.62],
                          wspace=0.40, left=0.06, right=0.925,
                          top=0.885, bottom=0.185)
    # (a) Pearl profiles
    ax = fig.add_subplot(gs[0])
    d = np.load("data/pearl_profiles.npz")
    for i, so in enumerate([10, 25, 50, 100]):
        ax.plot(d["rho_um"] * 1e3, d[f"so{so}"], color=ORDER[i],
                label=f"{so} nm")
    ax.set_xlabel("lateral distance (nm)", labelpad=1.5)
    ax.set_ylabel(r"$B_z$ (mT)")
    ax.set_xlim(0, 400); ax.set_ylim(0, 10.4)
    ax.legend(title="membrane standoff", loc="upper right",
              handlelength=1.4, title_fontsize=6.8, fontsize=6.8,
              borderaxespad=0.3)
    plab(ax, "(a)", dx=-0.155)

    # (b,c) maps sharing one colorbar in its own gridspec slot
    vmax, maps = 0.0, []
    for tag in ["bare_B4", "anti15_B8"]:
        m = np.load(f"data/nvmap_{tag}.npz")
        maps.append(m)
        vmax = max(vmax, float(np.nanmax(np.abs(m["Bz_mT"]))))
    map_axes = []
    for j, (m, ti) in enumerate(zip(maps, ["bare film, 4.14 mT",
                                           "antidot, $B_\\Phi$"])):
        ax = fig.add_subplot(gs[1 + j])
        im = ax.imshow(m["Bz_mT"], extent=[m["xs"][0], m["xs"][-1]] * 2,
                       origin="lower", cmap=CMAP_DIV, vmin=-vmax, vmax=vmax)
        ax.set_title(ti, fontsize=7.4, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        panel_label(ax, f"({'bc'[j]})")
        map_axes.append(ax)
        if j == 0:
            ax.plot([-0.6, 0.6], [0, 0], ls=":", color="k", lw=0.9)
    p0 = map_axes[0].get_position(); p1 = map_axes[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, 0.105, (p1.x1 - p0.x0) - 0.04, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"$B_z$ (mT)", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6.5)

    # (d) line cut with NV-shift axis
    ax = fig.add_subplot(gs[3])
    m = maps[0]
    xs = m["xs"]
    row = m["Bz_mT"][len(xs) // 2]
    ax.plot(xs * 1e3, row, color=C["blue"], lw=1.4, label="exact (TDGL)")
    dfc = load_fieldcool("bare_B4")
    model = np.zeros_like(xs)
    for (vx, vy) in dfc["vortex_pos"]:
        model += pearl_bz(np.hypot(xs - vx, vy), Z_NV_UM)
    ax.plot(xs * 1e3, model - np.mean(model - row), ls="--", lw=1.3,
            color=C["orange"], label="Pearl kernel")
    ax.set_xlabel("position (nm)", labelpad=1.5)
    ax.set_ylabel(r"$B_z$ (mT)", labelpad=1)
    ax.set_ylim(-3.2, 10.2)
    ax2 = ax.secondary_yaxis(
        "right", functions=(lambda b: GAMMA_NV * PROJ * b * 1e-3 / 1e6,
                            lambda f: f * 1e6 / (GAMMA_NV * PROJ) * 1e3))
    ax2.set_ylabel("NV shift (MHz)", fontsize=7.5, labelpad=2)
    ax2.tick_params(labelsize=6.5)
    ax.tick_params(right=False)
    ax.legend(loc="upper left", handlelength=1.5, fontsize=6.8,
              borderaxespad=0.3)
    plab(ax, "(d)", dx=-0.155)
    fig.savefig("manuscript/figures/fig_field.pdf")
    plt.close(fig)


# ================= Fig. 4: reconstruction =================
def fig_recon():
    rec = json.load(open("data/reconstruction.json"))
    fig = plt.figure(figsize=(FW_2COL, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.02, 1.25, 1.15, 1.15],
                          wspace=0.52, left=0.045, right=0.99,
                          top=0.885, bottom=0.185)
    # (a) example noisy map + fit
    ax = fig.add_subplot(gs[0])
    d = np.load("data/recon_example.npz")
    v = float(np.nanmax(np.abs(d["Bn"])))
    ax.imshow(d["Bn"], extent=[d["xs"][0], d["xs"][-1]] * 2, origin="lower",
              cmap=CMAP_DIV, vmin=-v, vmax=v)
    ax.plot(d["truth"][:, 0], d["truth"][:, 1], "o", mfc="none",
            mec=C["ink"], ms=7, mew=1.1, label="TDGL truth")
    ax.plot(d["fit"][:, 0], d["fit"][:, 1], "x", color=C["green"], ms=5,
            mew=1.4, label="fit")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.015), ncols=2,
              fontsize=6.8, handlelength=1.0, handletextpad=0.4,
              columnspacing=0.9, frameon=False)
    plab(ax, "(a)", dx=0.07, dy=1.02)

    # (b) localization vs standoff
    ax = fig.add_subplot(gs[1])
    for i, tpx in enumerate([0.1, 1.0, 10.0]):
        so = [r["standoff_nm"] for r in rec["scan"]
              if r["t_px_ms"] == tpx and r["mode"] == "membrane"]
        err = [r["loc_err_nm"] for r in rec["scan"]
               if r["t_px_ms"] == tpx and r["mode"] == "membrane"]
        crb = [r["crb_nm"] for r in rec["scan"]
              if r["t_px_ms"] == tpx and r["mode"] == "membrane"]
        ax.plot(so, crb, color=ORDER[i], lw=1.0, ls="--")
        ax.plot(so, err, "o", color=ORDER[i], ms=4, label=f"{tpx:g} ms")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=C["grey"], ls="none", marker="o",
                       ms=4),
                Line2D([], [], color=C["grey"], lw=1.0, ls="--")]
    labels += ["fit (pixel-limited)", "CRB (stat. bound)"]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("standoff (nm)", labelpad=1.5)
    ax.set_ylabel("localization error (nm)", labelpad=1)
    ax.set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(8e-3, 200)
    leg = ax.legend(handles, labels, title="pixel time", title_fontsize=7,
                    fontsize=7, loc="center left", handlelength=1.3,
                    borderaxespad=0.3, labelspacing=0.35, frameon=True,
                    framealpha=0.92, edgecolor="none", facecolor="white")
    plab(ax, "(b)", dx=-0.20)

    # (c) occupancy accuracy vs standoff
    ax = fig.add_subplot(gs[2])
    occ = rec["occupancy_vs_standoff"]
    for i, (tpx, lab) in enumerate([(0.01, "10 $\\mu$s"), (0.1, "0.1 ms")]):
        so = [o["standoff_nm"] for o in occ if o["t_px_ms"] == tpx]
        a = [100 * o["acc"] for o in occ if o["t_px_ms"] == tpx]
        ax.plot(so, a, ["o-", "s--"][i], color=ORDER[i], ms=3.5, label=lab)
    ax.set_xlabel("standoff (nm)", labelpad=1.5)
    ax.set_ylabel("occupancy accuracy (%)", labelpad=1)
    ax.set_ylim(90, 101.4)
    ax.set_yticks([90, 95, 100])
    ax.legend(title="pixel time", title_fontsize=6.8, fontsize=6.8,
              loc="lower left", handlelength=1.5, borderaxespad=0.4)
    ax.text(160, 97.7, "100% at every\nstudied standoff", fontsize=7,
            ha="center", color=C["grey"])
    plab(ax, "(c)", dx=-0.20)

    # (d) frame time budget
    ax = fig.add_subplot(gs[3])
    npx = np.array([51, 101, 201, 301]) ** 2
    t_px = 0.7e-3
    ax.loglog(npx, npx * t_px, "o-", color=C["blue"], ms=3.5,
              label="confocal ($\\times$1)")
    ax.loglog(npx, npx * t_px / 7, "s-", color=C["orange"], ms=3.5,
              label="membrane ($\\times$7)")
    ax.set_xlabel("pixels per frame", labelpad=1.5)
    ax.set_ylabel("frame time (s)", labelpad=1)
    ax.legend(handlelength=1.5, fontsize=6.8, loc="upper left",
              borderaxespad=0.3)
    plab(ax, "(d)", dx=-0.20)
    fig.savefig("manuscript/figures/fig_recon.pdf")
    plt.close(fig)


# ================= Fig. 5: dynamics =================
def fig_dynamics():
    fig = plt.figure(figsize=(FW_2COL, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.0, 1.15, 1.15],
                          wspace=0.52, left=0.06, right=0.99,
                          top=0.885, bottom=0.185)
    # (a) spectra
    ax = fig.add_subplot(gs[0])
    d = np.load("data/langevin_spectra.npz")
    f = d["f_Hz"]
    ax.loglog(f, d["antidot"], color=C["blue"])
    ax.loglog(f, d["natural"], color=C["orange"])
    ax.text(1.5e4, 1.1e-13, "natural (weak)", fontsize=7,
            color=C["orange"])
    ax.text(1.5e4, 6.5e-18, "antidot pinned", fontsize=7,
            color=C["blue"], va="top")
    ax.set_ylim(1e-21, 3e-13)
    ax.axvspan(1e3, 1e6, color=C["grey"], alpha=0.13, lw=0)
    ax.text(3.2e4, 2.6e-21, "$T_2$ band", fontsize=7, ha="center",
            color=C["grey"])
    ax.axvline(2.87e9, color=C["grey"], lw=0.8, ls=":")
    ax.text(2.87e9, 4.5e-13, "$T_1$ (2.87 GHz)", fontsize=7, ha="center",
            va="bottom", color=C["grey"])
    ax.set_xlabel("frequency (Hz)", labelpad=1.5)
    ax.set_ylabel(r"$S_B$ (T$^2$/Hz)", labelpad=1)
    plab(ax, "(a)", dx=-0.175)

    # (b) T1/T2 bars
    ax = fig.add_subplot(gs[1])
    nz = RES["noise"]
    T1 = [nz[c]["T1_us"] for c in ["antidot", "natural"]]
    T2 = [nz[c]["T2phi_us"] for c in ["antidot", "natural"]]
    x = np.arange(2)
    ax.bar(x - 0.19, T1, 0.34, color=C["blue"], label="$T_1$")
    ax.bar(x + 0.19, T2, 0.34, color=C["orange"], label="$T_{2\\varphi}$")
    ax.set_yscale("log")
    ax.set_ylim(2.5e-3, 6e4)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(x, ["antidot\npinned", "natural\n(weak)"])
    ax.set_ylabel(r"NV time limit ($\mu$s)", labelpad=1)
    for xi, v in zip(x - 0.19, T1):
        ax.text(xi, v * 1.4, f"{v:.0f}", ha="center", fontsize=6.2)
    for xi, v in zip(x + 0.19, T2):
        s = f"{v:.0f}" if v >= 1 else f"{v:.3f}"
        ax.text(xi, v * 1.4, s, ha="center", fontsize=6.2)
    ax.legend(handlelength=1.1, fontsize=6.6, loc="upper right",
              borderaxespad=0.25, handletextpad=0.4, ncols=2,
              columnspacing=0.7)
    plab(ax, "(b)", dx=-0.22)

    # (c) telegraph trace (short window so hops are visible)
    ax = fig.add_subplot(gs[2])
    d = np.load("data/covariance.npz")
    n = 400  # 4 ms
    ax.plot(d["t_short"][:n] * 1e3, d["b1_mT"][:n], color=C["blue"], lw=1.0,
            drawstyle="steps-post")
    ax.set_xlabel("time (ms)", labelpad=1.5)
    ax.set_ylabel(r"$B_z$ at NV$_1$ (mT)", labelpad=1)
    ax.set_title("vortex telegraph hops", fontsize=7.4, pad=3)
    ax.set_ylim(3.4, 8.3)
    ax.text(0.5, 0.045, "2 kHz hop rate,  $\\Delta B$ = 3.1 mT",
            transform=ax.transAxes, fontsize=7, ha="center", va="bottom",
            color=C["grey"])
    plab(ax, "(c)", dx=-0.185)

    # (d) covariance
    ax = fig.add_subplot(gs[3])
    ax.plot(d["lags_s"] * 1e3, d["cov"] / d["theory"][0], "o",
            color=C["green"], ms=3, label="simulated")
    ax.plot(d["lags_s"] * 1e3, d["theory"] / d["theory"][0],
            color=C["ink"], lw=1.1, label=r"$e^{-2 r \tau}$ theory")
    ax.set_xlabel("lag (ms)", labelpad=1.5)
    ax.set_ylabel("normalized covariance", labelpad=1)
    ax.set_ylim(-0.45, 1.35)
    ax.legend(handlelength=1.4, fontsize=6.8, loc="upper right",
              borderaxespad=0.3)
    plab(ax, "(d)", dx=-0.185)
    fig.savefig("manuscript/figures/fig_dynamics.pdf")
    plt.close(fig)


# ================= Fig. 1: graphical abstract =================
def fig_ga():
    fig = plt.figure(figsize=(FW_2COL, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.55, 0.92],
                          wspace=0.28, left=0.015, right=0.985,
                          top=0.825, bottom=0.13)

    # ---- left: target ----
    ax = fig.add_subplot(gs[0])
    psi_panel(ax, "anti15_B8", "")
    ax.set_title("TARGET\nvortex matter in a Ta film\n(TDGL, measured parameters)",
                 fontsize=7.2, pad=5)
    ax.text(0.5, -0.085, "filled holes (+), one empty hole",
            transform=ax.transAxes, fontsize=6.6, ha="center", va="top")

    # ---- center: schematic ----
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, 10)
    ax.set_title("PLATFORM\npick-and-place diamond micromembrane NV sensor",
                 fontsize=7.2, pad=5)
    # substrate and film
    ax.add_patch(mpatches.Rectangle((1.3, 0.6), 7.4, 1.0, fc="#dcdcdc",
                                    ec="none"))
    ax.text(5.0, 1.1, "sapphire substrate", fontsize=6.4, ha="center",
            va="center", color="#555555")
    ax.add_patch(mpatches.Rectangle((1.3, 1.6), 7.4, 1.0, fc="#9fb8c9",
                                    ec="none"))
    ax.text(8.5, 2.1, "Ta film", fontsize=6.4, ha="right", va="center")
    for hx in [2.3, 4.1, 5.9]:
        ax.add_patch(mpatches.Rectangle((hx, 1.6), 0.5, 1.0, fc="white",
                                        ec="#7a95a8", lw=0.5))
    ax.annotate("pinning holes", xy=(2.55, 1.7), xytext=(1.35, -0.35),
                fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#555555"))
    # vortices and stray field
    for vx in [3.35, 6.9]:
        ax.plot([vx], [2.1], marker="o", color=C["verm"], ms=4)
        ax.annotate("", xy=(vx, 3.9), xytext=(vx, 2.35),
                    arrowprops=dict(arrowstyle="->", color=C["verm"],
                                    lw=1.0))
    ax.text(7.3, 3.2, "vortex\nstray field", fontsize=6.4, color=C["verm"],
            va="center")
    # standoff bracket in the gap between film top (2.6) and membrane (4.3)
    ax.annotate("", xy=(1.05, 2.6), xytext=(1.05, 4.3),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#555555"))
    ax.text(0.9, 3.45, "25 nm\nstandoff", fontsize=6.2, ha="right",
            va="center", color="#555555")
    # membrane with NVs
    ax.add_patch(mpatches.Rectangle((1.7, 4.3), 6.6, 1.0, fc="#cfc7ee",
                                    ec="#8d80c9", lw=0.7))
    ax.text(8.2, 4.8, "diamond\nmembrane", fontsize=6.4, ha="right",
            va="center")
    for nx in [2.7, 4.6, 6.5]:
        ax.plot([nx], [4.5], marker="v", color=C["pink"], ms=4.5,
                mec="none")
    ax.annotate("shallow NV centers (6 nm deep)", xy=(2.75, 4.52),
                xytext=(1.6, 6.15), fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#555555"))
    # laser cone
    ax.add_patch(mpatches.Polygon([[4.05, 9.4], [5.15, 9.4], [4.6, 5.45]],
                                  closed=True, fc="#7fbf7f", alpha=0.35,
                                  ec="none"))
    ax.text(5.45, 8.6, "520 nm excitation", fontsize=6.4, color="#3a7d3a")
    ax.text(5.45, 7.9, "PL out, $\\times$7 collection gain",
            fontsize=6.4, color="#3a7d3a")

    # ---- right: outcome ----
    ax = fig.add_subplot(gs[2])
    m = np.load("data/nvmap_anti15_B8.npz")
    v = float(np.nanmax(np.abs(m["Bz_mT"])))
    ax.imshow(m["Bz_mT"], extent=[m["xs"][0], m["xs"][-1]] * 2,
              origin="lower", cmap=CMAP_DIV, vmin=-v, vmax=v)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("OUTCOME\nNV-plane field map + inversion\n(census, pinning, dissipation)",
                 fontsize=7.2, pad=5)
    ax.text(0.5, -0.085,
            "13 nm localization  |  100% occupancy readout\n"
            "$T_1 \\rightarrow$ drag $\\eta$   |   "
            "$T_2 \\rightarrow$ pinning $k_p$",
            transform=ax.transAxes, fontsize=6.6, ha="center", va="top")
    fig.savefig("manuscript/figures/graphical_abstract.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("manuscript/figures", exist_ok=True)
    for fn in [fig_target, fig_field, fig_recon, fig_dynamics, fig_ga]:
        try:
            fn(); print("ok", fn.__name__, flush=True)
        except Exception as e:
            print("FAIL", fn.__name__, repr(e), flush=True)
