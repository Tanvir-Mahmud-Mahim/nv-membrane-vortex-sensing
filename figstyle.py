"""Shared publication figure style (IEEE two-column, colorblind-safe,
Times New Roman text and math, TrueType fonts embedded in PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, os
from matplotlib import font_manager as _fm

# Register Times New Roman if the TrueType files are available locally
# (they are not redistributed with this code). Fallbacks keep a Times face.
for _f in (glob.glob(os.path.expanduser("~/.fonts/times*.ttf"))
           + glob.glob(os.path.join(os.path.dirname(__file__), "fonts", "times*.ttf"))):
    _fm.fontManager.addfont(_f)

# Okabe-Ito categorical order (validated colorblind-safe)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "pink": "#CC79A7", "sky": "#56B4E9", "verm": "#D55E00",
     "grey": "#7F7F7F", "ink": "#1a1a1a"}
ORDER = [C["blue"], C["orange"], C["green"], C["pink"], C["sky"], C["verm"]]

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"],
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.dpi": 400,
    "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=ORDER),
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "mathtext.sf": "Times New Roman", "mathtext.cal": "Times New Roman:italic", "pdf.fonttype": 42, "ps.fonttype": 42,
})

CMAP_SEQ = "viridis"     # magnitude (|psi|)
CMAP_DIV = "RdBu_r"      # signed field maps, neutral midpoint
FW_1COL = 3.5            # inches, IEEE single column
FW_2COL = 7.16           # inches, IEEE double column


def panel_label(ax, s, dx=0.02, dy=0.98, color="k", box=True):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left", color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      pad=1.2) if box else None)
