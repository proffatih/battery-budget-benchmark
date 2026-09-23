"""Figure 1: what a sensing budget is, drawn on a real charge cycle."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "Batteries_MDPI", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 300, "savefig.bbox": "tight"})

m = pd.read_csv(os.path.join(RES, "cycles.csv")).reset_index(drop=True)
z = np.load(os.path.join(RES, "windows.npz"))
i = int(m[(m.cell == "CS2_35")].index[5])           # an early, healthy cycle

fig = plt.figure(figsize=(7.2, 2.7))
ax = fig.add_axes([0.06, 0.16, 0.40, 0.78])
V32 = z[f"{i}|w_wide_32_V"]; dur = float(m.iloc[i]["w_wide_32_dur"])
t32 = np.linspace(0, dur / 60, len(V32))
ax.plot(t32, V32, color="0.35", lw=1.2, label="CC charge (wide window)")
xmax = t32[-1]
for k, ((lo, hi), c, nm) in enumerate([((3.70, 4.15), "#1f77b4", "wide"),
                                       ((3.80, 4.10), "#ff7f0e", "medium"),
                                       ((3.90, 4.05), "#d62728", "narrow")]):
    xr = xmax * (1.05 + 0.075 * k)
    ax.plot([xr, xr], [lo, hi], color=c, lw=3.5, solid_capstyle="butt")
    ax.text(xr + xmax * 0.025, (lo + hi) / 2, nm, color=c, va="center", ha="left",
            fontsize=6.5, rotation=90)
    for v in (lo, hi):
        ax.plot([0, xr], [v, v], color=c, lw=0.5, ls=":", alpha=0.55)
V4 = z[f"{i}|w_wide_4_V"]; t4 = np.linspace(0, dur / 60, 4)
ax.plot(t4, V4, "o", ms=5, color="k", zorder=5, label="$N=4$ retained samples")
ax.set_xlabel("time in window (min)"); ax.set_ylabel("cell voltage (V)")
ax.set_xlim(0, t32[-1] * 1.32); ax.grid(alpha=0.3)
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
ax.set_title("(a) sensing budget = voltage window $\\times$ $N$ samples", fontsize=8, loc="left")

bx = fig.add_axes([0.54, 0.16, 0.44, 0.78]); bx.axis("off")
bx.set_xlim(0, 10); bx.set_ylim(0, 10)
boxes = [(0.1, "charge\nwindow", "#1f77b4"), (2.6, "$N$ samples\n$\\to$ 10 features", "#2ca02c"),
         (5.1, "estimator", "#ff7f0e"), (7.6, "SoH", "#d62728")]
for x, lab, c in boxes:
    bx.add_patch(FancyBboxPatch((x, 5.0), 2.2, 1.9, boxstyle="round,pad=0.10",
                                fc="white", ec=c, lw=1.3))
    bx.text(x + 1.1, 5.95, lab, ha="center", va="center", fontsize=7)
for x in (2.3, 4.8, 7.3):
    bx.add_patch(FancyArrowPatch((x, 5.95), (x + 0.3, 5.95), arrowstyle="-|>",
                                 mutation_scale=9, color="0.4", lw=1))
bx.annotate("", xy=(1.2, 4.9), xytext=(1.2, 4.0), arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.8))
bx.text(1.2, 3.6, "sensing cost:\n$3N$ stored samples", ha="center", va="top", fontsize=6.5, color="#1f77b4")
bx.annotate("", xy=(6.2, 4.9), xytext=(6.2, 4.0), arrowprops=dict(arrowstyle="-", color="#ff7f0e", lw=0.8))
bx.text(6.2, 3.6, "compute cost:\nint8 bytes, MACs", ha="center", va="top", fontsize=6.5, color="#ff7f0e")
bx.text(0.0, 8.2, "(b) what each stage costs", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig01_concept.{ext}"))
print("wrote fig01_concept")
