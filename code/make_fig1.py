"""Figure 1: what a sensing budget is, drawn on a real charge cycle, and what each
stage of the estimation chain costs."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "Batteries_MDPI", "figures")
os.makedirs(FIG, exist_ok=True)

m = pd.read_csv(os.path.join(RES, "cycles.csv")).reset_index(drop=True)
z = np.load(os.path.join(RES, "windows.npz"))
i = int(m[(m.cell == "CS2_35")].index[5])           # an early, healthy cycle

setup(8.5)
fig = plt.figure(figsize=(7.2, 3.15))

# ---------------------------------------------------------------- panel (a)
ax = fig.add_axes([0.065, 0.135, 0.40, 0.75])
V32 = z[f"{i}|w_wide_32_V"]; dur = float(m.iloc[i]["w_wide_32_dur"])
t32 = np.linspace(0, dur / 60, len(V32))
ax.plot(t32, V32, color=INK2, lw=1.3, zorder=3, label="CC charge")
xmax = t32[-1]
for k, ((lo, hi), c, nm) in enumerate([((3.70, 4.15), C1, "wide"),
                                       ((3.80, 4.10), C2, "medium"),
                                       ((3.90, 4.05), C4, "narrow")]):
    xr = xmax * (1.05 + 0.075 * k)
    ax.plot([xr, xr], [lo, hi], color=c, lw=4.0, solid_capstyle="butt")
    ax.text(xr + xmax * 0.028, (lo + hi) / 2, nm, color=c, va="center", ha="left",
            fontsize=6.6, rotation=90, weight="bold")
    for v in (lo, hi):
        ax.plot([0, xr], [v, v], color=c, lw=0.5, ls=":", alpha=0.55)
V4 = z[f"{i}|w_wide_4_V"]; t4 = np.linspace(0, dur / 60, 4)
ax.plot(t4, V4, "o", ms=5.2, color=C1, zorder=5, markeredgecolor="white",
        markeredgewidth=0.7, label="$N=4$ retained samples")
ax.set_xlabel("time in window (min)"); ax.set_ylabel("cell voltage (V)")
ax.set_xlim(0, xmax * 1.32); ax.grid(alpha=0.45, lw=0.5)
despine(ax)
ax.legend(fontsize=6.4, loc="lower right", frameon=False)
ax.set_title("(a)  sensing budget = voltage window $\\times$ $N$ samples",
             fontsize=8.2, loc="left", color=INK)

# ---------------------------------------------------------------- panel (b)
bx = fig.add_axes([0.525, 0.135, 0.46, 0.75]); bx.axis("off")
bx.set_xlim(0, 100); bx.set_ylim(28, 100)
bx.text(0, 99, "(b)  what each stage costs", fontsize=8.2, color=INK, va="top")

STAGES = [("charge\nwindow", C1), ("$N$ samples\n10 features", C2),
          ("estimator", C3), ("SoH", C4)]
bw, gap, y, bh = 20.5, 5.7, 44, 26
for k, (lab, col) in enumerate(STAGES):
    x = k * (bw + gap)
    for fc, lw in ((col, 0), ("none", 1.1)):
        bx.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=2.0",
                     linewidth=lw, edgecolor=col, facecolor=fc, alpha=0.10 if lw == 0 else 1.0))
    cy = y + bh - 7.0
    if k == 0:
        t = np.linspace(0, 1, 80)
        bx.add_patch(Rectangle((x + 3.5, cy - 3.6), 13.5, 7.2, facecolor=col, alpha=0.16, linewidth=0))
        bx.plot(x + 3.5 + 13.5 * t, cy - 3.6 + (t ** 0.55) * 7.2, color=col, lw=1.3)
    elif k == 1:
        t = np.linspace(0.06, 0.94, 4)
        bx.plot(x + 3.5 + 13.5 * np.linspace(0, 1, 60),
                cy - 3.6 + (np.linspace(0, 1, 60) ** 0.55) * 7.2, color=col, lw=0.8, alpha=0.4)
        for tt in t:
            bx.add_patch(Circle((x + 3.5 + 13.5 * tt, cy - 3.6 + (tt ** 0.55) * 7.2),
                                0.85, facecolor=col, linewidth=0))
    elif k == 2:
        xs_l = [x + 5.0, x + 10.2, x + 15.4]
        pos = [[(xs_l[l], cy + (np.arange(n) - (n - 1) / 2)[j] * 2.9) for j in range(n)]
               for l, n in enumerate([3, 2, 1])]
        for a_, b_ in zip(pos[:-1], pos[1:]):
            for (xa, ya) in a_:
                for (xb, yb) in b_:
                    bx.plot([xa, xb], [ya, yb], color=col, lw=0.4, alpha=0.45, zorder=1)
        for layer in pos:
            for (xx, yy) in layer:
                bx.add_patch(Circle((xx, yy), 0.9, facecolor=col, linewidth=0, alpha=0.9, zorder=2))
    else:
        cxg, cyg, r = x + bw / 2, cy - 2.6, 5.0
        bx.add_patch(Wedge((cxg, cyg), r, 20, 160, width=1.5, facecolor=col, alpha=0.30, linewidth=0))
        bx.add_patch(Wedge((cxg, cyg), r, 115, 160, width=1.5, facecolor=col, linewidth=0))
        ang = np.deg2rad(128)
        bx.plot([cxg, cxg + (r - 2.0) * np.cos(ang)], [cyg, cyg + (r - 2.0) * np.sin(ang)],
                color=INK, lw=1.1)
        bx.add_patch(Circle((cxg, cyg), 0.6, facecolor=INK, linewidth=0))
    bx.text(x + bw / 2, y + 4.5, lab, ha="center", va="center", fontsize=7.0,
            color=INK, linespacing=1.25)
    if k < len(STAGES) - 1:
        bx.add_patch(FancyArrowPatch((x + bw + 0.8, y + bh / 2), (x + bw + gap - 0.8, y + bh / 2),
                     arrowstyle="-|>", mutation_scale=8, lw=1.0, color=MUTED))

for x0_, x1_, lab, col in [(0, bw, "sensing cost:\n$3N$ stored samples", C1),
                           (2 * (bw + gap), 2 * (bw + gap) + bw,
                            "compute cost:\nint8 bytes, MACs", C3)]:
    cxm = (x0_ + x1_) / 2
    bx.plot([cxm, cxm], [y - 2.0, y - 7.0], color=col, lw=0.8)
    bx.text(cxm, y - 9.0, lab, ha="center", va="top", fontsize=6.6, color=col, linespacing=1.3)

save(fig, os.path.join(FIG, "fig01_concept"))
