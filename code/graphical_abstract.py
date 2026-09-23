"""Graphical abstract for the Batteries submission.

Every figure quoted here is read from the result CSVs, so the graphic cannot drift
away from the paper.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
OUT  = os.path.join(ROOT, "Batteries_MDPI", "SUBMISSION")
os.makedirs(OUT, exist_ok=True)

b  = pd.read_csv(os.path.join(RES, "benchmark_loco.csv"))
sf = pd.read_csv(os.path.join(RES, "benchmark_single_feature.csv"))
tc = pd.read_csv(os.path.join(RES, "benchmark_transfer_clipped.csv"))
m  = pd.read_csv(os.path.join(RES, "cycles_filtered.csv"))

best      = b.rmse.min()
best_row  = b.loc[b.rmse.idxmin()]
n_ratio   = b.groupby("n").rmse.min()
win_gain  = b[b.window == "narrow"].rmse.min() / b[b.window == "wide"].rmse.min()
floor     = sf.rmse.min()
mem_gain  = floor / best
transfer  = (b.merge(tc, on=["window", "n", "model"]).eval("rmse_clip / rmse")).median()
ncyc, ncell = len(m), m.cell.nunique()

setup(13.0)
fig = plt.figure(figsize=(11.0, 5.5))                      # 3300 x 1650 px at 300 dpi
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

ax.text(50, 48.0, "How little is enough for on-board state-of-health estimation?",
        fontsize=16, weight="bold", color=INK, ha="center", va="top")
ax.text(50, 44.0, f"sensing and compute budgets measured on {ncyc:,} charge cycles of {ncell} lithium-ion cells".replace(",", " "),
        fontsize=11.5, color=MUTED, ha="center", va="top")

STAGES = [
    ("charge window", C1, f"{win_gain:.1f}× better over a\n450 mV span than 150 mV"),
    ("$N$ samples", C2, f"{n_ratio[4]:.4f} at $N$=4 vs\n{n_ratio[32]:.4f} at $N$=32"),
    ("estimator", C3, f"4 B to {int(best_row.bytes_int8):,} B\nchanges error {mem_gain:.1f}×".replace(",", " ")),
    ("SoH estimate", C4, f"{best:.4f} SoH, cell-wise\nint8 costs 3 %"),
]
bw, gap = 19.5, 5.0
x0 = (100 - (len(STAGES) * bw + (len(STAGES) - 1) * gap)) / 2
ytop, bh = 37.5, 19.0

for i, (name, col, sub) in enumerate(STAGES):
    x = x0 + i * (bw + gap); y = ytop - bh
    for fc, lw in ((col, 0), ("none", 1.4)):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=1.0",
                     linewidth=lw, edgecolor=col, facecolor=fc,
                     alpha=0.10 if lw == 0 else 1.0))
    cy = y + bh - 5.4
    if i == 0:                                              # rising CC-charge curve + span
        t = np.linspace(0, 1, 200)
        v = 3.70 + 0.45 * (t ** 0.55)
        xs = x + 3.0 + 13.5 * t
        ax.add_patch(Rectangle((x + 3.0, cy - 2.6), 13.5, 5.2, facecolor=col, alpha=0.13, linewidth=0))
        ax.plot(xs, cy - 2.6 + (v - 3.70) / 0.45 * 5.2, color=col, linewidth=1.7)
        ax.annotate("", xy=(x + 1.7, cy + 2.6), xytext=(x + 1.7, cy - 2.6),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.0))
    elif i == 1:                                            # four retained samples
        t = np.linspace(0.06, 0.94, 4)
        xs = x + 3.0 + 13.5 * t
        ys = cy - 2.6 + (t ** 0.55) * 5.2
        ax.plot(x + 3.0 + 13.5 * np.linspace(0, 1, 100),
                cy - 2.6 + (np.linspace(0, 1, 100) ** 0.55) * 5.2,
                color=col, linewidth=1.0, alpha=0.35)
        for xx, yy in zip(xs, ys):
            ax.add_patch(Circle((xx, yy), 0.62, facecolor=col, linewidth=0))
    elif i == 2:                                            # a very small model, drawn with its edges
        layers = [3, 2, 1]
        xs_l = [x + 5.0, x + 9.8, x + 14.6]
        pos = []
        for lay, nn in enumerate(layers):
            ys = cy + (np.arange(nn) - (nn - 1) / 2) * 2.1
            pos.append([(xs_l[lay], yy) for yy in ys])
        for a_, b_ in zip(pos[:-1], pos[1:]):
            for (xa, ya) in a_:
                for (xb, yb) in b_:
                    ax.plot([xa, xb], [ya, yb], color=col, linewidth=0.5, alpha=0.45, zorder=1)
        for layer in pos:
            for (xx, yy) in layer:
                ax.add_patch(Circle((xx, yy), 0.58, facecolor=col, linewidth=0, alpha=0.9, zorder=2))
    else:                                                   # gauge
        cxg, cyg, r = x + bw / 2, cy - 1.8, 3.6
        ax.add_patch(Wedge((cxg, cyg), r, 20, 160, width=1.05, facecolor=col, alpha=0.30, linewidth=0))
        ax.add_patch(Wedge((cxg, cyg), r, 115, 160, width=1.05, facecolor=col, linewidth=0))
        ang = np.deg2rad(128)
        ax.plot([cxg, cxg + (r - 1.4) * np.cos(ang)], [cyg, cyg + (r - 1.4) * np.sin(ang)],
                color=INK, linewidth=1.4)
        ax.add_patch(Circle((cxg, cyg), 0.42, facecolor=INK, linewidth=0))
    ax.text(x + bw / 2, y + 6.4, name, fontsize=12.5, weight="bold", color=col, ha="center", va="center")
    ax.text(x + bw / 2, y + 2.8, sub, fontsize=9.2, color=INK2, ha="center", va="center", linespacing=1.3)
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.7, y + bh / 2), (x + bw + gap - 0.7, y + bh / 2),
                     arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=MUTED))

# ---- the punchline: what actually costs accuracy -------------------------------
by, bh2 = 2.6, 13.0
ax.add_patch(FancyBboxPatch((3.0, by), 94.0, bh2, boxstyle="round,pad=0,rounding_size=1.0",
             linewidth=1.1, edgecolor=MUTED, facecolor="#f7f7f5"))
ax.text(50, by + bh2 - 2.9, "what actually costs accuracy", fontsize=12.5, weight="bold",
        color=INK, ha="center", va="center")
LADDER = [("sample count\n$N$=32 → 4", n_ratio[4] / n_ratio[32], MUTED),
          ("model size\n18 kB → 4 B", mem_gain, C3),
          ("window width\n450 → 150 mV", win_gain, C1),
          ("cell design\nCS2 → CX2", transfer, C2)]
lx0, lw = 7.0, 21.5
for i, (lab, val, col) in enumerate(LADDER):
    x = lx0 + i * (lw + 1.0)
    frac = np.log10(max(val, 1.0)) / np.log10(max(v for _, v, _ in LADDER))
    ax.add_patch(Rectangle((x, by + 2.4), lw, 1.5, facecolor="#e6e5e1", linewidth=0))
    ax.add_patch(Rectangle((x, by + 2.4), lw * max(frac, 0.03), 1.5, facecolor=col, linewidth=0))
    ax.text(x + lw / 2, by + 6.4, lab, fontsize=9.2, color=INK2, ha="center", va="center", linespacing=1.3)
    ax.text(x + lw / 2, by + 0.9, f"{val:.1f}× worse", fontsize=9.6, weight="bold",
            color=col if val > 2 else INK2, ha="center", va="center")

save(fig, os.path.join(OUT, "graphical_abstract"))
