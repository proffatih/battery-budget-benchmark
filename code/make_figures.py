"""Figures for the sensing/compute budget benchmark."""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
FIG  = os.path.join(ROOT, "Batteries_MDPI", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 300, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.bbox": "tight"})
C = {"ridge": "#1f77b4", "mlp8": "#ff7f0e", "mlp32x16": "#d62728",
     "svr": "#2ca02c", "rf": "#9467bd", "hgb": "#8c564b"}
MK = {"narrow": "o", "medium": "s", "wide": "^"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)

def fig_fade():
    m = pd.read_csv(os.path.join(RES, "cycles_filtered.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k, fam in enumerate(["CS2", "CX2"]):
        sub = m[m.cell.str.startswith(fam)]
        for c, g in sub.groupby("cell"):
            ax[k].plot(np.arange(len(g)), g.soh.values, lw=0.8, label=c)
        ax[k].set_xlabel("cycle index"); ax[k].set_ylabel("SoH (–)")
        ax[k].set_title(f"{fam} cells"); ax[k].legend(fontsize=6, frameon=False)
    save(fig, "fig02_fade")

def fig_budget_curves():
    b = pd.read_csv(os.path.join(RES, "benchmark_loco.csv"))
    sf = pd.read_csv(os.path.join(RES, "benchmark_single_feature.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, w in zip(axes, ["narrow", "medium", "wide"]):
        s = b[b.window == w]
        for mdl, g in s.groupby("model"):
            g = g.sort_values("n")
            ax.plot(g.n, g.rmse, marker="o", ms=3, lw=1.1, color=C.get(mdl), label=mdl)
        v = sf[sf.window == w].rmse.iloc[0]
        ax.axhline(v, color="k", lw=1.2, ls="--",
                   label="duration only (4 par.)" if w == "wide" else None)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels([4, 8, 16, 32])
        ax.set_xlabel("samples per window $N$"); ax.set_title(w)
    axes[0].set_ylabel("LOCO RMSE (SoH)")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, frameon=False, ncol=7, loc="lower center",
               bbox_to_anchor=(0.5, -0.13))
    save(fig, "fig03_budget")

def fig_pareto():
    b = pd.read_csv(os.path.join(RES, "benchmark_loco.csv"))
    sf = pd.read_csv(os.path.join(RES, "benchmark_single_feature.csv"))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for mdl, g in b.groupby("model"):
        ax.scatter(g.bytes_int8, g.rmse, s=14, color=C.get(mdl), label=mdl, alpha=0.8,
                   edgecolor="none")
    ax.scatter(sf.bytes_int8, sf.rmse, s=42, marker="*", color="k",
               label="duration only", zorder=6)
    b = pd.concat([b[["bytes_int8", "rmse"]], sf[["bytes_int8", "rmse"]]], ignore_index=True)
    # Pareto front
    pts = b[["bytes_int8", "rmse"]].values
    order = np.argsort(pts[:, 0]); best = np.inf; fx, fy = [], []
    for i in order:
        if pts[i, 1] < best:
            best = pts[i, 1]; fx.append(pts[i, 0]); fy.append(pts[i, 1])
    ax.step(fx, fy, where="post", color="k", lw=1.0, ls="--", label="Pareto front")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("model memory, int8 (bytes)"); ax.set_ylabel("LOCO RMSE (SoH)")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "fig04_pareto")

def fig_quant():
    b = pd.read_csv(os.path.join(RES, "benchmark_loco.csv"))
    s = b.dropna(subset=["rmse_int8"])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot([s.rmse.min()*0.9, s.rmse.max()*1.1], [s.rmse.min()*0.9, s.rmse.max()*1.1],
            color="k", lw=0.8, ls=":")
    for mdl, g in s.groupby("model"):
        ax.scatter(g.rmse, g.rmse_int8, s=16, color=C.get(mdl), label=mdl, alpha=0.85, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("RMSE, float32"); ax.set_ylabel("RMSE, int8 weights")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "fig05_int8")

def fig_transfer():
    t = pd.read_csv(os.path.join(RES, "benchmark_transfer_clipped.csv"))
    b = pd.read_csv(os.path.join(RES, "benchmark_loco.csv"))
    k = b.merge(t, on=["window", "n", "model"]).rename(
        columns={"rmse_clip": "rmse_cx2", "bias_clip": "bias_cx2"})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for mdl, g in k.groupby("model"):
        ax[0].scatter(g.rmse, g.rmse_cx2, s=16, color=C.get(mdl), label=mdl, alpha=0.85, edgecolor="none")
        ax[1].scatter(g.n, g.bias_cx2, s=16, color=C.get(mdl), alpha=0.85, edgecolor="none")
    lim = [k.rmse.min()*0.9, max(k.rmse.max(), k.rmse_cx2.max())*1.1]
    ax[0].plot(lim, lim, color="k", lw=0.8, ls=":")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("RMSE, held-out CS2 cell"); ax[0].set_ylabel("RMSE, unseen CX2 design")
    ax[0].legend(fontsize=6, frameon=False, ncol=2)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xscale("log", base=2); ax[1].set_xticks([4, 8, 16, 32]); ax[1].set_xticklabels([4, 8, 16, 32])
    ax[1].set_xlabel("samples per window $N$"); ax[1].set_ylabel("mean signed error, CX2")
    save(fig, "fig06_transfer")

if __name__ == "__main__":
    fig_fade(); fig_budget_curves(); fig_pareto(); fig_quant(); fig_transfer()
