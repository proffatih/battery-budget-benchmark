"""R1 data figures (house style of v1), from the R1 result CSVs only."""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import setup, despine, INK, INK2, MUTED, C1, C2, C3, C4, C5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import r1_lib as L
warnings.filterwarnings("ignore")

FIG = os.path.join(L.R1, "figures")
os.makedirs(FIG, exist_ok=True)
RES = L.RES
setup(8.0)
plt.rcParams.update({"figure.dpi": 300, "axes.grid": True, "axes.axisbelow": True, "savefig.bbox": "tight"})
C6 = "#7b5cd6"
C = {"ridge": C1, "mlp8": C2, "mlp32x16": C5, "svr": C3, "rf": C6, "hgb": C4, "ridge_dur1": INK}
LAB = {"ridge_dur1": "duration only"}
REPM = {"fp32": "o", "int8": "s", "int16": "D", "fixed16": "*"}
REPL = {"fp32": "float32", "int8": "int8", "int16": "int16", "fixed16": "int16 fixed-point"}
def pool(x, col="sse"): return float(np.sqrt(x[col].sum() / x.ncyc.sum()))

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("wrote", name)

def fig_budget():
    bt = pd.read_csv(os.path.join(RES, "r1_boot_rmse.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, w in zip(axes, L.WINDOWS):
        s = bt[bt.window == w]
        for mdl in L.MODELS + ["ridge_dur1"]:
            g = s[s.model == mdl].sort_values("n")
            ls = "--" if mdl == "ridge_dur1" else "-"
            ax.plot(g.n, g.rmse, marker="o" if mdl != "ridge_dur1" else None, ms=3, lw=1.1, ls=ls,
                    color=C[mdl], label=LAB.get(mdl, mdl))
            if mdl in ("ridge", "hgb"):
                ax.fill_between(g.n, g.lo, g.hi, color=C[mdl], alpha=0.10 if mdl != "ridge_dur1" else 0.06, lw=0)
        despine(ax); ax.grid(alpha=0.45, lw=0.5)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(L.NS); ax.set_xticklabels(L.NS)
        ax.set_xlabel("retained samples $N$"); ax.set_title(w)
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        tk = [0.01, 0.015, 0.02, 0.03, 0.05]
        ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tk]))
        ax.yaxis.set_minor_locator(NullLocator()); ax.set_ylim(0.0085, 0.06)
    axes[0].set_ylabel("LOCO RMSE (SoH)")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    save(fig, "fig03_budget")

def fig_frontier():
    R = pd.read_csv(os.path.join(RES, "r1_representations.csv"))
    F = pd.read_csv(os.path.join(RES, "r1_frontier.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.32))
    V = R[R.rep.isin(REPM) & (R.rmse < 0.05)]
    for (mdl, rp), g in V.groupby(["model", "rep"]):
        ax[0].scatter(g.bytes_med, g.rmse, s=30 if rp == "fixed16" else 12, marker=REPM[rp], color=C[mdl],
                      alpha=0.8, edgecolor="none")
    ax[0].step(F.bytes_med, F.rmse, where="post", color=INK, lw=1.0, ls="--")
    for mdl in L.MODELS + ["ridge_dur1"]:
        ax[0].scatter([], [], s=14, color=C[mdl], label=LAB.get(mdl, mdl))
    for rp in REPM:
        ax[0].scatter([], [], s=14, marker=REPM[rp], color=MUTED, label=REPL[rp])
    ax[0].plot([], [], color=INK, ls="--", lw=1.0, label="frontier")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("model data of the executable representation (bytes)"); ax[0].set_ylabel("LOCO RMSE (SoH)")
    ax[0].legend(fontsize=5.6, frameon=False, ncol=7, loc="upper left", bbox_to_anchor=(0.0, -0.2), columnspacing=0.9, handletextpad=0.3)
    pairs = {"ridge": "int8", "mlp8": "int8", "mlp32x16": "int8", "svr": "int16", "rf": "int8", "hgb": "int8"}
    for mdl, rp in pairs.items():
        a = R[(R.model == mdl) & (R.rep == "fp32")].set_index(["window", "n"]).rmse
        b = R[(R.model == mdl) & (R.rep == rp)].set_index(["window", "n"]).rmse
        ax[1].scatter(a, b.reindex(a.index), s=14, color=C[mdl], marker=REPM[rp], alpha=0.85, edgecolor="none",
                      label=f"{mdl} ({rp})")
    lim = [0.009, 0.06]
    ax[1].plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [0.01, 0.02, 0.03, 0.05]
    for axis in (ax[1].xaxis, ax[1].yaxis):
        axis.set_major_locator(FixedLocator(tk)); axis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tk]))
        axis.set_minor_locator(NullLocator())
    ax[1].set_xlabel("RMSE, float32 representation"); ax[1].set_ylabel("RMSE, compact representation")
    ax[1].legend(fontsize=5.6, frameon=False, loc="upper left")
    for a_ in ax: despine(a_); a_.grid(alpha=0.45, lw=0.5)
    ax[0].set_title("(a)", loc="left", fontsize=8); ax[1].set_title("(b)", loc="left", fontsize=8)
    save(fig, "fig04_pareto")

def fig_acq():
    on = pd.read_csv(os.path.join(RES, "r1_online_cells.csv"))
    sp = pd.read_csv(os.path.join(RES, "r1_span_cells.csv"))
    d = pd.read_csv(os.path.join(RES, "r1_loco_cells.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    po = on[(on.qmode == "counter") & (on.window == "wide") & (on.rate_s == 30)].groupby(["n", "model"]).apply(pool).rename("rmse").reset_index()
    pr = d[(d.rep == "ref") & (d.seed == 0) & (d.window == "wide")].groupby(["n", "model"]).apply(pool).rename("rmse").reset_index()
    for mdl in ["ridge", "svr", "hgb", "ridge_dur1"]:
        a = pr[pr.model == mdl].sort_values("n"); b = po[po.model == mdl].sort_values("n")
        ax[0].plot(a.n, a.rmse, color=C[mdl], lw=1.0, ls=":", marker="o", ms=2.5)
        ax[0].plot(b.n, b.rmse, color=C[mdl], lw=1.2, marker="s", ms=2.5, label=LAB.get(mdl, mdl))
    ax[0].plot([], [], color=MUTED, ls=":", marker="o", ms=2.5, label="retrospective")
    ax[0].plot([], [], color=MUTED, marker="s", ms=2.5, label="online, $\\leq 2N{+}1$ kept")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(L.NS); ax[0].set_xticklabels(L.NS)
    ax[0].set_xlabel("$N$ (buffer capacity $2N$)"); ax[0].set_ylabel("LOCO RMSE (SoH)")
    ax[0].legend(fontsize=5.2, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.06))
    rr = on[(on.qmode == "counter") & (on.window == "wide")].groupby(["rate_s", "model"]).apply(pool).rename("rmse").reset_index()
    rr = rr.groupby(["rate_s", "model"]).rmse.min().reset_index()
    for mdl in ["ridge", "svr", "hgb", "ridge_dur1"]:
        g = rr[rr.model == mdl].sort_values("rate_s")
        ax[1].plot(g.rate_s, g.rmse, color=C[mdl], marker="o", ms=3, lw=1.1, label=LAB.get(mdl, mdl))
    ax[1].set_xscale("log", base=2); ax[1].set_xticks([30, 60, 120, 240]); ax[1].set_xticklabels([30, 60, 120, 240])
    ax[1].set_xlabel("acquisition interval (s)"); ax[1].set_ylabel("LOCO RMSE (SoH), wide window")
    sc = sp[sp.cycle_set == "common"]
    t = sc.groupby(["window", "model"]).apply(pool).rename("rmse").reset_index()
    dur = sc.groupby("window").dur_median_s.median() / 60
    vv = sc.groupby("window")[["v_lo", "v_hi"]].first()
    for mdl in ["ridge", "hgb", "ridge_dur1"]:
        g = t[t.model == mdl].set_index("window")
        ax[2].scatter(dur[g.index], g.rmse, color=C[mdl], s=16, label=LAB.get(mdl, mdl), zorder=3)
    OFF = {"s15_lo": (4, 6), "s15_hi": (6, -2), "narrow": (6, -6), "s15_mid": (7, 3),
           "s30_lo": (-6, 9), "s30_hi": (-8, 9), "s45": (-12, -12)}
    for wn in dur.index:
        lab = f"{vv.loc[wn,'v_lo']:.2f}–{vv.loc[wn,'v_hi']:.2f} V"
        y = t[(t.window == wn)].rmse.max()
        ax[2].annotate(lab, (dur[wn], y), xytext=OFF.get(wn, (3, 2)), textcoords="offset points",
                       fontsize=5.0, color=INK2)
    ax[2].set_yscale("log"); ax[2].set_xlabel("median traversal time (min)"); ax[2].set_ylabel("LOCO RMSE (SoH)")
    ax[2].legend(fontsize=5.2, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.06))
    for a_, tt in zip(ax, ["(a)", "(b)", "(c)"]):
        despine(a_); a_.grid(alpha=0.45, lw=0.5); a_.text(-0.02, 1.02, tt, transform=a_.transAxes, ha="right", fontsize=8)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig05_acquisition")

if __name__ == "__main__":
    which = sys.argv[1:] or ["budget", "frontier", "acq", "transfer"]
    if "budget" in which: fig_budget()
    if "frontier" in which: fig_frontier()
    if "acq" in which: fig_acq()
    if "transfer" in which and os.path.exists(os.path.join(RES, "r1_transfer_cells.csv")):
        exec(open(os.path.join(L.HERE, "make_r1_fig_transfer.py")).read())
