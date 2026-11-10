# Executed inside make_r1_figures.py: Figure 6, transfer decomposition.
def fig_transfer():
    D = pd.read_csv(os.path.join(RES, "r1_transfer_groups_raw.csv")).set_index("model")
    D8 = pd.read_csv(os.path.join(RES, "r1_transfer_groups_noI_selected.csv")).set_index("model")
    d = pd.read_csv(os.path.join(RES, "r1_loco_cells.csv"))
    tr = pd.read_csv(os.path.join(RES, "r1_transfer_cells.csv"))
    s0 = d[(d.rep == "ref") & (d.seed == 0)]
    P = s0.groupby(["window", "n", "model"]).apply(pool).rename("rmse").reset_index()
    sel = P.loc[P.groupby("model").rmse.idxmin()].set_index("model")
    cx = {}
    for m_, s_ in sel.iterrows():
        x = tr[(tr.scenario == "cx2_loco") & (tr.seed == 0) & (tr.featset == "raw") & (tr.model == m_)
               & (tr.window == s_.window) & (tr.n == s_.n)]
        cx[m_] = pool(x, "sse_clip")
    D["loco"] = sel.rmse; D["cx2_loco"] = pd.Series(cx)
    cols = [("loco", "CS2 LOCO", INK2), ("prot", "protocol shift", C3), ("des", "design shift", C2),
            ("both", "design + protocol", C5), ("cx2_loco", "within-CX2 LOCO", C6)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for a_, (tab, title, models) in zip(ax, [(D, "ten features", L.MODELS + ["ridge_dur1"]),
                                              (D8, "eight features (no current features)", L.MODELS)]):
        x = np.arange(len(models)); wdt = 0.16
        for k, (col, lab, colr) in enumerate(cols):
            a_.bar(x + (k - 2) * wdt, tab.loc[models, col].values, wdt, color=colr, label=lab, edgecolor="none")
        a_.set_xticks(x); a_.set_xticklabels([LAB.get(m_, m_) for m_ in models], rotation=30, ha="right", fontsize=6.5)
        a_.set_yscale("log"); a_.set_title(title, fontsize=7.5); despine(a_); a_.grid(alpha=0.45, lw=0.5, axis="y")
    ax[0].set_ylabel("RMSE (SoH), clipped")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.3, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.98))
    ax[0].text(-0.02, 1.02, "(a)", transform=ax[0].transAxes, ha="right", fontsize=8)
    ax[1].text(-0.02, 1.02, "(b)", transform=ax[1].transAxes, ha="right", fontsize=8)
    save(fig, "fig06_transfer")
fig_transfer()
