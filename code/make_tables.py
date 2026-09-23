"""Generate the LaTeX tables directly from the result CSVs."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); MAN = os.path.join(ROOT, "Batteries_MDPI", "manuscript")
m = pd.read_csv(os.path.join(RES, "cycles_filtered.csv"))
b = pd.read_csv(os.path.join(RES, "benchmark_loco.csv"))
t = pd.read_csv(os.path.join(RES, "benchmark_transfer_clipped.csv")).rename(
    columns={"rmse_clip": "rmse_cx2", "bias_clip": "bias_cx2"})

# ---- Table 1: cells -------------------------------------------------------
rows = []
for c, g in m.groupby("cell"):
    rows.append((c.replace("_", r"\_"), "CS2" if c.startswith("CS2") else "CX2", len(g),
                 g.q_dis.max(), g.soh.min()))
L = [r"\begin{table}[H]", r"\caption{Cells retained after screening. $Q_0$ is the reference"
     r" capacity (median of the first five retained cycles); SoH$_\text{min}$ is the lowest"
     r" state of health reached.\label{tab:cells}}",
     r"\begin{tabularx}{\textwidth}{lCCCC}", r"\toprule",
     r"\textbf{Cell} & \textbf{Design} & \textbf{Cycles} & \textbf{$Q_0$ (Ah)} & \textbf{SoH$_\text{min}$}\\",
     r"\midrule"]
for r in rows:
    L.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]:.3f} & {r[4]:.3f} \\\\")
L += [r"\midrule",
      f"\\textbf{{Total}} & & \\textbf{{{len(m)}}} & & \\\\",
      r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
open(os.path.join(MAN, "table_cells.tex"), "w").write("\n".join(L) + "\n")

# ---- Table 2: full benchmark (best window per model x N) ------------------
piv = (b.sort_values("rmse").groupby(["model", "n"]).first().reset_index())
L = [r"\begin{table}[H]",
     r"\caption{Leave-one-cell-out accuracy and cost. For each estimator and sample"
     r" count the best of the three voltage windows is shown. Memory is int8 parameter"
     r" bytes including the standardiser; MACs are multiply--accumulates per inference"
     r" (for the tree ensembles the figure is a comparison count and the memory a node"
     r" count, marked $^\dagger$, and is not directly comparable).\label{tab:bench}}",
     r"\begin{tabularx}{\textwidth}{llCCCCC}", r"\toprule",
     r"\textbf{Model} & \textbf{Window} & \textbf{$N$} & \textbf{RMSE} &"
     r" \textbf{Worst cell} & \textbf{Bytes} & \textbf{MACs}\\", r"\midrule"]
for mdl in ["ridge", "mlp8", "mlp32x16", "svr", "rf", "hgb"]:
    for _, r in piv[piv.model == mdl].sort_values("n").iterrows():
        dag = r"$^\dagger$" if mdl in ("rf", "hgb") else ""
        L.append(f"\\texttt{{{mdl.replace('_', chr(92)+'_')}}} & {r.window} & {int(r.n)} & {r.rmse:.4f} & "
                 f"{r.worst_cell_rmse:.4f} & {int(r.bytes_int8):,}{dag} & {int(r.macs):,}{dag} \\\\".replace(",", r"\,"))
    L.append(r"\midrule")
L = L[:-1] + [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
open(os.path.join(MAN, "table_bench.tex"), "w").write("\n".join(L) + "\n")

# ---- Table 3: the accuracy-cost frontier ---------------------------------
sf = pd.read_csv(os.path.join(RES, "benchmark_single_feature.csv"))
sf["sense_bytes"] = 4                       # two timestamps, 16-bit each
cand = pd.concat([b[["model", "window", "n", "rmse", "worst_cell_rmse", "bytes_int8", "macs", "sense_bytes"]],
                  sf[["model", "window", "n", "rmse", "worst_cell_rmse", "bytes_int8", "macs", "sense_bytes"]]],
                 ignore_index=True).sort_values(["bytes_int8", "rmse"])
front, best_so_far = [], np.inf
for _, r in cand.iterrows():
    if r.rmse < best_so_far:
        best_so_far = r.rmse; front.append(r)
L = [r"\begin{table}[H]",
     r"\caption{The accuracy--cost frontier: every configuration that is not beaten"
     r" on both accuracy and memory by a cheaper one. Sensing payload is the stored"
     r" window ($3N$ channels as 16-bit integers) or, for the one-feature estimator,"
     r" two timestamps.\label{tab:cheap}}",
     r"\begin{tabularx}{\textwidth}{lCCCCCC}", r"\toprule",
     r"\textbf{Estimator} & \textbf{Window} & \textbf{$N$} & \textbf{RMSE} &"
     r" \textbf{Worst cell} & \textbf{Bytes} & \textbf{Sensing (B)}\\", r"\midrule"]
for r in front:
    nm = str(r.model).replace("_", r"\_")
    nn = "---" if nm.startswith("ridge") and "dur" in nm else str(int(r.n))
    L.append(f"\\texttt{{{nm}}} & {r.window} & {nn} & {r.rmse:.4f} & "
             f"{r.worst_cell_rmse:.4f} & {int(r.bytes_int8):,} & {int(r.sense_bytes)} \\\\".replace(",", r"\,"))
L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
open(os.path.join(MAN, "table_cheap.tex"), "w").write("\n".join(L) + "\n")

# ---- Table 4: transfer ----------------------------------------------------
k = b.merge(t, on=["window", "n", "model"])
best = k.sort_values("rmse_cx2").groupby("model").first().reset_index()
L = [r"\begin{table}[H]",
     r"\caption{External transfer to the unseen CX2 design. For each estimator the"
     r" configuration with the lowest transferred error is shown, together with the"
     r" leave-one-cell-out error of the same configuration on CS2. Transferred"
     r" predictions are clipped to the admissible range $[0.10,\,1.05]$; the last"
     r" column is the share of raw predictions that fell outside"
     r" it.\label{tab:transfer}}",
     r"\begin{tabularx}{\textwidth}{lCCCCCC}", r"\toprule",
     r"\textbf{Model} & \textbf{Window} & \textbf{$N$} & \textbf{RMSE (CS2)} &"
     r" \textbf{RMSE (CX2)} & \textbf{Mean signed error} & \textbf{Out of range}\\", r"\midrule"]
for _, r in best.sort_values("rmse_cx2").iterrows():
    L.append(f"\\texttt{{{str(r.model).replace('_', chr(92)+'_')}}} & {r.window} & {int(r.n)} & {r.rmse:.4f} & "
             f"{r.rmse_cx2:.4f} & {r.bias_cx2:+.4f} & {100*r.out_of_range:.0f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
open(os.path.join(MAN, "table_transfer.tex"), "w").write("\n".join(L) + "\n")
print("tables written")
