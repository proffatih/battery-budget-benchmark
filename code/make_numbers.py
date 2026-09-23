"""Emit every number the manuscript quotes as a LaTeX macro, straight from the result CSVs."""
import os, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "Batteries_MDPI", "manuscript", "numbers.tex")

m = pd.read_csv(os.path.join(RES, "cycles_filtered.csv"))
b = pd.read_csv(os.path.join(RES, "benchmark_loco.csv"))
t = pd.read_csv(os.path.join(RES, "benchmark_transfer.csv"))
L = []
def mac(name, val): L.append(r"\newcommand{\%s}{%s}" % (name, val))

import json as _json
_fs = _json.load(open(os.path.join(RES, "filter_stats.json")))
mac("Nextracted", f"{_fs['all']['total']:,}".replace(",", "\\,"))
mac("Ndropped", str(_fs['all']['total'] - _fs['all']['kept']))
mac("dropPct", f"{100*(1-_fs['all']['kept']/_fs['all']['total']):.1f}")

fam = m.cell.str[:3]
mac("Ncycles", f"{len(m):,}".replace(",", "\\,"))
mac("NcyclesCS", f"{int((fam=='CS2').sum()):,}".replace(",", "\\,"))
mac("NcyclesCX", f"{int((fam=='CX2').sum()):,}".replace(",", "\\,"))
mac("Ncells", str(m.cell.nunique()))
mac("SoHmin", f"{m.soh.min():.3f}")
mac("SoHmax", f"{m.soh.max():.3f}")
mac("rawNarrow", str(int(m.n_raw_narrow.median())))
mac("rawMedium", str(int(m.n_raw_medium.median())))
mac("rawWide", str(int(m.n_raw_wide.median())))
for w, key in [("narrow", "durNarrow"), ("medium", "durMedium"), ("wide", "durWide")]:
    mac(key, f"{m[f'w_{w}_4_dur'].median()/60:.0f}")

best = b.loc[b.rmse.idxmin()]
mac("bestRMSE", f"{best.rmse:.4f}")
mac("bestModel", str(best.model)); mac("bestWindow", str(best.window))
mac("bestN", str(int(best.n))); mac("bestBytes", f"{int(best.bytes_int8):,}".replace(",", "\\,"))
mac("bestMACs", f"{int(best.macs):,}".replace(",", "\\,"))

r4 = b[b.model == "ridge"]
mac("ridgeBestRMSE", f"{r4.rmse.min():.4f}")
mac("ridgeBytes", str(int(r4.bytes_int8.iloc[0])))
mac("ridgeMACs", str(int(r4.macs.iloc[0])))
rid_best = r4.loc[r4.rmse.idxmin()]
mac("ridgeBestWindow", str(rid_best.window)); mac("ridgeBestN", str(int(rid_best.n)))

# sensing-budget effect: best RMSE at each N (any model, any window)
WORD = {4: "four", 8: "eight", 16: "sixteen", 32: "thirtytwo"}
for n in [4, 8, 16, 32]:
    mac(f"bestRMSEn{WORD[n]}", f"{b[b.n == n].rmse.min():.4f}")
for w in ["narrow", "medium", "wide"]:
    mac(f"bestRMSE{w.capitalize()}", f"{b[b.window == w].rmse.min():.4f}")

# quantisation
q = b.dropna(subset=["rmse_int8"]).copy()
q["delta"] = q.rmse_int8 - q.rmse
mac("quantMedDelta", f"{q.delta.median():.5f}")
mac("quantMaxDelta", f"{q.delta.max():.5f}")
mac("quantRelMed", f"{(100*q.delta/q.rmse).median():.2f}")
mac("quantNconfig", str(len(q)))

# memory span
mac("minBytes", f"{int(b.bytes_int8.min()):,}".replace(",", "\\,"))
mac("maxBytes", f"{int(b.bytes_int8.max()):,}".replace(",", "\\,"))
mac("rfBytes", f"{int(b[b.model=='rf'].bytes_int8.median()):,}".replace(",", "\\,"))
mac("rfBestRMSE", f"{b[b.model=='rf'].rmse.min():.4f}")
mac("hgbBytes", f"{int(b[b.model=='hgb'].bytes_int8.median()):,}".replace(",", "\\,"))
mac("mlpBytes", f"{int(b[b.model=='mlp8'].bytes_int8.iloc[0]):,}".replace(",", "\\,"))
mac("mlpBestRMSE", f"{b[b.model=='mlp8'].rmse.min():.4f}")

# worst-cell gap
mac("bestWorstCell", f"{best.worst_cell_rmse:.4f}")
mac("worstGapFactor", f"{(b.worst_cell_rmse/b.rmse).median():.2f}")

mac("windowGain", f"{b[b.window=='narrow'].rmse.min()/b[b.window=='wide'].rmse.min():.1f}")
mac("memRatio", f"{int(best.bytes_int8)/int(r4.bytes_int8.iloc[0]):.0f}")
mac("bestVsRidgePct", f"{100*(r4.rmse.min()-best.rmse)/r4.rmse.min():.0f}")

# one-feature floor
sf = pd.read_csv(os.path.join(RES, "benchmark_single_feature.csv"))
sfb = sf.loc[sf.rmse.idxmin()]
mac("sfBest", f"{sfb.rmse:.4f}"); mac("sfBestWindow", str(sfb.window))
mac("sfWorstCell", f"{sfb.worst_cell_rmse:.4f}")
mac("sfBytes", str(int(sfb.bytes_int8))); mac("sfMACs", str(int(sfb.macs)))
mac("sfNarrow", f"{sf[sf.window=='narrow'].rmse.iloc[0]:.4f}")
mac("sfMedium", f"{sf[sf.window=='medium'].rmse.iloc[0]:.4f}")
mac("sfWide", f"{sf[sf.window=='wide'].rmse.iloc[0]:.4f}")
mac("sfTransfer", f"{sfb.rmse_cx2:.4f}"); mac("sfTransferBias", f"{sfb.bias_cx2:+.4f}")
mac("sfVsBest", f"{sfb.rmse/b.rmse.min():.1f}")
mac("sfMemRatio", f"{int(best.bytes_int8)/int(sfb.bytes_int8):.0f}")

# transfer (clipped to the admissible SoH range, as any controller would)
tc = pd.read_csv(os.path.join(RES, "benchmark_transfer_clipped.csv"))
k = b.merge(tc, on=["window", "n", "model"])
mac("transferBest", f"{k.rmse_clip.min():.4f}")
mac("transferMedRatio", f"{(k.rmse_clip/k.rmse).median():.1f}")
kb = k.loc[k.rmse_clip.idxmin()]
mac("transferBestModel", str(kb.model)); mac("transferBestWindow", str(kb.window))
mac("transferBestN", str(int(kb.n)))
mac("transferBias", f"{kb.bias_clip:+.4f}")
mac("transferRawWorst", f"{k.rmse_raw.max():.1f}")
mac("transferOutMax", f"{100*k.out_of_range.max():.0f}")
mac("transferOutMed", f"{100*k.out_of_range.median():.1f}")
mac("transferBestRatio", f"{kb.rmse_clip/kb.rmse:.1f}")

# cheapest configuration meeting accuracy targets
_sf = pd.read_csv(os.path.join(RES, "benchmark_single_feature.csv"))
_sf["sense_bytes"] = 4
_cand = pd.concat([b[["model", "window", "n", "rmse", "bytes_int8", "macs", "sense_bytes"]],
                   _sf[["model", "window", "n", "rmse", "bytes_int8", "macs", "sense_bytes"]]],
                  ignore_index=True)
for thr, name in [(0.015, "OneFive"), (0.02, "Two"), (0.03, "Three")]:
    ok = _cand[_cand.rmse <= thr].sort_values(["bytes_int8", "sense_bytes", "rmse"])
    if len(ok):
        c = ok.iloc[0]
        mac(f"cheap{name}Model", str(c.model)); mac(f"cheap{name}Bytes", f"{int(c.bytes_int8):,}".replace(",", "\\,"))
        mac(f"cheap{name}N", str(int(c.n))); mac(f"cheap{name}Window", str(c.window))
        mac(f"cheap{name}RMSE", f"{c.rmse:.4f}")
        mac(f"cheap{name}MACs", f"{int(c.macs):,}".replace(",", "\\,"))
    else:
        mac(f"cheap{name}Model", "none")
open(OUT, "w").write("% generated by code/make_numbers.py -- do not edit by hand\n" + "\n".join(L) + "\n")
print("wrote", OUT, len(L), "macros")
