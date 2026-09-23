"""How far does ONE feature get you?  Duration-only ridge, same folds, same budgets.

This is the floor of the compute axis: three parameters (slope, intercept and one
standardiser pair), one multiply per inference.
"""
import os, itertools, json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import benchmark as B

RES = B.RES
meta, z = B.load()
cs2 = meta[meta.cell.str.startswith("CS2")]
rows = []
for w, n in itertools.product(B.WINDOWS, B.NS):
    dur = cs2[f"w_{w}_{n}_dur"].values.reshape(-1, 1)
    y = cs2.soh.values; cells = cs2.cell.values
    pred = np.full(len(y), np.nan)
    for c in np.unique(cells):
        tr, te = cells != c, cells == c
        sc = StandardScaler().fit(dur[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(dur[tr]), y[tr])
        pred[te] = m.predict(sc.transform(dur[te]))
    rows.append(dict(window=w, n=n, model="ridge_dur1",
                     rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
                     mae=float(np.mean(np.abs(pred - y))),
                     worst_cell_rmse=float(max(np.sqrt(np.mean((pred[cells == c] - y[cells == c]) ** 2))
                                               for c in np.unique(cells))),
                     params=4, bytes_int8=4, macs=1))
# external transfer of the same one-feature model to the unseen CX2 design
cx2 = meta[meta.cell.str.startswith("CX2")]
for r in rows:
    w, n = r["window"], r["n"]
    dtr = cs2[f"w_{w}_{n}_dur"].values.reshape(-1, 1)
    dte = cx2[f"w_{w}_{n}_dur"].values.reshape(-1, 1)
    sc = StandardScaler().fit(dtr)
    m = Ridge(alpha=1.0).fit(sc.transform(dtr), cs2.soh.values)
    p = m.predict(sc.transform(dte))
    r["rmse_cx2"] = float(np.sqrt(np.mean((p - cx2.soh.values) ** 2)))
    r["bias_cx2"] = float(np.mean(p - cx2.soh.values))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "benchmark_single_feature.csv"), index=False)
print(df.sort_values("rmse").to_string(index=False))
print("best single-feature RMSE: %.4f" % df.rmse.min())
print("best transferred RMSE:    %.4f" % df.rmse_cx2.min())
