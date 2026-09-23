"""External transfer, raw and under the guard any controller would apply.

An estimator carried to an unseen cell design extrapolates, and an unguarded
extrapolation can leave the physically admissible SoH range entirely.  Any
deployed estimator clips; we therefore report the transferred error both raw and
after clipping to [0.10, 1.05].
"""
import os, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
import benchmark as B

LO, HI = 0.10, 1.05
meta, z = B.load()
cs2 = meta[meta.cell.str.startswith("CS2")].reset_index(drop=True)
cx2 = meta[meta.cell.str.startswith("CX2")].reset_index(drop=True)
rows = []
for w, n in itertools.product(B.WINDOWS, B.NS):
    Xtr, _ = B.build_matrix(cs2, z, w, n)
    Xte, _ = B.build_matrix(cx2, z, w, n)
    ytr, yte = cs2.soh.values, cx2.soh.values
    for name, spec in B.make_models(Xtr.shape[1]).items():
        sc = StandardScaler().fit(Xtr)
        m = spec["build"]().fit(sc.transform(Xtr), ytr)
        p = m.predict(sc.transform(Xte))
        pc = np.clip(p, LO, HI)
        rows.append(dict(window=w, n=n, model=name,
                         rmse_raw=float(np.sqrt(np.mean((p - yte) ** 2))),
                         rmse_clip=float(np.sqrt(np.mean((pc - yte) ** 2))),
                         bias_clip=float(np.mean(pc - yte)),
                         out_of_range=float(np.mean((p < LO) | (p > HI)))))
    print("done", w, n, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(B.RES, "benchmark_transfer_clipped.csv"), index=False)
print(df.groupby("model")[["rmse_raw", "rmse_clip", "out_of_range"]].min().to_string())
