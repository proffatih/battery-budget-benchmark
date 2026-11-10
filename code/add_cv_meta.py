"""Add constant-voltage-phase metadata (duration, end current) to cycles_r1_all.csv.
A cycle whose CV phase was cut short was not fully charged, so its discharge
capacity understates the cell capacity; this is observable at measurement time
and needs no neighbouring-cycle labels."""
import os, numpy as np, pandas as pd
import r1_lib as L
m = L.load_all()
cv_s = np.full(len(m), np.nan); i_end = np.full(len(m), np.nan)
for k in np.where(m.has_windows.values)[0]:
    t, V, I, Q = L.raw_segment(k); cv = (I > 0.01) & (V > 4.19)
    if cv.sum() > 1:
        cv_s[k] = t[cv][-1] - t[cv][0]; i_end[k] = I[cv].min()
    else:
        cv_s[k] = 0.0
m["cv_s"] = cv_s; m["cv_i_end"] = i_end
m.drop(columns="row").to_csv(os.path.join(L.RES, "cycles_r1_all.csv"), index=False)
print(m.groupby("cell").cv_s.describe()[["min", "25%", "50%"]].round(0))
