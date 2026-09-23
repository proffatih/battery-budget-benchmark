# Sensing- and compute-budget benchmark for on-board battery state-of-health estimation

Code and derived data for the paper *"How Little Is Enough? Sensing- and Compute-Budget
Limits of On-Board State-of-Health Estimation for Lithium-Ion Batteries"*
(submitted to *Batteries*, MDPI).

The study asks two questions that accuracy-only benchmarks leave implicit:

* **Sensing budget** — how much of the constant-current charge must a battery management
  system observe? Here: a voltage window (narrow 3.90–4.05 V, medium 3.80–4.10 V,
  wide 3.70–4.15 V) and the number `N ∈ {4, 8, 16, 32}` of samples retained inside it.
* **Compute budget** — how much memory and arithmetic does the estimator need? Reported
  hardware-independently as int8 parameter bytes and multiply–accumulate operations per
  inference, plus post-training int8 evaluation.

Twelve budgets × six estimator families, evaluated leave-one-cell-out on four CALCE CS2
cells and then transferred to four unseen CX2 cells.

## Data

Raw cycling data are **not** redistributed here. They are public and come from the
[CALCE Battery Research Group](https://web.calce.umd.edu/batteries/data.htm)
(cells CS2_35–38 and CX2_16/34/36/37). `code/dl_calce.sh` downloads them.

`results/cycles_all.csv` is the derived cycle-level dataset produced by
`code/extract_calce.py` (one row per extracted charge–discharge cycle: discharge
capacity, SoH, per-window sample counts and window durations), and
`results/cycles_filtered.csv` is the subset the analysis uses, after the
rolling-median screening that drops interrupted discharges. The per-cycle
resampled window arrays (`results/windows.npz`, ~116 MB) are not tracked here;
rerunning the extractor regenerates them deterministically.

`results/benchmark_loco.csv`, `benchmark_single_feature.csv`,
`benchmark_transfer.csv` and `benchmark_transfer_clipped.csv` are the complete
result tables behind every number and figure in the paper.

## Reproducing

```bash
bash code/dl_calce.sh            # download + unzip the CALCE cells
python3 code/extract_calce.py    # -> results/cycles.csv, results/windows.npz
python3 code/benchmark.py        # -> results/benchmark_loco.csv, benchmark_transfer.csv
python3 code/single_feature.py   # the one-feature floor
python3 code/transfer_clipped.py # transfer, raw and clipped to the admissible range
python3 code/make_numbers.py     # -> every number quoted in the paper, as LaTeX macros
python3 code/make_tables.py      # -> the paper's tables
python3 code/make_fig1.py ; python3 code/make_figures.py
```

Tested with Python 3.13, NumPy 2.2, pandas 2.2, scikit-learn 1.8, Matplotlib 3.10.

## Notes on the protocol

* Splitting is **cell-wise**, never random over cycles: neighbouring cycles of one cell
  are near-duplicates, and random splitting inflates accuracy substantially.
* No cycle-index, calendar-time or cumulative-throughput feature is used. Such features
  encode position along the ageing trajectory rather than cell state and are unavailable
  at first diagnosis of a cell of unknown history.
* Per-cycle discharge capacity is the **increment** of the discharge-capacity column
  across the discharge step; the Arbin files accumulate capacity within a file.
* Costs are model properties (bytes, MACs). No microcontroller timing is claimed.

## Citation

Gül, F.; [author 2]; Babu, M. *How Little Is Enough? Sensing- and Compute-Budget Limits
of On-Board State-of-Health Estimation for Lithium-Ion Batteries.* Submitted to
*Batteries*, 2026.

## Licence

Code: MIT. Derived data (`results/*.csv`): CC BY 4.0. The underlying raw cycling data
remain subject to the CALCE terms of use.
