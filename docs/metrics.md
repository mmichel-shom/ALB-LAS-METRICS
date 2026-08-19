# Metrics: Paper / Code Correspondence

This document maps every metric in Section 2.2 of the paper to the exact function and output field that computes it, allowing the implementation to be checked equation-by-equation.

Notation: `TP`, `FP`, `TN`, and `FN` are point-level confusion-matrix counts; `TP_GB`, `FP_GB`, `TN_GB`, and `FN_GB` are the corresponding grid-based counts over DBM cells.

## Binary labeling

The evaluation uses a binary Bathy/NotBathy label derived exclusively from the LAS `withheld` field. The original LAS `classification` field is never read during evaluation.

For the NOAA reference data, the `withheld` field is constructed from the original LAS classification as:

```text
withheld = ~isin([2, 40])
```

where LAS class 2 corresponds to ground and class 40 corresponds to bathymetry. Therefore, points belonging to class 2 or 40 have `withheld=False`, while all other points have `withheld=True`.

## Point-level metrics

Point-level metrics are computed in `alb_metrics.metrics.metrics()` from the four raw `(TP, FP, TN, FN)` counts, built point-by-point in `alb_metrics.metrics.accumulate_point_confusion()`.

| Paper                                 | Eq. | Definition                                    | Code field      | Formula in code  |
| ------------------------------------- | --: | --------------------------------------------- | --------------- | ---------------- |
| Global Accuracy                       | (1) | Agreement over all classes                    | `GA`            | `(TP + TN) / N`  |
| True Negative Rate                    | (2) | NOAA NotBathy correctly identified            | `TNR`           | `TN / (TN + FP)` |
| False Negative Rate                   | (3) | NOAA Bathy incorrectly identified as NotBathy | `FNR`           | `FN / (TP + FN)` |
| False Positive Rate                   | (4) | NOAA NotBathy incorrectly identified as Bathy | `FPR`           | `FP / (TN + FP)` |
| Producer's Accuracy for Bathy / TPR   | (5) | NOAA Bathy correctly identified               | `PAB_TPR`       | `TP / (TP + FN)` |
| User's Accuracy for Bathy / Precision | (6) | Predicted Bathy points that are NOAA Bathy    | `UAB_Precision` | `TP / (TP + FP)` |

All divisions by zero return `NaN` via `alb_metrics.metrics._safe_div` rather than raising. A degenerate input, such as one containing no NOAA Bathy points, therefore produces a diagnosable `null` in the JSON output instead of crashing the run.

## Grid/DBM-level metrics

The DBM is built independently for the reference and prediction on a **shared** grid using `alb_metrics.metrics.make_grid_spec()`, with the same origin, resolution, and extent for both inputs.

For each grid cell, the DBM elevation is the mean Z of the valid (finite X/Y/Z) Bathy points falling within that cell, accumulated by `alb_metrics.metrics.accumulate_grid()` and finalized by `finalize_dbm()`.

A grid cell is considered positive when it contains at least one valid Bathy point. The grid-based confusion matrix is computed in `alb_metrics.metrics.accumulate_grid_metrics()` by comparing, for every cell, whether the reference and/or prediction DBM contains Bathy points.

| Paper                                             |  Eq. | Definition                                              | Code field         | Formula in code           |                |                       |
| ------------------------------------------------- | ---: | ------------------------------------------------------- | ------------------ | ------------------------- | -------------- | --------------------- |
| Grid-Based Producer's Accuracy for Bathy / GB-TPR |  (7) | NOAA Bathy cells correctly identified                   | `GB_PAB_TPR`       | `TP_GB / (TP_GB + FN_GB)` |                |                       |
| Grid-Based User's Accuracy for Bathy / Precision  |  (8) | Cells classified Bathy that NOAA classified Bathy       | `GB_UAB_Precision` | `TP_GB / (TP_GB + FP_GB)` |                |                       |
| Mean Absolute Error                               |  (9) | Mean absolute vertical error, TP_GB cells only          | `MAE`              | `sum(                     | z_ref - z_pred | ) / TP_GB`            |
| Error Exceedance Ratio                            | (10) | Fraction of TP_GB cells exceeding IHO S-44 Order 1a TVU | `EER`              | `count(                   | z_ref - z_pred | > threshold) / TP_GB` |

The IHO S-44 Order 1a maximum allowable Total Vertical Uncertainty (TVU) is evaluated per cell using the reference DBM elevation:

```text
threshold = sqrt(0.5^2 + (0.013 * z_ref)^2)
```

A cell is an exceedance when its absolute vertical error is **strictly greater** than this threshold.

MAE and EER are restricted to `TP_GB` cells, i.e. cells that are positive in both the reference and prediction DBMs. These are the only cells for which both a reference and prediction elevation are available for comparison. The evaluation records this domain explicitly in `metrics.json` under `metadata.grid_error_domain`.

## Grid confusion-matrix scope

The grid pass in `accumulate_grid_metrics()` iterates over **every cell of the shared grid**, including unpopulated cells, to build the complete `TP_GB`/`FP_GB`/`TN_GB`/`FN_GB` confusion matrix.

Consequently, `TN_GB` includes cells containing no valid Bathy point in either the reference or prediction DBM. `N_GB` in `metrics.json` reports the resulting total number of grid cells, making the evaluation scope auditable from the output itself.

The grid is sized tightly around the union of the two LAS extents. No additional padding is introduced beyond the partial cells required at the edges. Therefore, `TN_GB` and `N_GB` reflect empty space within the evaluated survey extent rather than artificial grid-construction padding.

## Implementation and reproducibility notes

Reference and prediction must represent classifications of the **same underlying point cloud**. `alb-metrics` verifies that both files contain the same number of points and that their X/Y/Z coordinates match within the configured `--coordinate-atol` (default: `1e-8`) before computing the metrics.

LAS/LAZ files are processed chunk-by-chunk with `laspy`, so the complete point cloud is never loaded into memory. Point-level confusion, DBM accumulation, and grid-metric passes use compiled Numba kernels rather than Python loops over individual points.

DBM sums and counts use deterministic serial accumulation rather than floating-point atomics, providing bit-for-bit reproducibility across runs on the same machine.
