# ALB LAS Metrics

Reproducible and efficient evaluation code for the ALB benchmark described in *"Operational Evaluation of Submanifold Sparse Convolutional Neural Networks for Airborne LiDAR Bathymetry Processing."*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## Scope

This repository is released in full alongside the paper and provides everything needed to independently recompute the benchmark metrics from a pair of classified LAS/LAZ point clouds.

| Content                                                          | Status                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| Point-level metrics: GA, TNR, FNR, FPR, PAB/TPR, UAB/Precision   | ✅ Released                                                         |
| Grid/DBM-based metrics: GB-PAB, GB-UAB, MAE, EER                 | ✅ Released                                                         |
| Trained SSCN model and weights (`resources/model/`)              | ✅ Released                                                         |
| SSCN training pipeline (internal-infrastructure-dependent parts) | 🔒 Available from the corresponding author upon reasonable request |

**What this repository is not:** a turnkey pipeline for reproducing the paper's trained model from raw data. The trained weights and inference code in `resources/model/` are provided for inspection and reuse independently of the evaluation framework. They are not required to run `alb-metrics`. The evaluation framework is entirely model-agnostic and can evaluate the LAS/LAZ output of any classifier, not only the provided SSCN model.

## Quickstart

`examples/data/` contains a real NOAA reference LAS/LAZ file (`20160424_420500e_2717500n`). An additional `withheld` field has been added to the reference file to define the binary ground-filtering labels: LAS class 2 corresponds to ground and class 40 to bathymetry, and points outside these two classes are marked as `withheld`, i.e. `withheld = ~isin([2, 40])`. A matching prediction LAS/LAZ file is also provided, produced with the trained model released in `resources/model/`.

```bash
pip install -e .

alb-metrics examples/data/reference.las examples/data/prediction.las \
  --resolution 1.0 --output results
```

See [`examples/README.md`](examples/README.md) for details.

## Evaluation workflow

```bash
alb-metrics reference.las prediction.las \
  --resolution 1.0 \
  --output results
```

The Bathy/NotBathy decision is derived exclusively from the `withheld` field; the LAS `classification` dimension is never read. The reference and prediction must represent two classifications of the **same underlying point cloud**. `alb-metrics` verifies that both files contain the same number of points and that their X/Y/Z coordinates match within `--coordinate-atol` (default: `1e-8`). Misaligned inputs therefore raise an error instead of being silently compared.

For the reference data, the binary label is derived from the original LAS classification using:

```text
withheld = ~isin([2, 40])
```

where LAS class 2 denotes ground and class 40 denotes bathymetry. Thus, points belonging to classes 2 or 40 have `withheld=False`, while all other points have `withheld=True`. During evaluation, only `withheld` is used to determine the binary label; the original `classification` field is ignored.

## Metrics

### Point-level metrics

GA, TNR, FNR, FPR, PAB/TPR, and UAB/Precision are computed from a single point-level confusion matrix over the complete input point cloud.

### Grid/DBM metrics

For each input, a Digital Bathymetric Model (DBM) is accumulated on a shared regular grid. A cell is considered positive when it contains at least one valid (finite X/Y/Z) Bathy point, and its elevation is computed as the mean Z of those points.

GB-PAB/TPR, GB-UAB/Precision, MAE, and EER are then computed from a second confusion matrix over grid cells, comparing reference-positive and prediction-positive cells. MAE and EER are evaluated only on TP_GB cells, i.e. cells that are positive in both reference and prediction, since only these cells have both reference and prediction elevations available.

EER uses the IHO S-44 Order 1a maximum allowable Total Vertical Uncertainty (TVU) defined in the paper:

```text
threshold = sqrt(0.5^2 + (0.013 * z_ref)^2)
```

where `z_ref` is the reference DBM elevation of the cell. A cell is counted as an exceedance when its absolute vertical error is strictly greater than this threshold. `EER` is the fraction of TP_GB cells exceeding the threshold.

See [`docs/metrics.md`](docs/metrics.md) for the full equation-by-equation correspondence with Section 2.2 of the paper.

## Efficiency and reproducibility

* LAS/LAZ files are read chunk-by-chunk with `laspy`; the complete point cloud is never loaded into memory during evaluation.
* DBM accumulation, point-level confusion, and grid-metric passes use compiled **Numba** kernels (`@njit` functions in `src/alb_metrics/metrics.py`), avoiding Python loops over individual points.
* DBM sums and counts use deterministic serial accumulation rather than floating-point atomics, ensuring bit-for-bit reproducibility across runs on the same machine.
* On the provided ~10.4 million-point dataset, the complete evaluation runs in a few seconds on a single CPU core.

## Grid definition

By default, the grid origin is the reference LAS minimum X/Y, each coordinate floored to an integer. The grid dimensions tightly cover the union of the two LAS extents, with no additional padding beyond the partial cell required at each edge. Both DBMs therefore use exactly the same origin, resolution, and dimensions.

An explicit origin can be provided with:

```bash
alb-metrics ... --origin X0,Y0
```

## Installation

```bash
pip install .
```

For LAZ support:

```bash
pip install "alb-las-metrics[laz]"
```

## Outputs

The evaluation produces:

* `metrics.json` — all point-level and grid-level metrics, together with run metadata such as the grid specification, point count, and EER threshold definition.
* `classification_metrics.json` — point-level metrics only.
* `grid_metrics.json` — grid/DBM-level metrics only.
* `dbm_reference.csv`, `dbm_prediction.csv` — per-cell elevations, with one row per populated cell.

Optional GeoTIFF DBMs can be generated with:

```bash
pip install "alb-las-metrics[raster]"
alb-metrics ... --write-dbms
```

This produces `dbm_reference.tif` and `dbm_prediction.tif` in the output directory.

## Repository structure

* `src/alb_metrics/` — metric computation and evaluation code.
* `examples/` — example LAS/LAZ inputs and evaluation outputs.
* `docs/metrics.md` — equation-by-equation correspondence with Section 2.2 of the paper.
* `resources/model/` — released SSCN model, weights, and inference code.
* `CHANGES.md` — documented corrections and changes made during revision.

## Reproducibility

This repository releases the trained SSCN model and weights, the evaluation framework, and the complete metric-computation code, including both point-level and DBM-based metrics and the reference-preparation protocol described in the paper. The benchmarking framework can therefore be independently run and inspected without access to the original training infrastructure.

The implementation has been checked against the equations and reported results in Section 2.2 of the paper. The correspondence between each paper metric, its implementation, and its output field is documented in [`docs/metrics.md`](docs/metrics.md).

The parts of the SSCN training pipeline that depend on internal infrastructure are not included in this repository and are available from the corresponding author upon reasonable request. See [`CHANGES.md`](CHANGES.md) for a complete account of the issues identified and fixed in this revision.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). The license for the model weights in `resources/model/` is specified separately in that directory.
