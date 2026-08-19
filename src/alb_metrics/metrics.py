"""Point-level and grid/DBM-level metric computation.

The `@njit` functions do the per-point and per-cell work; the plain Python
functions around them build grid geometry and turn raw counts into the
ratios reported in the paper. See docs/metrics.md for the equation-by-
equation correspondence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

# --- Point-level classification --------------------------------------------


def _safe_div(n, d):
    return float(n / d) if d else math.nan


def metrics(tp, fp, tn, fn):
    """Point-level metrics from a confusion matrix (paper Eq. 1-6)."""
    n = tp + fp + tn + fn
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "N": n,
        "GA": _safe_div(tp + tn, n),
        "TNR": _safe_div(tn, tn + fp),
        "FNR": _safe_div(fn, tp + fn),
        "FPR": _safe_div(fp, tn + fp),
        "PAB_TPR": _safe_div(tp, tp + fn),
        "UAB_Precision": _safe_div(tp, tp + fp),
    }


@njit(cache=True)
def accumulate_point_confusion(reference_bathy, prediction_bathy):
    tp = fp = tn = fn = 0
    for i in range(reference_bathy.shape[0]):
        r, p = reference_bathy[i], prediction_bathy[i]
        if r:
            if p:
                tp += 1
            else:
                fn += 1
        else:
            if p:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn


# --- Grid / DBM --------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    x0: float
    y0: float
    resolution: float
    nx: int
    ny: int


def make_grid_spec(reference_bounds, prediction_bounds, resolution, origin=None):
    if resolution <= 0:
        raise ValueError("Grid resolution must be > 0.")
    x0, y0 = (reference_bounds[0], reference_bounds[2]) if origin is None else origin
    max_x = max(reference_bounds[1], prediction_bounds[1])
    max_y = max(reference_bounds[3], prediction_bounds[3])
    if max_x < x0 or max_y < y0:
        raise ValueError("Grid origin lies outside the input extent.")
    nx = int(np.floor((max_x - x0) / resolution)) + 1
    ny = int(np.floor((max_y - y0) / resolution)) + 1
    return GridSpec(float(x0), float(y0), float(resolution), nx, ny)


def new_dbm(spec):
    """Zero-initialized (sums, counts) accumulator arrays for `spec`."""
    shape = (spec.ny, spec.nx)
    return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.int64)


@njit(cache=True)
def accumulate_grid(x, y, z, bathy, x0, y0, resolution, sums, counts):
    ny, nx = sums.shape
    for i in range(x.shape[0]):
        if not bathy[i]:
            continue
        xi, yi, zi = x[i], y[i], z[i]
        if not (math.isfinite(xi) and math.isfinite(yi) and math.isfinite(zi)):
            continue
        # int() is required here: under Numba's njit, math.floor() returns a
        # float64, not a Python int, so the cast is needed for valid integer
        # array indexing (ruff flags this as redundant only because it
        # analyzes the un-compiled Python semantics).
        col = int(math.floor((xi - x0) / resolution))  # noqa: RUF046
        row = int(math.floor((yi - y0) / resolution))  # noqa: RUF046
        if 0 <= col < nx and 0 <= row < ny:
            sums[row, col] += zi
            counts[row, col] += 1


@njit(cache=True)
def accumulate_grid_metrics(
    reference_sum, reference_count, prediction_sum, prediction_count
):
    ny, nx = reference_sum.shape
    tp = fp = tn = fn = 0
    abs_error_sum = 0.0
    exceedances = 0
    for row in range(ny):
        for col in range(nx):
            rb = reference_count[row, col] > 0
            pb = prediction_count[row, col] > 0
            if rb:
                if pb:
                    tp += 1
                    rz = reference_sum[row, col] / reference_count[row, col]
                    pz = prediction_sum[row, col] / prediction_count[row, col]
                    error = abs(rz - pz)
                    abs_error_sum += error
                    threshold = math.sqrt(0.25 + (0.013 * rz) ** 2)
                    if error > threshold:
                        exceedances += 1
                else:
                    fn += 1
            else:
                if pb:
                    fp += 1
                else:
                    tn += 1
    return tp, fp, tn, fn, abs_error_sum, exceedances


@njit(cache=True)
def finalize_dbm(sums, counts):
    out = np.empty(sums.shape, dtype=np.float64)
    ny, nx = sums.shape
    for row in range(ny):
        for col in range(nx):
            if counts[row, col] > 0:
                out[row, col] = sums[row, col] / counts[row, col]
            else:
                out[row, col] = np.nan
    return out
