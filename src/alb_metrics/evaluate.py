import math
from pathlib import Path

import laspy
import numpy as np

from .metrics import (
    accumulate_grid,
    accumulate_grid_metrics,
    accumulate_point_confusion,
    finalize_dbm,
    make_grid_spec,
    new_dbm,
)
from .metrics import metrics as classification_metrics


def _bounds(reader):
    """Conservative (x_min, x_max, y_min, y_max) envelope for one LAS/LAZ file.

    Only the lower bounds are snapped outward to an integer, so that the
    default grid origin (`make_grid_spec`) lands on a round coordinate. The
    upper bounds are left as the exact header values: `make_grid_spec`
    already grows them to the next full cell via `floor(...) + 1`, so
    rounding them up here first would double-pad the grid. Before this fix,
    `np.ceil(h.x_max)` (and `y_max`) rounded the upper bound out to the
    nearest *integer*, one full extra cell beyond what `+ 1` already adds
    whenever `resolution <= 1`, and up to `2 / resolution` extra empty
    cells for sub-1 resolutions. That inflated `nx`/`ny` (and therefore
    `TN_GB`/`N_GB`) and produced DBM rasters with a spurious empty
    NaN border row/column that isn't part of the actual data extent.
    """
    h = reader.header
    return np.floor(h.x_min), h.x_max, np.floor(h.y_min), h.y_max


def evaluate(
    reference_path,
    prediction_path,
    *,
    resolution,
    origin=None,
    chunk_size=1_000_000,
    coordinate_atol=1e-8,
):
    with (
        laspy.open(Path(reference_path)) as ref_reader,
        laspy.open(Path(prediction_path)) as pred_reader,
    ):
        for reader, name in ((ref_reader, "Reference"), (pred_reader, "Prediction")):
            if "withheld" not in reader.header.point_format.dimension_names:
                raise ValueError(f"{name} LAS/LAZ must contain the withheld flag.")
        if ref_reader.header.point_count != pred_reader.header.point_count:
            raise ValueError(
                "Reference and prediction must contain the same number of points."
            )
        ref_crs = ref_reader.header.parse_crs()
        pred_crs = pred_reader.header.parse_crs()

        if ref_crs != pred_crs:
            raise ValueError("Reference and prediction CRS differ.")

        crs = ref_crs

        spec = make_grid_spec(
            _bounds(ref_reader), _bounds(pred_reader), resolution, origin
        )
        ref_sums, ref_counts = new_dbm(spec)
        pred_sums, pred_counts = new_dbm(spec)
        tp = fp = tn = fn = processed = 0

        for ref_chunk, pred_chunk in zip(
            ref_reader.chunk_iterator(chunk_size),
            pred_reader.chunk_iterator(chunk_size),
            strict=True,
        ):
            if len(ref_chunk) != len(pred_chunk):
                raise ValueError("LAS chunk alignment differs between inputs.")
            rx = np.asarray(ref_chunk.x, dtype=np.float64)
            ry = np.asarray(ref_chunk.y, dtype=np.float64)
            rz = np.asarray(ref_chunk.z, dtype=np.float64)
            px = np.asarray(pred_chunk.x, dtype=np.float64)
            py = np.asarray(pred_chunk.y, dtype=np.float64)
            pz = np.asarray(pred_chunk.z, dtype=np.float64)
            if not (
                np.allclose(rx, px, rtol=0, atol=coordinate_atol, equal_nan=True)
                and np.allclose(ry, py, rtol=0, atol=coordinate_atol, equal_nan=True)
                and np.allclose(rz, pz, rtol=0, atol=coordinate_atol, equal_nan=True)
            ):
                raise ValueError(
                    f"Reference/prediction coordinates differ near point {processed}."
                )
            rb = ~np.asarray(ref_chunk.withheld, dtype=np.bool_)
            pb = ~np.asarray(pred_chunk.withheld, dtype=np.bool_)
            a, b, c, d = accumulate_point_confusion(rb, pb)
            tp, fp, tn, fn = tp + a, fp + b, tn + c, fn + d
            g0 = (spec.x0, spec.y0, spec.resolution)
            accumulate_grid(rx, ry, rz, rb, *g0, ref_sums, ref_counts)
            accumulate_grid(px, py, pz, pb, *g0, pred_sums, pred_counts)
            processed += len(ref_chunk)
        cls = classification_metrics(tp, fp, tn, fn)
        gtp, gfp, gtn, gfn, error_sum, exceedances = accumulate_grid_metrics(
            ref_sums, ref_counts, pred_sums, pred_counts
        )
        grid = {
            "TP_GB": int(gtp),
            "FP_GB": int(gfp),
            "TN_GB": int(gtn),
            "FN_GB": int(gfn),
            "N_GB": int(gtp + gfp + gtn + gfn),
            "GB_PAB_TPR": float(gtp / (gtp + gfn)) if gtp + gfn else math.nan,
            "GB_UAB_Precision": float(gtp / (gtp + gfp)) if gtp + gfp else math.nan,
            "MAE": float(error_sum / gtp) if gtp else math.nan,
            "EER": float(exceedances / gtp) if gtp else math.nan,
            "EER_exceedances": int(exceedances),
        }
        metadata = {
            "reference": str(Path(reference_path)),
            "prediction": str(Path(prediction_path)),
            "point_count": processed,
            "crs": crs.to_wkt() if crs is not None else None,
            "grid": {
                "x0": spec.x0,
                "y0": spec.y0,
                "resolution": spec.resolution,
                "nx": spec.nx,
                "ny": spec.ny,
            },
            "eer_threshold": "sqrt(0.5^2 + (0.013 * DBM_NOAA)^2)",
            "grid_error_domain": "TP_GB cells only (cells positive in both "
            "reference and prediction DBMs); see docs/metrics.md.",
        }
    ref_dbm = finalize_dbm(ref_sums, ref_counts)
    pred_dbm = finalize_dbm(pred_sums, pred_counts)
    return cls, grid, ref_dbm, pred_dbm, metadata
