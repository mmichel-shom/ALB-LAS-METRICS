"""Command-line entry point: `alb-metrics`."""

import argparse
from pathlib import Path

from .evaluate import evaluate
from .io import write_dbm_geotiff, write_json


def _origin(value):
    if value == "auto":
        return None
    try:
        x, y = value.split(",", 1)
        return float(x), float(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Origin must be auto or X0,Y0.") from exc


def metrics_main(argv=None):
    p = argparse.ArgumentParser(description="Compute ALB LAS/LAZ benchmark metrics.")
    p.add_argument("reference", type=Path)
    p.add_argument("prediction", type=Path)
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--origin", type=_origin, default=None)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--coordinate-atol", type=float, default=1e-8)
    p.add_argument("--write-dbms", action="store_true")
    a = p.parse_args(argv)
    if a.resolution <= 0 or a.chunk_size <= 0:
        p.error("resolution and chunk-size must be > 0")
    a.output.mkdir(parents=True, exist_ok=True)
    cls, grid, ref, pred, meta = evaluate(
        a.reference,
        a.prediction,
        resolution=a.resolution,
        origin=a.origin,
        chunk_size=a.chunk_size,
        coordinate_atol=a.coordinate_atol,
    )
    s = meta["grid"]

    write_json(
        a.output / "metrics.json",
        {"classification": cls, "grid": grid, "metadata": meta},
    )

    if a.write_dbms:
        write_dbm_geotiff(
            a.output / "dbm_reference.tif", ref, s["x0"], s["y0"], s["resolution"], meta["crs"]
        )
        write_dbm_geotiff(
            a.output / "dbm_prediction.tif", pred, s["x0"], s["y0"], s["resolution"], meta["crs"]
        )

if __name__ == "__main__":
    metrics_main()
