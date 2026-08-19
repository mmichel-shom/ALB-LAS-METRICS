import json
from pathlib import Path

import numpy as np


def _safe(v):
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    return v



def _format_number(key, value):
    if not isinstance(value, float):
        return None

    if key in {
        "FNR",
        "FPR",
        "GA",
        "PAB_TPR",
        "TNR",
        "UAB_Precision",
        "GB_PAB_TPR",
        "GB_UAB_Precision",
    }:
        return f"{value:.3f}"

    if key in {"MAE", "EER"}:
        return f"{value:.3e}"

    return None


def _json_text(data, indent=2, level=0):
    prefix = " " * (indent * level)
    child_prefix = " " * (indent * (level + 1))

    if isinstance(data, dict):
        if not data:
            return "{}"

        items = []
        for key, value in data.items():
            formatted = _format_number(key, value)

            if formatted is None:
                formatted = _json_text(value, indent, level + 1)

            items.append(
                f'{child_prefix}{json.dumps(str(key))}: {formatted}'
            )

        return "{\n" + ",\n".join(items) + f"\n{prefix}}}"

    if isinstance(data, list):
        if not data:
            return "[]"

        items = [
            f"{child_prefix}{_json_text(value, indent, level + 1)}"
            for value in data
        ]

        return "[\n" + ",\n".join(items) + f"\n{prefix}]"

    if isinstance(data, float):
        return json.dumps(data)

    return json.dumps(data)


def write_json(path, data):
    safe_data = _safe(data)

    Path(path).write_text(
        _json_text(safe_data) + "\n",
        encoding="utf-8",
    )


def write_dbm_geotiff(path, values, x0, y0, resolution, crs):
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(
        x0, y0 + values.shape[0] * resolution, resolution, resolution
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        nodata=np.nan,
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(np.flipud(values).astype(np.float32), 1)
