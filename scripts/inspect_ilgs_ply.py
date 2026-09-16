from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_occ_bridge import load_ilgs_ply


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _axis_quantiles(values: np.ndarray) -> dict[str, list[float]]:
    levels = [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]
    return {
        f"q{level:g}": np.quantile(values, level, axis=0).astype(float).tolist()
        for level in levels
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect activated ILGS Gaussian parameters.")
    parser.add_argument("--ply", required=True)
    args = parser.parse_args()

    path = Path(args.ply)
    vertices = PlyData.read(str(path))["vertex"].data
    fields = vertices.dtype.names or ()
    batch = load_ilgs_ply(path, semantic_mode="constant")
    summary = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "gaussian_count": int(len(batch.means)),
        "field_count": len(fields),
        "scale_field_count": sum(name.startswith("scale_") for name in fields),
        "rotation_field_count": sum(name.startswith("rot_") for name in fields),
        "object_feature_count": sum(name.startswith("obj_dc_") for name in fields),
        "semantic_feature_count": sum(name.startswith("semantic_") for name in fields),
        "bbox_min": batch.means.min(axis=0).astype(float).tolist(),
        "bbox_max": batch.means.max(axis=0).astype(float).tolist(),
        "bbox_extent": np.ptp(batch.means, axis=0).astype(float).tolist(),
        "xyz_quantiles": _axis_quantiles(batch.means),
        "opacity": _stats(batch.opacities),
        "axis_scale_all": _stats(batch.axis_aligned_scales),
        "max_axis_scale_per_gaussian": _stats(batch.axis_aligned_scales.max(axis=1)),
        "filter_diagnostics": {},
    }
    max_scale = batch.axis_aligned_scales.max(axis=1)
    for opacity_threshold in (0.01, 0.05, 0.1, 0.2):
        mask = batch.opacities >= opacity_threshold
        summary["filter_diagnostics"][f"opacity_gte_{opacity_threshold:g}"] = {
            "count": int(mask.sum()),
            "bbox_min": batch.means[mask].min(axis=0).astype(float).tolist(),
            "bbox_max": batch.means[mask].max(axis=0).astype(float).tolist(),
        }
    for scale_threshold in (0.05, 0.1, 0.2):
        mask = max_scale <= scale_threshold
        summary["filter_diagnostics"][f"max_scale_lte_{scale_threshold:g}"] = {
            "count": int(mask.sum()),
            "bbox_min": batch.means[mask].min(axis=0).astype(float).tolist(),
            "bbox_max": batch.means[mask].max(axis=0).astype(float).tolist(),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
