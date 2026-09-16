from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_occ_bridge import (
    VoxelGridSpec,
    gaussian_to_voxels,
    load_ilgs_ply,
    reduce_to_bev,
)


def _vec3(values: list[float] | None, name: str) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(f"{name} requires exactly three values")
    return tuple(float(value) for value in values)


def _load_transform(path: str) -> np.ndarray | None:
    if not path:
        return None
    transform = np.asarray(json.loads(Path(path).read_text(encoding="utf-8")), dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform JSON must contain a 4x4 nested list")
    return transform


def _auto_grid(
    means: np.ndarray,
    scales: np.ndarray,
    voxel_size: tuple[float, float, float],
    padding_sigma: float,
) -> VoxelGridSpec:
    lower = np.min(means - padding_sigma * scales, axis=0)
    upper = np.max(means + padding_sigma * scales, axis=0)
    size = np.asarray(voxel_size, dtype=np.float64)
    lower = np.floor(lower / size) * size
    upper = np.ceil(upper / size) * size
    upper = np.maximum(upper, lower + size)
    return VoxelGridSpec(tuple(lower), tuple(upper), voxel_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a real ILGS point_cloud.ply or query subset to voxel occupancy and BEV."
    )
    parser.add_argument("--ply", required=True, help="ILGS point_cloud.ply or target_gaussians_full.ply")
    parser.add_argument("--output-dir", default="outputs/ilgs_ply")
    parser.add_argument(
        "--semantic-mode",
        choices=["constant", "semantic-fields", "object-id", "object-classifier"],
        default="constant",
    )
    parser.add_argument("--classifier-npz", default="")
    parser.add_argument("--transform-json", default="")
    parser.add_argument("--voxel-size", nargs=3, type=float, default=[0.05, 0.05, 0.05])
    parser.add_argument("--grid-min", nargs=3, type=float)
    parser.add_argument("--grid-max", nargs=3, type=float)
    parser.add_argument("--padding-sigma", type=float, default=3.0)
    parser.add_argument("--radius-sigma", type=float, default=3.0)
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--max-axis-scale", type=float, default=None)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--max-voxels", type=int, default=64_000_000)
    parser.add_argument("--unknown-evidence-threshold", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    voxel_size = _vec3(args.voxel_size, "voxel-size")
    transform = _load_transform(args.transform_json)
    batch = load_ilgs_ply(
        args.ply,
        semantic_mode=args.semantic_mode,
        classifier_npz=args.classifier_npz or None,
        transform=transform,
        min_opacity=args.min_opacity,
        max_axis_scale=args.max_axis_scale,
        max_gaussians=args.max_gaussians,
    )
    if len(batch.means) == 0:
        raise RuntimeError("No Gaussians remain after filtering")

    grid_min = _vec3(args.grid_min, "grid-min")
    grid_max = _vec3(args.grid_max, "grid-max")
    if (grid_min is None) != (grid_max is None):
        raise ValueError("grid-min and grid-max must be provided together")
    grid = (
        VoxelGridSpec(grid_min, grid_max, voxel_size)
        if grid_min is not None
        else _auto_grid(batch.means, batch.axis_aligned_scales, voxel_size, args.padding_sigma)
    )
    voxel_count = int(np.prod(grid.shape, dtype=np.int64))
    if voxel_count > args.max_voxels:
        raise RuntimeError(
            f"Grid {grid.shape} contains {voxel_count:,} voxels, above --max-voxels "
            f"{args.max_voxels:,}. Increase voxel size or pass an explicit local grid."
        )

    voxels = gaussian_to_voxels(
        batch.means,
        batch.axis_aligned_scales,
        batch.opacities,
        batch.semantic_logits,
        grid,
        radius_sigma=args.radius_sigma,
        unknown_evidence_threshold=args.unknown_evidence_threshold,
    )
    bev = reduce_to_bev(
        voxels, unknown_evidence_threshold=args.unknown_evidence_threshold
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "ilgs_occupancy.npz",
        occupancy=voxels.occupancy,
        semantic_probs=voxels.semantic_probs,
        evidence=voxels.evidence,
        unknown_mask=voxels.unknown_mask,
        bev_occupancy=bev.occupancy,
        bev_semantic_probs=bev.semantic_probs,
        bev_unknown_mask=bev.unknown_mask,
        grid_min=np.asarray(grid.min_xyz),
        grid_max=np.asarray(grid.max_xyz),
        voxel_size=np.asarray(grid.voxel_size_xyz),
        source_indices=batch.source_indices,
    )
    summary = {
        "source_ply": str(Path(args.ply)),
        "gaussians_after_filtering": int(len(batch.means)),
        "semantic_source": batch.semantic_source,
        "grid_shape_xyz": grid.shape,
        "voxel_count": voxel_count,
        "occupied_voxels_at_0_5": int((voxels.occupancy >= 0.5).sum()),
        "known_voxels": int((~voxels.unknown_mask).sum()),
        "bev_known_cells": int((~bev.unknown_mask).sum()),
        "max_occupancy": float(voxels.occupancy.max()),
        "coordinate_note": (
            "Metric meaning depends on the supplied transform. Without --transform-json, "
            "the output remains in ILGS reconstruction coordinates."
        ),
        "free_space_note": "Cells without Gaussian evidence remain unknown, not free.",
        "output": str(output_dir / "ilgs_occupancy.npz"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

