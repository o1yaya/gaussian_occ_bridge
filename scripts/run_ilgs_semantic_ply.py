from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_occ_bridge import (
    VoxelGridSpec,
    gaussian_to_voxels_hard_labels,
    load_ilgs_ply_hard_labels,
    reduce_hard_labels_to_bev,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vec3(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("voxel-size requires three values")
    return tuple(float(value) for value in values)


def _auto_grid(
    means: np.ndarray,
    scales: np.ndarray,
    voxel_size: tuple[float, float, float],
    crop_quantile: float,
    padding_sigma: float,
) -> VoxelGridSpec:
    if not 0.0 <= crop_quantile < 0.5:
        raise ValueError("auto-grid-crop-quantile must be in [0, 0.5)")
    lower = np.quantile(means, crop_quantile, axis=0)
    upper = np.quantile(means, 1.0 - crop_quantile, axis=0)
    padding = padding_sigma * np.quantile(scales, 0.99, axis=0)
    size = np.asarray(voxel_size, dtype=np.float64)
    lower = np.floor((lower - padding) / size) * size
    upper = np.ceil((upper + padding) / size) * size
    upper = np.maximum(upper, lower + size)
    return VoxelGridSpec(tuple(lower), tuple(upper), voxel_size)


def _histogram(values: np.ndarray, limit: int = 20) -> list[dict[str, int]]:
    labels, counts = np.unique(values[values >= 0], return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [
        {"object_id": int(labels[index]), "count": int(counts[index])}
        for index in order
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-efficient ILGS object-ID occupancy baseline."
    )
    parser.add_argument("--ply", required=True)
    parser.add_argument("--classifier-npz", required=True)
    parser.add_argument("--output-dir", default="outputs/ilgs_semantic")
    parser.add_argument("--voxel-size", nargs=3, type=float, default=[0.10, 0.10, 0.10])
    parser.add_argument("--min-opacity", type=float, default=0.10)
    parser.add_argument("--max-axis-scale", type=float, default=0.20)
    parser.add_argument("--max-gaussians", type=int, default=200_000)
    parser.add_argument("--auto-grid-crop-quantile", type=float, default=0.01)
    parser.add_argument("--padding-sigma", type=float, default=3.0)
    parser.add_argument("--radius-sigma", type=float, default=3.0)
    parser.add_argument("--unknown-evidence-threshold", type=float, default=1e-3)
    parser.add_argument("--max-voxels", type=int, default=32_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.ply)
    classifier_path = Path(args.classifier_npz)
    voxel_size = _vec3(args.voxel_size)

    load_start = time.perf_counter()
    batch = load_ilgs_ply_hard_labels(
        source_path,
        classifier_path,
        min_opacity=args.min_opacity,
        max_axis_scale=args.max_axis_scale,
        max_gaussians=args.max_gaussians,
    )
    load_seconds = time.perf_counter() - load_start
    grid = _auto_grid(
        batch.means,
        batch.axis_aligned_scales,
        voxel_size,
        args.auto_grid_crop_quantile,
        args.padding_sigma,
    )
    voxel_count = int(np.prod(grid.shape, dtype=np.int64))
    if voxel_count > args.max_voxels:
        raise RuntimeError(
            f"Grid {grid.shape} contains {voxel_count:,} voxels, above "
            f"--max-voxels {args.max_voxels:,}."
        )
    grid_mask = np.all(batch.means >= grid.mins, axis=1) & np.all(
        batch.means < grid.maxs, axis=1
    )
    means = batch.means[grid_mask]
    scales = batch.axis_aligned_scales[grid_mask]
    opacities = batch.opacities[grid_mask]
    semantic_ids = batch.semantic_ids[grid_mask]
    source_indices = batch.source_indices[grid_mask]

    splat_start = time.perf_counter()
    voxels = gaussian_to_voxels_hard_labels(
        means,
        scales,
        opacities,
        semantic_ids,
        grid,
        radius_sigma=args.radius_sigma,
        unknown_evidence_threshold=args.unknown_evidence_threshold,
    )
    bev = reduce_hard_labels_to_bev(
        voxels, unknown_evidence_threshold=args.unknown_evidence_threshold
    )
    splat_seconds = time.perf_counter() - splat_start

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "semantic_occupancy.npz"
    np.savez_compressed(
        output_path,
        occupancy=voxels.occupancy,
        evidence=voxels.evidence,
        semantic_ids=voxels.semantic_ids,
        label_score=voxels.label_score,
        unknown_mask=voxels.unknown_mask,
        bev_occupancy=bev.occupancy,
        bev_evidence=bev.evidence,
        bev_semantic_ids=bev.semantic_ids,
        bev_label_score=bev.label_score,
        bev_unknown_mask=bev.unknown_mask,
        grid_min=np.asarray(grid.min_xyz),
        grid_max=np.asarray(grid.max_xyz),
        voxel_size=np.asarray(grid.voxel_size_xyz),
        source_indices=source_indices,
    )
    summary = {
        "source_ply": source_path.name,
        "source_file_size_bytes": source_path.stat().st_size,
        "source_sha256": _sha256(source_path),
        "classifier_npz": classifier_path.name,
        "classifier_sha256": _sha256(classifier_path),
        "semantic_source": batch.semantic_source,
        "hard_label_rule": "strongest individual Gaussian evidence per voxel",
        "gaussians_after_filtering_and_topk": int(len(batch.means)),
        "gaussians_inside_grid": int(len(means)),
        "active_object_ids_in_gaussians": int(np.unique(semantic_ids).size),
        "top_object_ids_by_gaussian_count": _histogram(semantic_ids),
        "active_object_ids_in_bev": int(np.unique(bev.semantic_ids[bev.semantic_ids >= 0]).size),
        "top_object_ids_by_bev_cell_count": _histogram(bev.semantic_ids),
        "min_opacity": float(args.min_opacity),
        "max_axis_scale": float(args.max_axis_scale),
        "max_gaussians": int(args.max_gaussians),
        "auto_grid_crop_quantile": float(args.auto_grid_crop_quantile),
        "radius_sigma": float(args.radius_sigma),
        "unknown_evidence_threshold": float(args.unknown_evidence_threshold),
        "grid_min": grid.mins.astype(float).tolist(),
        "grid_max": grid.maxs.astype(float).tolist(),
        "voxel_size": grid.voxel_sizes.astype(float).tolist(),
        "grid_shape_xyz": grid.shape,
        "voxel_count": voxel_count,
        "occupied_voxels_at_0_5": int((voxels.occupancy >= 0.5).sum()),
        "known_voxels": int((~voxels.unknown_mask).sum()),
        "bev_known_cells": int((~bev.unknown_mask).sum()),
        "load_and_classify_seconds": load_seconds,
        "splat_and_reduce_seconds": splat_seconds,
        "coordinate_note": "Reconstruction coordinates; axis 2 is not verified gravity.",
        "semantic_note": "Object IDs are classifier indices, not human-readable categories.",
        "free_space_note": "Cells without Gaussian evidence remain unknown, not free.",
        "output": output_path.as_posix(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

