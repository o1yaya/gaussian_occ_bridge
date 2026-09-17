from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_occ_bridge import load_ilgs_ply_hard_labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _robust_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.quantile(points, 0.01, axis=0), np.quantile(points, 0.99, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit query-to-object-ID mappings against the complete ILGS scene."
    )
    parser.add_argument("--ply", required=True, help="Complete ILGS scene PLY")
    parser.add_argument("--classifier-npz", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--bev-summary", required=True)
    parser.add_argument("--output", required=True, help="Output audit JSON")
    parser.add_argument("--figure", required=True, help="Output diagnostic PNG")
    parser.add_argument("--spatial-figure", required=True, help="Output per-ID projection PNG")
    parser.add_argument("--spatial-query", default="chopsticks")
    parser.add_argument("--max-spatial-points-per-id", type=int, default=8000)
    parser.add_argument("--warning-scene-fraction", type=float, default=0.10)
    parser.add_argument("--warning-bev-fraction", type=float, default=0.10)
    parser.add_argument("--warning-count-error", type=float, default=0.01)
    args = parser.parse_args()

    source_path = Path(args.ply)
    classifier_path = Path(args.classifier_npz)
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    bev_summary = json.loads(Path(args.bev_summary).read_text(encoding="utf-8"))
    batch = load_ilgs_ply_hard_labels(source_path, classifier_path)

    scene_lower, scene_upper = _robust_bounds(batch.means)
    scene_extent = np.maximum(scene_upper - scene_lower, 1e-12)
    scene_count = int(len(batch.means))
    known_bev_cells = int(bev_summary["bounded_full_scene_bev_known_cells"])
    audits: dict[str, dict[str, object]] = {}

    for query, info in mapping["queries"].items():
        object_ids = np.asarray(info["object_ids"], dtype=np.int32)
        mask = np.isin(batch.semantic_ids, object_ids)
        points = batch.means[mask]
        full_count = int(mask.sum())
        if full_count == 0:
            raise ValueError(f"Query {query!r} has no mapped-ID Gaussians in the scene")
        export_count = int(info["selected_gaussians"])
        count_error = abs(full_count - export_count) / max(export_count, 1)
        lower, upper = _robust_bounds(points)
        extent = upper - lower
        per_id = {
            str(int(object_id)): int(np.count_nonzero(batch.semantic_ids == object_id))
            for object_id in object_ids
        }
        bev_cells = int(bev_summary["queries"][query]["bev_cells"])
        scene_fraction = full_count / scene_count
        bev_fraction = bev_cells / known_bev_cells if known_bev_cells else 0.0
        warnings: list[str] = []
        if scene_fraction >= args.warning_scene_fraction:
            warnings.append("mapped IDs cover an unusually large fraction of scene Gaussians")
        if bev_fraction >= args.warning_bev_fraction:
            warnings.append("mapped IDs cover an unusually large fraction of known BEV cells")
        if count_error >= args.warning_count_error:
            warnings.append("query-export count does not match full-scene membership of mapped IDs")
        if "quality_note" in info:
            warnings.append(str(info["quality_note"]))

        audits[query] = {
            "object_ids": object_ids.astype(int).tolist(),
            "query_export_gaussians": export_count,
            "full_scene_gaussians_with_mapped_ids": full_count,
            "count_difference": full_count - export_count,
            "relative_count_error": count_error,
            "fraction_of_all_scene_gaussians": scene_fraction,
            "gaussians_by_object_id": per_id,
            "centroid_reconstruction_xyz": points.mean(axis=0).astype(float).tolist(),
            "robust_q01_q99_min_reconstruction_xyz": lower.astype(float).tolist(),
            "robust_q01_q99_max_reconstruction_xyz": upper.astype(float).tolist(),
            "robust_extent_reconstruction_xyz": extent.astype(float).tolist(),
            "robust_extent_fraction_of_scene_xyz": (extent / scene_extent).astype(float).tolist(),
            "bounded_bev_cells": bev_cells,
            "fraction_of_known_bev_cells": bev_fraction,
            "warnings": list(dict.fromkeys(warnings)),
            "audit_status": "warning" if warnings else "no-threshold-warning",
        }

    output = {
        "scene": mapping["scene"],
        "source_ply": source_path.name,
        "source_ply_sha256": _sha256(source_path),
        "classifier_npz": classifier_path.name,
        "classifier_npz_sha256": _sha256(classifier_path),
        "scene_gaussian_count": scene_count,
        "scene_robust_q01_q99_min_reconstruction_xyz": scene_lower.astype(float).tolist(),
        "scene_robust_q01_q99_max_reconstruction_xyz": scene_upper.astype(float).tolist(),
        "thresholds": {
            "warning_scene_fraction": args.warning_scene_fraction,
            "warning_bev_fraction": args.warning_bev_fraction,
            "warning_count_error": args.warning_count_error,
        },
        "coordinate_note": "All spatial statistics use reconstruction coordinates; no gravity or metric robot frame is assumed.",
        "interpretation": "Query labels come from mask-vote exports. Threshold warnings identify mappings that require visual audit; they do not prove semantic failure.",
        "queries": audits,
    }

    output_path = Path(args.output)
    figure_path = Path(args.figure)
    spatial_figure_path = Path(args.spatial_figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    spatial_figure_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    names = list(audits)
    colors = ["#E15759" if audits[name]["warnings"] else "#4E79A7" for name in names]
    export_counts = [audits[name]["query_export_gaussians"] for name in names]
    scene_fractions = [100.0 * audits[name]["fraction_of_all_scene_gaussians"] for name in names]
    bev_fractions = [100.0 * audits[name]["fraction_of_known_bev_cells"] for name in names]
    positions = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    axes[0].bar(positions, export_counts, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("query-export Gaussians (log scale)")
    axes[0].set_title("Export size")
    axes[1].bar(positions, scene_fractions, color=colors)
    axes[1].axhline(100.0 * args.warning_scene_fraction, color="#E15759", linestyle="--")
    axes[1].set_ylabel("fraction of full scene (%)")
    axes[1].set_title("Mapped-ID scene coverage")
    axes[2].bar(positions, bev_fractions, color=colors)
    axes[2].axhline(100.0 * args.warning_bev_fraction, color="#E15759", linestyle="--")
    axes[2].set_ylabel("fraction of known BEV cells (%)")
    axes[2].set_title("Bounded-BEV coverage")
    for axis in axes:
        axis.set_xticks(positions, names, rotation=32, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Ramen query-to-object-ID mapping audit (red = threshold warning)")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.spatial_query not in mapping["queries"]:
        raise ValueError(f"Unknown spatial-query {args.spatial_query!r}")
    spatial_ids = [int(value) for value in mapping["queries"][args.spatial_query]["object_ids"]]
    spatial_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(spatial_ids)))
    projection_pairs = [(0, 1), (0, 2), (1, 2)]
    spatial_fig, spatial_axes = plt.subplots(
        1, 3, figsize=(13.5, 4.2), constrained_layout=True
    )
    random = np.random.default_rng(0)
    for object_id, color in zip(spatial_ids, spatial_colors):
        points = batch.means[batch.semantic_ids == object_id]
        if len(points) > args.max_spatial_points_per_id:
            points = points[
                random.choice(len(points), args.max_spatial_points_per_id, replace=False)
            ]
        for axis, (first, second) in zip(spatial_axes, projection_pairs):
            axis.scatter(
                points[:, first],
                points[:, second],
                s=1.3,
                alpha=0.22,
                color=color,
                linewidths=0,
                label=f"ID {object_id}",
                rasterized=True,
            )
    for axis, (first, second) in zip(spatial_axes, projection_pairs):
        axis.set_xlabel(f"reconstruction axis {first}")
        axis.set_ylabel(f"reconstruction axis {second}")
        axis.set_title(f"axis {first}–{second} projection")
        axis.grid(alpha=0.15)
        axis.set_aspect("equal", adjustable="box")
    spatial_axes[-1].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, markerscale=5
    )
    spatial_fig.suptitle(
        f"{mapping['scene']} / {args.spatial_query}: spatial extent by mapped object ID"
    )
    spatial_fig.savefig(spatial_figure_path, dpi=180, bbox_inches="tight")
    plt.close(spatial_fig)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(figure_path)
    print(spatial_figure_path)


if __name__ == "__main__":
    main()
