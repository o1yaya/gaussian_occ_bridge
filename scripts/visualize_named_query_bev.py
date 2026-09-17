from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


QUERY_COLORS = {
    "chopsticks": "#4E79A7",
    "egg": "#F28E2B",
    "glass of water": "#59A14F",
    "pork belly": "#E15759",
    "wavy noodles in bowl": "#B07AA1",
    "yellow bowl": "#EDC948",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map query-grounded ILGS object IDs onto a full-scene semantic BEV."
    )
    parser.add_argument("--input", required=True, help="Full-scene semantic occupancy NPZ")
    parser.add_argument("--mapping", required=True, help="Query-to-object-ID JSON")
    parser.add_argument("--output", required=True, help="Output PNG")
    parser.add_argument("--summary", required=True, help="Output JSON summary")
    args = parser.parse_args()

    data = np.load(args.input)
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    occupancy = data["bev_occupancy"]
    evidence = data["bev_evidence"]
    labels = data["bev_semantic_ids"]
    unknown = data["bev_unknown_mask"]

    configured_queries = list(mapping["queries"].items())
    queries = [
        (query, info)
        for query, info in configured_queries
        if info.get("include_in_named_bev", True)
    ]
    excluded_queries = [
        {
            "query": query,
            "status": info["status"],
            "object_ids": info["object_ids"],
            "quality_note": info.get("quality_note", ""),
        }
        for query, info in configured_queries
        if not info.get("include_in_named_bev", True)
    ]
    named = np.full(labels.shape, -1, dtype=np.int16)
    owner_by_id: dict[int, str] = {}
    stats: dict[str, dict[str, object]] = {}
    overlaps: list[dict[str, object]] = []

    active_ids = set(int(value) for value in np.unique(labels[~unknown]))
    for query_index, (query, info) in enumerate(queries):
        object_ids = [int(value) for value in info["object_ids"]]
        for object_id in object_ids:
            if object_id in owner_by_id:
                overlaps.append(
                    {
                        "object_id": object_id,
                        "first_query": owner_by_id[object_id],
                        "second_query": query,
                    }
                )
            owner_by_id[object_id] = query

        query_mask = np.isin(labels, object_ids) & ~unknown
        named[query_mask] = query_index
        per_id = {
            str(object_id): int(np.count_nonzero((labels == object_id) & ~unknown))
            for object_id in object_ids
        }
        stats[query] = {
            "configured_object_ids": object_ids,
            "object_ids_visible_in_bounded_bev": [
                object_id for object_id in object_ids if object_id in active_ids
            ],
            "object_ids_absent_from_bounded_bev": [
                object_id for object_id in object_ids if object_id not in active_ids
            ],
            "bev_cells": int(np.count_nonzero(query_mask)),
            "bev_cells_at_occupancy_ge_0_5": int(
                np.count_nonzero(query_mask & (occupancy >= 0.5))
            ),
            "bev_cells_by_object_id": per_id,
            "selected_gaussians_in_query_export": int(info["selected_gaussians"]),
            "usable_view_count": int(info["usable_view_count"]),
            "status": info["status"],
            **({"quality_note": info["quality_note"]} if "quality_note" in info else {}),
        }

    mapped = named >= 0
    if overlaps:
        raise ValueError(f"Object IDs must be assigned to one query only: {overlaps}")
    known_cells = int(np.count_nonzero(~unknown))
    mapped_cells = int(np.count_nonzero(mapped))
    summary = {
        "scene": mapping["scene"],
        "mapping_method": mapping["mapping_method"],
        "configured_query_count": len(configured_queries),
        "accepted_query_count": len(queries),
        "excluded_queries": excluded_queries,
        "configured_object_id_count": len(owner_by_id),
        "overlapping_object_id_assignments": overlaps,
        "bounded_full_scene_bev_known_cells": known_cells,
        "mapped_bev_cells": mapped_cells,
        "mapped_fraction_of_known_bev": mapped_cells / known_cells if known_cells else 0.0,
        "coordinate_note": mapping["coordinate_note"],
        "unknown_space_note": "Cells without Gaussian evidence are unknown, not free.",
        "hard_label_note": "Each BEV cell carries the strongest height-wise voxel label from the memory-bounded hard-label baseline.",
        "queries": stats,
    }

    grid_min = data["grid_min"]
    voxel_size = data["voxel_size"]
    extent = [
        float(grid_min[0]),
        float(grid_min[0] + occupancy.shape[0] * voxel_size[0]),
        float(grid_min[1]),
        float(grid_min[1] + occupancy.shape[1] * voxel_size[1]),
    ]
    colors = [QUERY_COLORS.get(query, "#777777") for query, _ in queries]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(queries) + 0.5), len(queries))

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
    base = axes[0].imshow(
        occupancy.T,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title("Full-scene occupancy evidence")
    fig.colorbar(base, ax=axes[0], fraction=0.046, pad=0.04)

    named_masked = np.ma.masked_where(named.T < 0, named.T)
    axes[1].set_facecolor("#F2F2F2")
    axes[1].imshow(
        named_masked,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Accepted query-grounded object IDs")

    axes[2].imshow(
        np.log1p(evidence).T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
        aspect="equal",
    )
    axes[2].imshow(
        named_masked,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        alpha=0.88,
        interpolation="nearest",
        aspect="equal",
    )
    axes[2].set_title("Mapped queries over log evidence")

    handles = [
        Patch(facecolor=colors[index], label=query)
        for index, (query, _) in enumerate(queries)
    ]
    axes[2].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
    )
    for axis in axes:
        axis.set_xlabel("reconstruction axis 0")
        axis.set_ylabel("reconstruction axis 1")
    fig.suptitle(
        f"{mapping['scene']}: quality-gated query to object-ID BEV bridge",
        fontsize=13,
    )

    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
