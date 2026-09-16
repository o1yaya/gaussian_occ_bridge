from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Highlight mapped query IDs in semantic BEV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.input)
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    object_ids = np.asarray(mapping["object_ids"], dtype=np.int32)
    occupancy = data["bev_occupancy"]
    evidence = data["bev_evidence"]
    labels = data["bev_semantic_ids"]
    unknown = data["bev_unknown_mask"]
    query_mask = np.isin(labels, object_ids) & ~unknown
    grid_min = data["grid_min"]
    voxel_size = data["voxel_size"]
    extent = [
        float(grid_min[0]),
        float(grid_min[0] + occupancy.shape[0] * voxel_size[0]),
        float(grid_min[1]),
        float(grid_min[1] + occupancy.shape[1] * voxel_size[1]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.1, 3.8), constrained_layout=True)
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
    axes[0].set_title("Full-scene occupancy")
    fig.colorbar(base, ax=axes[0], fraction=0.046, pad=0.04)

    mask_image = axes[1].imshow(
        query_mask.T,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["white", "#d62728"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title(f"Mapped query mask: {mapping['query']}\nIDs {mapping['object_ids']}")
    fig.colorbar(mask_image, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1])

    axes[2].imshow(
        np.log1p(evidence).T,
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="nearest",
        aspect="equal",
    )
    overlay = np.ma.masked_where(~query_mask.T, query_mask.T)
    axes[2].imshow(
        overlay,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#d62728"]),
        alpha=0.85,
        interpolation="nearest",
        aspect="equal",
    )
    axes[2].set_title("Query IDs over full-scene evidence")

    for axis in axes:
        axis.set_xlabel("reconstruction axis 0")
        axis.set_ylabel("reconstruction axis 1")
    fig.suptitle(
        f"{mapping['scene']} / {mapping['query']}: object-ID to query bridge",
        fontsize=13,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

