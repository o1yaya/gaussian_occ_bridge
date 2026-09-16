from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize hard-label semantic occupancy.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="ILGS object-ID occupancy")
    args = parser.parse_args()

    data = np.load(args.input)
    occupancy = data["bev_occupancy"]
    evidence = data["bev_evidence"]
    labels = data["bev_semantic_ids"].astype(np.float32)
    unknown = data["bev_unknown_mask"]
    labels[unknown] = np.nan
    grid_min = data["grid_min"]
    voxel_size = data["voxel_size"]
    extent = [
        float(grid_min[0]),
        float(grid_min[0] + occupancy.shape[0] * voxel_size[0]),
        float(grid_min[1]),
        float(grid_min[1] + occupancy.shape[1] * voxel_size[1]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    image = axes[0].imshow(
        occupancy.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title("Probabilistic-union occupancy")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    image = axes[1].imshow(
        labels.T,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Strongest-evidence object ID")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("classifier object ID")

    image = axes[2].imshow(
        np.log1p(evidence).T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[2].set_title("log(1 + accumulated evidence)")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_xlabel("reconstruction axis 0")
        axis.set_ylabel("reconstruction axis 1")
    fig.suptitle(args.title, fontsize=13)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

