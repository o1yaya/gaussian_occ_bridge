from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an occupancy NPZ produced by this repo.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Gaussian occupancy in reconstruction coordinates")
    args = parser.parse_args()

    data = np.load(args.input)
    occupancy = data["bev_occupancy"]
    max_height_occupancy = data["occupancy"].max(axis=2)
    unknown = data["bev_unknown_mask"]
    evidence_3d = data["evidence"]
    evidence = evidence_3d.sum(axis=2)
    grid_min = data["grid_min"] if "grid_min" in data else np.zeros(3)
    voxel_size = data["voxel_size"] if "voxel_size" in data else np.ones(3)
    extent = [
        float(grid_min[0]),
        float(grid_min[0] + occupancy.shape[0] * voxel_size[0]),
        float(grid_min[1]),
        float(grid_min[1] + occupancy.shape[1] * voxel_size[1]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5), constrained_layout=True)
    image = axes[0, 0].imshow(
        max_height_occupancy.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("Maximum voxel occupancy along axis 2")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    image = axes[0, 1].imshow(
        occupancy.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 1].set_title("Probabilistic-union BEV occupancy")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    image = axes[1, 0].imshow(
        np.log1p(evidence).T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 0].set_title("log(1 + accumulated evidence)")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    image = axes[1, 1].imshow(
        (~unknown).T,
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 1].set_title("Observed evidence mask")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04, ticks=[0, 1])

    for axis in axes.flat:
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
