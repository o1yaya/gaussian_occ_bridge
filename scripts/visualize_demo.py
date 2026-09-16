from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    input_path = ROOT / "outputs" / "synthetic_demo.npz"
    output_path = ROOT / "docs" / "assets" / "synthetic_bev_demo.png"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run `python scripts/run_demo.py` first."
        )

    data = np.load(input_path)
    occupancy = data["bev_occupancy"]
    semantic_probs = data["bev_semantic_probs"]
    unknown = data["bev_unknown_mask"]
    semantic_class = np.argmax(semantic_probs, axis=-1).astype(np.float32)
    semantic_class[unknown] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    occupancy_plot = axes[0].imshow(
        occupancy.T,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[0].set_title("BEV occupancy")
    fig.colorbar(occupancy_plot, ax=axes[0], fraction=0.046, pad=0.04)

    semantic_plot = axes[1].imshow(
        semantic_class.T,
        origin="lower",
        cmap="tab10",
        vmin=0,
        vmax=max(semantic_probs.shape[-1] - 1, 1),
        interpolation="nearest",
    )
    axes[1].set_title("Semantic class (known cells)")
    fig.colorbar(
        semantic_plot,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        ticks=np.arange(semantic_probs.shape[-1]),
    )

    known_plot = axes[2].imshow(
        (~unknown).T,
        origin="lower",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title("Observed evidence mask")
    fig.colorbar(known_plot, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1])

    for axis in axes:
        axis.set_xlabel("voxel x")
        axis.set_ylabel("voxel y")
        axis.grid(False)

    fig.suptitle("Synthetic semantic Gaussian → voxel occupancy → BEV", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()

