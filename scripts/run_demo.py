from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_occ_bridge import VoxelGridSpec, gaussian_to_voxels, reduce_to_bev


def main() -> None:
    grid = VoxelGridSpec(
        min_xyz=(-1.0, -1.0, -0.5),
        max_xyz=(1.0, 1.0, 1.0),
        voxel_size_xyz=(0.05, 0.05, 0.05),
    )
    means = np.array([[-0.25, 0.0, 0.15], [0.30, 0.10, 0.35]], dtype=np.float64)
    scales = np.array([[0.16, 0.24, 0.20], [0.20, 0.14, 0.28]], dtype=np.float64)
    opacities = np.array([0.92, 0.85], dtype=np.float64)
    semantic_logits = np.array([[5.0, 0.0, -1.0], [-1.0, 4.0, 0.0]], dtype=np.float64)

    voxels = gaussian_to_voxels(means, scales, opacities, semantic_logits, grid)
    bev = reduce_to_bev(voxels)
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_demo.npz"
    summary_path = output_dir / "synthetic_demo_summary.json"
    np.savez_compressed(
        output_path,
        occupancy=voxels.occupancy,
        semantic_probs=voxels.semantic_probs,
        evidence=voxels.evidence,
        unknown_mask=voxels.unknown_mask,
        bev_occupancy=bev.occupancy,
        bev_semantic_probs=bev.semantic_probs,
        bev_unknown_mask=bev.unknown_mask,
    )
    summary = {
        "grid_shape_xyz": grid.shape,
        "gaussian_count": int(means.shape[0]),
        "occupied_voxels_at_0_5": int((voxels.occupancy >= 0.5).sum()),
        "known_voxels": int((~voxels.unknown_mask).sum()),
        "bev_known_cells": int((~bev.unknown_mask).sum()),
        "max_occupancy": float(voxels.occupancy.max()),
        "output": output_path.relative_to(ROOT).as_posix(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
