import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_occ_bridge import (
    VoxelGridSpec,
    gaussian_to_voxels_hard_labels,
    reduce_hard_labels_to_bev,
)


class HardLabelSplatTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridSpec(
            min_xyz=(-0.5, -0.5, -0.5),
            max_xyz=(0.5, 0.5, 0.5),
            voxel_size_xyz=(0.25, 0.25, 0.25),
        )

    def test_stronger_gaussian_wins_voxel_label(self):
        result = gaussian_to_voxels_hard_labels(
            means=np.array([[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]]),
            scales=np.full((2, 3), 0.2),
            opacities=np.array([0.4, 0.9]),
            semantic_ids=np.array([3, 7]),
            grid=self.grid,
        )
        self.assertEqual(int(result.semantic_ids[2, 2, 2]), 7)
        self.assertAlmostEqual(float(result.occupancy[2, 2, 2]), 0.94, places=6)

    def test_bev_uses_strongest_height_label(self):
        result = gaussian_to_voxels_hard_labels(
            means=np.array([[0.125, 0.125, -0.125], [0.125, 0.125, 0.125]]),
            scales=np.full((2, 3), 0.08),
            opacities=np.array([0.3, 0.8]),
            semantic_ids=np.array([2, 9]),
            grid=self.grid,
        )
        bev = reduce_hard_labels_to_bev(result)
        self.assertEqual(int(bev.semantic_ids[2, 2]), 9)
        self.assertFalse(bool(bev.unknown_mask[2, 2]))
        self.assertEqual(int(bev.semantic_ids[0, 0]), -1)


if __name__ == "__main__":
    unittest.main()

