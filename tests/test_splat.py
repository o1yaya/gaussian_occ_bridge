import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_occ_bridge import VoxelGridSpec, gaussian_to_voxels, reduce_to_bev


class GaussianVoxelSplatTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridSpec(
            min_xyz=(-1.125, -1.125, -1.125),
            max_xyz=(1.125, 1.125, 1.125),
            voxel_size_xyz=(0.25, 0.25, 0.25),
        )

    def test_single_gaussian_peaks_at_mean(self):
        result = gaussian_to_voxels(
            means=np.array([[0.0, 0.0, 0.0]]),
            scales=np.array([[0.25, 0.25, 0.25]]),
            opacities=np.array([0.8]),
            semantic_logits=np.array([[5.0, 0.0]]),
            grid=self.grid,
        )
        peak = np.unravel_index(np.argmax(result.occupancy), result.occupancy.shape)
        self.assertEqual(peak, (4, 4, 4))
        self.assertAlmostEqual(float(result.occupancy[peak]), 0.8, places=6)
        self.assertTrue(np.all((result.occupancy >= 0) & (result.occupancy <= 1)))

    def test_semantic_class_is_preserved(self):
        result = gaussian_to_voxels(
            means=np.array([[0.0, 0.0, 0.0]]),
            scales=np.array([[0.2, 0.2, 0.2]]),
            opacities=np.array([0.9]),
            semantic_logits=np.array([[-2.0, 4.0, 0.0]]),
            grid=self.grid,
        )
        self.assertEqual(int(np.argmax(result.semantic_probs[4, 4, 4])), 1)

    def test_probabilistic_union_for_overlap(self):
        result = gaussian_to_voxels(
            means=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            scales=np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
            opacities=np.array([0.5, 0.5]),
            semantic_logits=np.array([[1.0, 0.0], [1.0, 0.0]]),
            grid=self.grid,
        )
        self.assertAlmostEqual(float(result.occupancy[4, 4, 4]), 0.75, places=6)

    def test_bev_reduction_and_unknown_mask(self):
        result = gaussian_to_voxels(
            means=np.array([[0.0, 0.0, 0.0]]),
            scales=np.array([[0.2, 0.2, 0.2]]),
            opacities=np.array([0.9]),
            semantic_logits=np.array([[2.0, 0.0]]),
            grid=self.grid,
        )
        bev = reduce_to_bev(result)
        self.assertEqual(bev.occupancy.shape, self.grid.shape[:2])
        self.assertEqual(bev.semantic_probs.shape, self.grid.shape[:2] + (2,))
        self.assertFalse(bool(bev.unknown_mask[4, 4]))
        self.assertTrue(bool(bev.unknown_mask[0, 0]))


if __name__ == "__main__":
    unittest.main()
