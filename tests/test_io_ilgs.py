import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_occ_bridge import load_ilgs_ply, quaternion_wxyz_to_matrix


def _write_ilgs_ply(path: Path) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("obj_dc_0", "f4"),
        ("obj_dc_1", "f4"),
        ("semantic_0", "f4"),
        ("semantic_1", "f4"),
    ]
    rows = np.zeros(2, dtype=dtype)
    rows["x"] = [1.0, 2.0]
    rows["y"] = [2.0, 3.0]
    rows["z"] = [3.0, 4.0]
    rows["opacity"] = [0.0, -4.0]
    rows["scale_0"] = np.log([0.1, 0.2])
    rows["scale_1"] = np.log([0.2, 0.3])
    rows["scale_2"] = np.log([0.3, 0.4])
    angle = np.pi / 2
    rows["rot_0"] = [np.cos(angle / 2), 1.0]
    rows["rot_3"] = [np.sin(angle / 2), 0.0]
    rows["obj_dc_0"] = [2.0, -1.0]
    rows["obj_dc_1"] = [-1.0, 2.0]
    rows["semantic_0"] = [0.25, 0.75]
    rows["semantic_1"] = [0.75, 0.25]
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(str(path))


class ILGSPLYAdapterTests(unittest.TestCase):
    def test_quaternion_uses_scalar_first_order(self):
        angle = np.pi / 2
        rotation = quaternion_wxyz_to_matrix(
            np.array([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]])
        )[0]
        rotated_x = rotation @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated_x, [0.0, 1.0, 0.0], atol=1e-7)

    def test_activations_rotation_and_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_cloud.ply"
            _write_ilgs_ply(path)
            batch = load_ilgs_ply(path, min_opacity=0.1)

        self.assertEqual(batch.means.shape, (1, 3))
        np.testing.assert_allclose(batch.means[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(batch.axis_aligned_scales[0], [0.2, 0.1, 0.3], atol=1e-6)
        self.assertAlmostEqual(float(batch.opacities[0]), 0.5, places=6)
        self.assertEqual(batch.semantic_logits.shape, (1, 1))
        np.testing.assert_array_equal(batch.source_indices, [0])

    def test_object_classifier_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            ply_path = directory / "point_cloud.ply"
            classifier_path = directory / "classifier.npz"
            _write_ilgs_ply(ply_path)
            np.savez(
                classifier_path,
                weight=np.array([[1.0, 0.0], [0.0, 1.0]]),
                bias=np.array([0.1, -0.1]),
            )
            batch = load_ilgs_ply(
                ply_path,
                semantic_mode="object-classifier",
                classifier_npz=classifier_path,
            )

        np.testing.assert_allclose(
            batch.semantic_logits,
            [[2.1, -1.1], [-0.9, 1.9]],
            atol=1e-6,
        )
        self.assertEqual(batch.semantic_source, "obj_dc_* + classifier_npz")

    def test_transform_updates_means_and_spatial_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_cloud.ply"
            _write_ilgs_ply(path)
            transform = np.array(
                [
                    [2.0, 0.0, 0.0, 10.0],
                    [0.0, 2.0, 0.0, -1.0],
                    [0.0, 0.0, 2.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            batch = load_ilgs_ply(path, transform=transform, min_opacity=0.1)

        np.testing.assert_allclose(batch.means[0], [12.0, 3.0, 6.5], atol=1e-6)
        np.testing.assert_allclose(
            batch.axis_aligned_scales[0], [0.4, 0.2, 0.6], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()

