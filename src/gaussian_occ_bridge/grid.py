from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _vec3(value: float | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 3)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a scalar or length-3 vector, got {array.shape}")
    return array


@dataclass(frozen=True)
class VoxelGridSpec:
    """Axis-aligned voxel grid stored in X, Y, Z array order."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    voxel_size_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        mins = _vec3(self.min_xyz, "min_xyz")
        maxs = _vec3(self.max_xyz, "max_xyz")
        sizes = _vec3(self.voxel_size_xyz, "voxel_size_xyz")
        if np.any(maxs <= mins):
            raise ValueError("Every max_xyz value must be greater than min_xyz")
        if np.any(sizes <= 0):
            raise ValueError("voxel_size_xyz values must be positive")

    @property
    def mins(self) -> np.ndarray:
        return _vec3(self.min_xyz, "min_xyz")

    @property
    def maxs(self) -> np.ndarray:
        return _vec3(self.max_xyz, "max_xyz")

    @property
    def voxel_sizes(self) -> np.ndarray:
        return _vec3(self.voxel_size_xyz, "voxel_size_xyz")

    @property
    def shape(self) -> tuple[int, int, int]:
        extent = (self.maxs - self.mins) / self.voxel_sizes
        return tuple(np.ceil(extent - 1e-12).astype(np.int64).tolist())

    def centers(self, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return voxel-center coordinates for a half-open integer box."""
        lower = np.asarray(lower, dtype=np.int64)
        upper = np.asarray(upper, dtype=np.int64)
        axes = []
        for axis in range(3):
            indices = np.arange(lower[axis], upper[axis], dtype=np.float64)
            axes.append(self.mins[axis] + (indices + 0.5) * self.voxel_sizes[axis])
        return tuple(axes)

    def clipped_index_box(self, lower_xyz: np.ndarray, upper_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.floor((np.asarray(lower_xyz) - self.mins) / self.voxel_sizes).astype(np.int64)
        upper = np.ceil((np.asarray(upper_xyz) - self.mins) / self.voxel_sizes).astype(np.int64)
        grid_shape = np.asarray(self.shape, dtype=np.int64)
        lower = np.clip(lower, 0, grid_shape)
        upper = np.clip(upper, 0, grid_shape)
        return lower, upper
