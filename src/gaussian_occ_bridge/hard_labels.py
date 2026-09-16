from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import VoxelGridSpec


@dataclass(frozen=True)
class HardLabelSplatResult:
    occupancy: np.ndarray
    evidence: np.ndarray
    semantic_ids: np.ndarray
    label_score: np.ndarray
    unknown_mask: np.ndarray


def gaussian_to_voxels_hard_labels(
    means: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    semantic_ids: np.ndarray,
    grid: VoxelGridSpec,
    *,
    radius_sigma: float = 3.0,
    unknown_evidence_threshold: float = 1e-3,
) -> HardLabelSplatResult:
    """Memory-efficient splat for many discrete object IDs.

    Occupancy uses probabilistic union. Each voxel label is assigned to the
    individual Gaussian with the strongest local opacity evidence. This avoids
    allocating [X, Y, Z, C], but does not accumulate evidence per class.
    """
    means = np.asarray(means, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64)
    semantic_ids = np.asarray(semantic_ids, dtype=np.int32)
    count = means.shape[0]
    if means.shape != (count, 3) or scales.shape != (count, 3):
        raise ValueError("means and scales must have shape [N, 3]")
    if opacities.shape != (count,) or semantic_ids.shape != (count,):
        raise ValueError("opacities and semantic_ids must have shape [N]")
    if np.any(scales <= 0) or np.any(semantic_ids < 0):
        raise ValueError("scales must be positive and semantic_ids non-negative")
    if radius_sigma <= 0 or unknown_evidence_threshold < 0:
        raise ValueError("invalid radius_sigma or unknown_evidence_threshold")
    if not all(np.all(np.isfinite(value)) for value in (means, scales, opacities)):
        raise ValueError("inputs must be finite")

    occupancy = np.zeros(grid.shape, dtype=np.float32)
    evidence = np.zeros(grid.shape, dtype=np.float32)
    label_score = np.zeros(grid.shape, dtype=np.float32)
    labels = np.full(grid.shape, -1, dtype=np.int32)
    opacities = np.clip(opacities, 0.0, 1.0 - 1e-6)

    for mean, scale, opacity, semantic_id in zip(
        means, scales, opacities, semantic_ids, strict=True
    ):
        if opacity <= 0:
            continue
        lower, upper = grid.clipped_index_box(
            mean - radius_sigma * scale,
            mean + radius_sigma * scale,
        )
        if np.any(upper <= lower):
            continue
        x, y, z = grid.centers(lower, upper)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        squared_distance = (
            ((xx - mean[0]) / scale[0]) ** 2
            + ((yy - mean[1]) / scale[1]) ** 2
            + ((zz - mean[2]) / scale[2]) ** 2
        )
        local_evidence = (opacity * np.exp(-0.5 * squared_distance)).astype(np.float32)
        slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
        current = occupancy[slices]
        occupancy[slices] = 1.0 - (1.0 - current) * (1.0 - local_evidence)
        evidence[slices] += local_evidence
        local_score = label_score[slices]
        stronger = local_evidence > local_score
        local_score[stronger] = local_evidence[stronger]
        labels[slices][stronger] = int(semantic_id)

    unknown = evidence < unknown_evidence_threshold
    labels[unknown] = -1
    return HardLabelSplatResult(occupancy, evidence, labels, label_score, unknown)


def reduce_hard_labels_to_bev(
    result: HardLabelSplatResult,
    *,
    height_axis: int = 2,
    unknown_evidence_threshold: float = 1e-3,
) -> HardLabelSplatResult:
    if height_axis not in (0, 1, 2):
        raise ValueError("height_axis must be 0, 1 or 2")
    occupancy = 1.0 - np.prod(1.0 - result.occupancy, axis=height_axis)
    evidence = result.evidence.sum(axis=height_axis)
    best_height = np.argmax(result.label_score, axis=height_axis)
    expanded = np.expand_dims(best_height, axis=height_axis)
    labels = np.take_along_axis(result.semantic_ids, expanded, axis=height_axis).squeeze(
        axis=height_axis
    )
    score = np.take_along_axis(result.label_score, expanded, axis=height_axis).squeeze(
        axis=height_axis
    )
    unknown = evidence < unknown_evidence_threshold
    labels = labels.astype(np.int32)
    labels[unknown] = -1
    return HardLabelSplatResult(
        occupancy.astype(np.float32),
        evidence.astype(np.float32),
        labels,
        score.astype(np.float32),
        unknown,
    )

