from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import VoxelGridSpec


@dataclass(frozen=True)
class VoxelSplatResult:
    occupancy: np.ndarray
    semantic_probs: np.ndarray
    evidence: np.ndarray
    unknown_mask: np.ndarray


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def _validate_inputs(
    means: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    semantic_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray(means, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64)
    semantic_logits = np.asarray(semantic_logits, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape [N, 3]")
    if scales.shape != means.shape:
        raise ValueError("scales must have shape [N, 3]")
    if opacities.shape != (means.shape[0],):
        raise ValueError("opacities must have shape [N]")
    if semantic_logits.ndim != 2 or semantic_logits.shape[0] != means.shape[0]:
        raise ValueError("semantic_logits must have shape [N, C]")
    if semantic_logits.shape[1] < 1:
        raise ValueError("semantic_logits must contain at least one class")
    if np.any(scales <= 0):
        raise ValueError("all Gaussian scales must be positive")
    if not all(np.all(np.isfinite(x)) for x in (means, scales, opacities, semantic_logits)):
        raise ValueError("inputs must be finite")
    return means, scales, opacities, semantic_logits


def gaussian_to_voxels(
    means: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    semantic_logits: np.ndarray,
    grid: VoxelGridSpec,
    *,
    radius_sigma: float = 3.0,
    unknown_evidence_threshold: float = 1e-3,
) -> VoxelSplatResult:
    """Splat axis-aligned semantic Gaussians into an XYZ voxel grid.

    This is a correctness-oriented NumPy reference. `scales` are standard
    deviations along world XYZ. Rotation and full covariance are intentionally
    deferred to the PyTorch/CUDA implementation.
    """
    means, scales, opacities, semantic_logits = _validate_inputs(
        means, scales, opacities, semantic_logits
    )
    if radius_sigma <= 0:
        raise ValueError("radius_sigma must be positive")
    if unknown_evidence_threshold < 0:
        raise ValueError("unknown_evidence_threshold must be non-negative")

    num_classes = semantic_logits.shape[1]
    occupancy = np.zeros(grid.shape, dtype=np.float32)
    evidence = np.zeros(grid.shape, dtype=np.float32)
    semantic_mass = np.zeros(grid.shape + (num_classes,), dtype=np.float32)
    semantic_probs_per_gaussian = _softmax(semantic_logits)
    opacities = np.clip(opacities, 0.0, 1.0 - 1e-6)

    for mean, scale, opacity, class_probs in zip(
        means, scales, opacities, semantic_probs_per_gaussian, strict=True
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
        slices = tuple(slice(int(lower[a]), int(upper[a])) for a in range(3))
        current = occupancy[slices]
        occupancy[slices] = 1.0 - (1.0 - current) * (1.0 - local_evidence)
        evidence[slices] += local_evidence
        semantic_mass[slices] += local_evidence[..., None] * class_probs.astype(np.float32)

    semantic_probs = np.zeros_like(semantic_mass)
    np.divide(
        semantic_mass,
        evidence[..., None],
        out=semantic_probs,
        where=evidence[..., None] > 0,
    )
    unknown_mask = evidence < unknown_evidence_threshold
    return VoxelSplatResult(
        occupancy=occupancy,
        semantic_probs=semantic_probs,
        evidence=evidence,
        unknown_mask=unknown_mask,
    )


def reduce_to_bev(
    result: VoxelSplatResult,
    *,
    height_axis: int = 2,
    unknown_evidence_threshold: float = 1e-3,
) -> VoxelSplatResult:
    """Reduce 3D occupancy and semantics along height using probabilistic union."""
    if height_axis not in (0, 1, 2):
        raise ValueError("height_axis must be 0, 1 or 2")
    occupancy = 1.0 - np.prod(1.0 - result.occupancy, axis=height_axis)
    evidence = result.evidence.sum(axis=height_axis)
    semantic_mass = (result.semantic_probs * result.evidence[..., None]).sum(axis=height_axis)
    semantic_probs = np.zeros_like(semantic_mass)
    np.divide(
        semantic_mass,
        evidence[..., None],
        out=semantic_probs,
        where=evidence[..., None] > 0,
    )
    unknown_mask = evidence < unknown_evidence_threshold
    return VoxelSplatResult(
        occupancy=occupancy.astype(np.float32),
        semantic_probs=semantic_probs.astype(np.float32),
        evidence=evidence.astype(np.float32),
        unknown_mask=unknown_mask,
    )
