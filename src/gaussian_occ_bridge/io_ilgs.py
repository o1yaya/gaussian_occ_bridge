from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData


@dataclass(frozen=True)
class ILGSGaussianBatch:
    """Activated ILGS Gaussian parameters ready for the reference splatter."""

    means: np.ndarray
    axis_aligned_scales: np.ndarray
    opacities: np.ndarray
    semantic_logits: np.ndarray
    rotations_wxyz: np.ndarray
    source_indices: np.ndarray
    semantic_source: str


@dataclass(frozen=True)
class ILGSHardLabelBatch:
    means: np.ndarray
    axis_aligned_scales: np.ndarray
    opacities: np.ndarray
    semantic_ids: np.ndarray
    rotations_wxyz: np.ndarray
    source_indices: np.ndarray
    semantic_source: str


def _ordered_fields(names: tuple[str, ...], prefix: str) -> list[str]:
    fields = [name for name in names if name.startswith(prefix)]
    return sorted(fields, key=lambda name: int(name.rsplit("_", 1)[-1]))


def _stack_fields(vertices: np.ndarray, fields: list[str]) -> np.ndarray:
    if not fields:
        raise ValueError("No requested PLY fields were found")
    return np.stack(
        [np.asarray(vertices[name], dtype=np.float64) for name in fields], axis=1
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert GraphDECO/ILGS scalar-first quaternions to rotation matrices."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape [N, 4] in wxyz order")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("ILGS PLY contains a zero-norm rotation quaternion")
    w, x, y, z = (quaternions / norms).T
    matrices = np.empty((quaternions.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _semantic_logits(
    vertices: np.ndarray,
    names: tuple[str, ...],
    mode: str,
    classifier_npz: str | Path | None,
) -> tuple[np.ndarray, str]:
    count = len(vertices)
    if mode == "constant":
        return np.zeros((count, 1), dtype=np.float64), "constant-target-class"

    if mode == "semantic-fields":
        fields = _ordered_fields(names, "semantic_")
        if not fields:
            raise ValueError("semantic-fields mode requires semantic_* PLY properties")
        return _stack_fields(vertices, fields), "raw-semantic-fields"

    if mode == "object-id":
        field = next(
            (candidate for candidate in ("predicted_object_id", "object_id") if candidate in names),
            None,
        )
        if field is None:
            raise ValueError("object-id mode requires object_id or predicted_object_id")
        ids = np.asarray(vertices[field], dtype=np.int64)
        if np.any(ids < 0):
            raise ValueError("object ids must be non-negative")
        logits = np.full((count, int(ids.max(initial=0)) + 1), -8.0, dtype=np.float64)
        logits[np.arange(count), ids] = 8.0
        return logits, field

    if mode == "object-classifier":
        if classifier_npz is None:
            raise ValueError("object-classifier mode requires classifier_npz")
        fields = _ordered_fields(names, "obj_dc_")
        if not fields:
            raise ValueError("object-classifier mode requires obj_dc_* PLY properties")
        objects = _stack_fields(vertices, fields)
        checkpoint = np.load(classifier_npz)
        weight = np.asarray(checkpoint["weight"], dtype=np.float64)
        bias = np.asarray(
            checkpoint["bias"] if "bias" in checkpoint else np.zeros(weight.shape[0]),
            dtype=np.float64,
        )
        if weight.ndim != 2 or weight.shape[1] != objects.shape[1]:
            raise ValueError(
                f"classifier weight must have shape [C, {objects.shape[1]}], got {weight.shape}"
            )
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"classifier bias must have shape {(weight.shape[0],)}")
        return objects @ weight.T + bias, "obj_dc_* + classifier_npz"

    raise ValueError(
        "semantic_mode must be constant, semantic-fields, object-id, or object-classifier"
    )


def load_ilgs_ply(
    path: str | Path,
    *,
    semantic_mode: str = "constant",
    classifier_npz: str | Path | None = None,
    transform: np.ndarray | None = None,
    min_opacity: float = 0.0,
    max_axis_scale: float | None = None,
    max_gaussians: int | None = None,
) -> ILGSGaussianBatch:
    """Load an ILGS/GraphDECO PLY and activate its stored parameters.

    ILGS stores log-scales, opacity logits, and unnormalized scalar-first
    quaternions. This adapter applies the same activations as GaussianModel.
    Rotated covariance is conservatively converted to XYZ marginal standard
    deviations for the axis-aligned NumPy reference splatter.

    `semantic-fields` exposes ILGS latent semantic fields as channels; these are
    not guaranteed to be categorical logits. Use `constant` for a query-isolated
    target, `object-id` for an exported labeled PLY, or `object-classifier` with
    an explicitly converted classifier NPZ for categorical semantics.
    """
    if not 0.0 <= min_opacity <= 1.0:
        raise ValueError("min_opacity must be in [0, 1]")
    if max_axis_scale is not None and max_axis_scale <= 0:
        raise ValueError("max_axis_scale must be positive")
    if max_gaussians is not None and max_gaussians <= 0:
        raise ValueError("max_gaussians must be positive")

    ply = PlyData.read(str(Path(path)))
    vertices = ply["vertex"].data
    if len(vertices) == 0:
        raise ValueError("ILGS PLY contains no vertices")
    names = vertices.dtype.names or ()
    required = {"x", "y", "z", "opacity"}
    missing = sorted(required.difference(names))
    scale_fields = _ordered_fields(names, "scale_")
    rotation_fields = _ordered_fields(names, "rot_")
    if missing or len(scale_fields) != 3 or len(rotation_fields) != 4:
        raise ValueError(
            "Invalid ILGS PLY schema: "
            f"missing={missing}, scale_fields={scale_fields}, rotation_fields={rotation_fields}"
        )

    means = _stack_fields(vertices, ["x", "y", "z"])
    native_scales = np.exp(_stack_fields(vertices, scale_fields))
    rotations = _stack_fields(vertices, rotation_fields)
    rotation_matrices = quaternion_wxyz_to_matrix(rotations)
    covariance = (
        rotation_matrices
        @ np.apply_along_axis(np.diag, 1, native_scales**2)
        @ np.swapaxes(rotation_matrices, 1, 2)
    )
    opacities = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))
    logits, semantic_source = _semantic_logits(
        vertices, names, semantic_mode, classifier_npz
    )

    if transform is None:
        linear = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
    else:
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("transform must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("transform must be affine with last row [0, 0, 0, 1]")
        linear = transform[:3, :3]
        translation = transform[:3, 3]
    means = means @ linear.T + translation
    covariance = linear[None, ...] @ covariance @ linear.T[None, ...]
    axis_scales = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0))

    valid = np.all(np.isfinite(means), axis=1)
    valid &= np.all(np.isfinite(axis_scales), axis=1)
    valid &= np.all(np.isfinite(logits), axis=1)
    valid &= np.isfinite(opacities)
    valid &= opacities >= min_opacity
    if max_axis_scale is not None:
        valid &= np.max(axis_scales, axis=1) <= max_axis_scale
    source_indices = np.flatnonzero(valid)
    if max_gaussians is not None and len(source_indices) > max_gaussians:
        ranking = np.argsort(opacities[source_indices], kind="stable")[::-1]
        source_indices = source_indices[ranking[:max_gaussians]]

    return ILGSGaussianBatch(
        means=means[source_indices].astype(np.float32),
        axis_aligned_scales=axis_scales[source_indices].astype(np.float32),
        opacities=opacities[source_indices].astype(np.float32),
        semantic_logits=logits[source_indices].astype(np.float32),
        rotations_wxyz=(rotations[source_indices] / np.linalg.norm(rotations[source_indices], axis=1, keepdims=True)).astype(np.float32),
        source_indices=source_indices.astype(np.int64),
        semantic_source=semantic_source,
    )


def load_ilgs_ply_hard_labels(
    path: str | Path,
    classifier_npz: str | Path,
    *,
    transform: np.ndarray | None = None,
    min_opacity: float = 0.0,
    max_axis_scale: float | None = None,
    max_gaussians: int | None = None,
    classifier_chunk_size: int = 32_768,
) -> ILGSHardLabelBatch:
    """Load ILGS geometry and predict one object ID per Gaussian in chunks."""
    if classifier_chunk_size <= 0:
        raise ValueError("classifier_chunk_size must be positive")
    geometry = load_ilgs_ply(
        path,
        semantic_mode="constant",
        transform=transform,
        min_opacity=min_opacity,
        max_axis_scale=max_axis_scale,
        max_gaussians=max_gaussians,
    )
    ply = PlyData.read(str(Path(path)))
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    object_fields = _ordered_fields(names, "obj_dc_")
    if not object_fields:
        raise ValueError("ILGS PLY does not contain obj_dc_* properties")
    source = geometry.source_indices
    objects = np.stack(
        [
            np.asarray(vertices[name], dtype=np.float32)[source]
            for name in object_fields
        ],
        axis=1,
    )
    checkpoint = np.load(classifier_npz)
    weight = np.asarray(checkpoint["weight"], dtype=np.float32)
    bias = np.asarray(
        checkpoint["bias"] if "bias" in checkpoint else np.zeros(weight.shape[0]),
        dtype=np.float32,
    )
    if weight.ndim != 2 or weight.shape[1] != objects.shape[1]:
        raise ValueError(
            f"classifier weight must have shape [C, {objects.shape[1]}], got {weight.shape}"
        )
    if bias.shape != (weight.shape[0],):
        raise ValueError(f"classifier bias must have shape {(weight.shape[0],)}")
    semantic_ids = np.empty(len(objects), dtype=np.int32)
    for start in range(0, len(objects), classifier_chunk_size):
        stop = min(start + classifier_chunk_size, len(objects))
        logits = objects[start:stop] @ weight.T + bias
        semantic_ids[start:stop] = np.argmax(logits, axis=1).astype(np.int32)

    return ILGSHardLabelBatch(
        means=geometry.means,
        axis_aligned_scales=geometry.axis_aligned_scales,
        opacities=geometry.opacities,
        semantic_ids=semantic_ids,
        rotations_wxyz=geometry.rotations_wxyz,
        source_indices=geometry.source_indices,
        semantic_source="obj_dc_* argmax via classifier_npz",
    )
