# Experiment Plan

## Objective

Build a measurable bridge from semantic 3D Gaussians to 3D occupancy and BEV. The first milestone tests representation correctness; it does not claim end-to-end dynamic reconstruction.

## Research questions

1. Can semantic Gaussians produce a stable voxel occupancy map without dense point conversion?
2. How do voxel size, Gaussian support radius and opacity aggregation affect occupied IoU, memory and latency?
3. How much information is lost when reducing 3D occupancy to BEV?
4. What additional evidence is required to distinguish free from unknown space?

## Phase 0 Synthetic reference

Input: one to four axis-aligned Gaussians with controlled class logits.

Checks:

- Occupancy peaks near each Gaussian mean.
- Occupancy remains in `[0, 1]`.
- Semantic argmax agrees with the source Gaussian in single-object regions.
- Overlapping Gaussians use probabilistic union rather than an unconstrained sum.
- BEV shape and height reduction are correct.
- Cells without Gaussian or ray evidence remain unknown.

Deliverable: NumPy CPU reference plus deterministic unit tests.

## Phase 1 ILGS static-scene export

Input:

- Gaussian means, scales, rotations and opacities from `point_cloud.ply`.
- Semantic or identity logits from the trained classifier.
- Scene normalization metadata and camera poses.

Comparisons:

1. Point-center voxelization.
2. Axis-aligned Gaussian splatting.
3. Rotation-aware full-covariance splatting.

Metrics:

- Occupied IoU against depth-fused or mesh-derived voxel targets.
- Semantic mIoU where ground truth is available.
- Latency, peak memory and active voxel count.

## Phase 2 Public multi-camera baseline

Candidate baseline: GaussianFormer on a small public subset, followed by an indoor adaptation if compute permits.

Required controls:

- Fixed train/validation split and grid bounds.
- Same image encoder and voxel resolution for fair comparison.
- Results reported separately for geometry, semantics and efficiency.
- No mixing of static ILGS query-mask metrics with occupancy metrics.

## Phase 3 Robot interface

1. Convert BEV occupancy to a 2D grid with explicit resolution and origin.
2. Transform from reconstruction/camera frame to robot map frame using calibrated SE(3).
3. Publish `nav_msgs/OccupancyGrid` in simulation.
4. Validate map orientation, origin, unknown cells and obstacle inflation in Nav2.

## Frozen first ablations

| Variable | Values |
|---|---|
| Voxel size | 0.05 m, 0.10 m, 0.20 m |
| Gaussian support | 2 sigma, 3 sigma, 4 sigma |
| Aggregation | sum and clip, max, probabilistic union |
| Geometry | center voxel, diagonal covariance, full covariance |
| Semantics | nearest Gaussian, evidence-weighted probabilities |

## Reporting rules

- Always state the coordinate frame and whether metric scale is known.
- Report free, occupied and unknown definitions before reporting IoU.
- Keep static reconstruction, occupancy prediction and navigation results in separate tables.
- Record failure cases caused by missing views, scale errors and semantic boundary bleeding.
