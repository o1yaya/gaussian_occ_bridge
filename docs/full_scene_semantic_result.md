# Full-scene ILGS object-ID semantic occupancy

## Goal

This run verifies the complete path from an ILGS full-scene PLY and trained object
classifier to a memory-bounded multi-object voxel/BEV representation.

It is an object-ID integration baseline. The IDs are ILGS classifier indices rather than
human-readable open-vocabulary categories, and there is no occupancy ground truth for this run.

## Input provenance

| Artifact | Size | SHA-256 |
|---|---:|---|
| `ramen_point_cloud.ply` | 276,862,267 bytes | `8fae40798cc9f27764e6e2bf29734c569da015726ecb8b30770a5c906acd6ec8` |
| `ramen_classifier.npz` | converted from 18,547-byte `classifier.pth` | `16e448eddb928db7e818fae95ca43974def2d7a974cf4c1d041c074d36296c9b` |

The full PLY contains 854,507 Gaussians, 16 `obj_dc_*` features per Gaussian and a
256-way `1×1` classifier. The PyTorch state dict was converted with the repository's
restricted NumPy loader; no arbitrary checkpoint code was executed.

## Memory design

A dense semantic tensor with shape `[X,Y,Z,256]` is impractical for a 15.2-million-voxel
grid. This reference uses:

- probabilistic union for occupancy;
- scalar accumulated evidence per voxel;
- one `int32` object ID and one label score per voxel;
- the object ID of the individual Gaussian with the strongest local opacity evidence.

This reduces semantic storage from `O(XYZC)` to `O(XYZ)`. The trade-off is that evidence
from several Gaussians of the same class is not accumulated before label selection.

## Frozen command

```bash
python scripts/convert_ilgs_classifier.py \
  --input /path/to/classifier.pth \
  --output /path/to/ramen_classifier.npz

python scripts/run_ilgs_semantic_ply.py \
  --ply /path/to/ramen_point_cloud.ply \
  --classifier-npz /path/to/ramen_classifier.npz \
  --voxel-size 0.10 0.10 0.10 \
  --min-opacity 0.10 \
  --max-axis-scale 0.20 \
  --max-gaussians 200000 \
  --auto-grid-crop-quantile 0.01 \
  --output-dir outputs/ramen_full_semantic
```

The Gaussian budget is a deterministic top-opacity selection after activation and filtering.
The grid uses the 1st–99th center quantiles plus scale-aware padding.

## Result

| Item | Value |
|---|---:|
| Source Gaussians | 854,507 |
| Filtered/top-opacity budget | 200,000 |
| Gaussian centers inside grid | 192,157 |
| Active object IDs in selected Gaussians | 47 |
| Active object IDs in BEV | 47 |
| Grid bounds | `[-7.2,-4.9,-12.3]` to `[15.9,25.4,9.4]` |
| Voxel size | `0.1³` reconstruction units |
| Grid shape | `231 × 303 × 217` |
| Total voxels | 15,188,481 |
| Occupied voxels at 0.5 | 84,665 |
| Voxels with evidence | 628,404 |
| BEV cells with evidence | 36,196 |
| Load and classify time | 2.28 s |
| CPU splat and BEV reduction | 10.86 s |

Timing is a single local reference run, not a cross-platform benchmark.

![Full-scene object-ID occupancy](assets/ramen_full_semantic_bev.png)

## Interpretation boundaries

- Axis 2 is a reconstruction axis, not a verified gravity/up axis. The middle panel is
  therefore an axis-reduced semantic map, not yet a robot-frame BEV.
- Object IDs are instance-classifier indices. Mapping them to human-readable query labels
  requires the ILGS query-voting or language-feature stage.
- Hard-label selection is memory efficient but not differentiable and does not aggregate
  evidence per class.
- No camera-ray model is present. Empty cells are unknown rather than free.
- No occupancy ground truth is available, so the run supports integration and efficiency
  claims only, not semantic OCC IoU/mIoU claims.

