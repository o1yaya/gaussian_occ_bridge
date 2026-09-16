# ILGS PLY adapter

`scripts/run_ilgs_ply.py` converts an ILGS `point_cloud.ply` or a query-isolated
`target_gaussians_full.ply` into the occupancy/BEV representation used by this repository.

## Why an adapter is necessary

ILGS follows the GraphDECO parameterization and stores optimization variables rather than
render-time values:

| PLY field | Stored value | Adapter activation |
|---|---|---|
| `scale_0..2` | log standard deviation | `exp(scale)` |
| `opacity` | opacity logit | `sigmoid(opacity)` |
| `rot_0..3` | unnormalized quaternion in `wxyz` order | L2 normalization |
| `semantic_*` | ILGS latent semantic feature | no automatic categorical meaning |
| `obj_dc_*` | object feature | requires the trained classifier |

The CPU splatter is axis-aligned. The adapter therefore builds the rotated covariance
`R diag(scale^2) R^T` and uses its XYZ marginal standard deviations. This preserves the
rotation-dependent spatial extent, but is still an approximation of a full-covariance splat.

## Query-isolated target

Use this for `target_gaussians_full.ply` produced by the ILGS query export. All selected
Gaussians are treated as one target category:

```bash
python scripts/run_ilgs_ply.py \
  --ply /path/to/target_gaussians_full.ply \
  --semantic-mode constant \
  --voxel-size 0.05 0.05 0.05 \
  --min-opacity 0.05 \
  --output-dir outputs/pork_belly
```

## Object classifier semantics

To reproduce ILGS object classes without importing CUDA/PyTorch, first convert the final
classifier to an NPZ containing a two-dimensional `weight` array and optional `bias` array.
Run this once inside the ILGS training environment:

```bash
python scripts/convert_ilgs_classifier.py \
  --input /path/to/classifier.pth \
  --output /path/to/classifier.npz
```

Then run the CPU adapter:

```bash
python scripts/run_ilgs_ply.py \
  --ply /path/to/point_cloud.ply \
  --semantic-mode object-classifier \
  --classifier-npz /path/to/classifier.npz \
  --voxel-size 0.10 0.10 0.10 \
  --max-axis-scale 0.50 \
  --max-gaussians 200000
```

`semantic-fields` mode is available for diagnostics, but ILGS `semantic_*` values are latent
features, not guaranteed class logits. Do not report their channel argmax as semantic mIoU
without the appropriate decoder/projection and label mapping.

## Coordinate transform

Without a transform the result remains in reconstruction coordinates. A robot-facing output
needs a calibrated metric transform. Supply a JSON file containing a 4x4 affine matrix:

```bash
python scripts/run_ilgs_ply.py \
  --ply /path/to/target_gaussians_full.ply \
  --transform-json configs/reconstruction_to_robot.json \
  --grid-min -2 -2 -0.5 \
  --grid-max 2 2 2 \
  --voxel-size 0.05 0.05 0.05
```

The adapter transforms both means and covariances. A guessed scale or axis swap should never
be presented as calibrated robot coordinates.

## Safety guards

- Auto-grid creation includes the configured Gaussian support and refuses grids above
  `--max-voxels` (64 million by default).
- `--max-axis-scale` removes extreme splats that would dominate memory and occupancy.
- `--max-gaussians` keeps the highest-opacity Gaussians deterministically.
- Unobserved cells remain unknown; free space still requires camera/depth-ray evidence.
