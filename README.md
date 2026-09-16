# Gaussian OCC Bridge

An in-progress bridge from semantic 3D Gaussians to voxel occupancy and bird's-eye-view representations for robot perception.

## Current status

Completed on 2026-09-16:

- Defined the input/output contract and evaluation protocol.
- Fixed the first indoor workspace grid in `configs/baseline.yaml`.
- Implemented a NumPy CPU reference for axis-aligned Gaussian-to-voxel splatting.
- Implemented 3D occupancy to BEV reduction.
- Added deterministic unit tests and a synthetic demo.

Not completed yet:

- Rotation-aware full covariance splatting.
- Camera-ray free-space supervision.
- PyTorch/CUDA implementation and gradient checks.
- Public dataset training or nuScenes/indoor RGB-D metrics.
- ROS2 `nav_msgs/OccupancyGrid` publisher.

The project therefore supports the resume statement “Gaussian-to-voxel reference baseline in progress,” but it does not support claims of a finished BEV/OCC or robot navigation system.

## Representation contract

Input Gaussians:

```text
mean              [N, 3] world or reconstruction coordinates
scale             [N, 3] standard deviations for the CPU reference
opacity            [N]
semantic_logits    [N, C]
```

Reference output:

```text
occupancy          [X, Y, Z] probabilistic union of Gaussian evidence
semantic_probs     [X, Y, Z, C]
evidence           [X, Y, Z]
unknown_mask       [X, Y, Z]
BEV occupancy      [X, Y]
BEV semantics      [X, Y, C]
```

Important boundary: Gaussians alone provide occupied-space evidence. They do not prove free space. The CPU baseline marks low-evidence cells as unknown; future free-space labels must come from calibrated camera or depth rays.

## Pipeline

```text
ILGS semantic Gaussians
  -> coordinate and scale normalization
  -> local Gaussian-to-voxel evidence accumulation
  -> occupied / unknown representation
  -> semantic probability aggregation
  -> height reduction to BEV
  -> occupancy and semantic metrics
  -> future ROS2 OccupancyGrid adapter
```

## Quick start

```bash
python -m unittest discover -s tests -v
python scripts/run_demo.py
```

If the package is not installed, run with `PYTHONPATH=src` on Linux/macOS or set `$env:PYTHONPATH="src"` in PowerShell.

## Planned baselines

1. CPU axis-aligned reference in this repository.
2. Rotation-aware PyTorch implementation with autograd.
3. GaussianFormer-style image-to-Gaussian and Gaussian-to-voxel comparison.
4. Camera-ray free/occupied/unknown supervision.
5. ROS2 occupancy publishing and Nav2 simulation.

See `docs/experiment_plan.md` for the frozen first experiment protocol.
