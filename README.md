# Gaussian OCC Bridge

[![tests](https://github.com/o1yaya/gaussian_occ_bridge/actions/workflows/tests.yml/badge.svg)](https://github.com/o1yaya/gaussian_occ_bridge/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Status](https://img.shields.io/badge/status-reference%20baseline-orange)

一个面向机器人感知的最小可运行基线：将带语义的 3D Gaussians 转换为体素占据表示，并进一步聚合为鸟瞰图（BEV）。项目当前重点是把 3DGS 重建结果连接到 BEV/OCC 下游，而不是声称已经完成自由空间预测或导航系统。

> A small, testable reference bridge from semantic 3D Gaussians to voxel occupancy and BEV representations for robot perception.

## 30-second overview

```mermaid
flowchart LR
    A[Semantic 3D Gaussians<br/>mean / scale / opacity / logits]
    B[Local Gaussian splatting]
    C[3D occupancy evidence<br/>occupied / unknown]
    D[Semantic aggregation]
    E[Height reduction]
    F[BEV occupancy + semantics]
    G[Future: ray free-space<br/>ROS2 / Nav2]
    A --> B --> C --> D --> E --> F -.-> G
```

The current NumPy implementation uses diagonal covariance, local `3σ` support, probabilistic-union occupancy, and evidence-weighted semantic fusion. It is intentionally correctness-oriented and CPU-only.

## Visible result

![Synthetic BEV result](docs/assets/synthetic_bev_demo.png)

Deterministic synthetic smoke test:

| Item | Result |
|---|---:|
| Input Gaussians | 2 |
| Voxel grid | 40 × 40 × 30 |
| Occupied voxels (`p ≥ 0.5`) | 649 |
| Voxels with Gaussian evidence | 18,712 |
| BEV cells with evidence | 834 |
| Maximum voxel occupancy | 0.898878 |

These values validate the data path; they are not public-dataset accuracy claims.

## Representation contract

Input:

```text
mean               [N, 3] world or reconstruction coordinates
scale              [N, 3] standard deviations for the CPU reference
opacity            [N]
semantic_logits    [N, C]
```

Output:

```text
occupancy          [X, Y, Z] probabilistic union of Gaussian evidence
semantic_probs     [X, Y, Z, C]
evidence           [X, Y, Z]
unknown_mask       [X, Y, Z]
BEV occupancy      [X, Y]
BEV semantics      [X, Y, C]
```

Important boundary: Gaussians provide **occupied-space evidence**, but do not prove that an unobserved cell is free. Low-evidence cells are therefore marked unknown. Free-space labels require calibrated camera/depth rays or another visibility model.

## Quick start

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_demo.py
```

To rebuild the figure:

```bash
python -m pip install -e ".[visualization]"
python scripts/run_demo.py
python scripts/visualize_demo.py
```

PowerShell users can also run the source directly with `$env:PYTHONPATH="src"` if the package is not installed.

## What is implemented

- Axis-aligned semantic Gaussian-to-voxel splatting with bounded local support.
- Probabilistic-union occupancy aggregation.
- Evidence-weighted semantic aggregation.
- Explicit unknown-space mask.
- 3D-to-BEV height reduction.
- Deterministic unit tests, synthetic artifact export and visualization.
- GitHub Actions test workflow.

## Honest project status

| Capability | Status |
|---|---|
| NumPy CPU reference | Completed |
| Unit tests and synthetic demo | Completed |
| Rotation-aware full covariance | Planned |
| PyTorch/autograd implementation | Planned |
| Camera-ray free-space supervision | Planned |
| Public-dataset evaluation | Planned |
| ROS2 `OccupancyGrid` publisher | Planned |

This supports the resume claim **“implemented a Gaussian-to-voxel reference baseline and BEV aggregation”**. It does not support claims of a finished dynamic 3DGS, BEV/OCC benchmark, or robot navigation stack.

## Next engineering steps

1. Export semantic Gaussians from an ILGS scene and define metric coordinates.
2. Add quaternion/full-covariance projection and PyTorch gradient checks.
3. Introduce camera-ray free/occupied/unknown supervision.
4. Evaluate occupied IoU, semantic mIoU, latency and memory on a public subset.
5. Add a ROS2 `nav_msgs/OccupancyGrid` adapter and simulation smoke test.

The frozen experiment protocol is in [`docs/experiment_plan.md`](docs/experiment_plan.md), and progress boundaries are tracked in [`docs/status.md`](docs/status.md).
