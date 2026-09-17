# Status

## 2026-09-16

- Created the standalone repository scaffold.
- Froze the first indoor grid and evaluation configuration.
- Implemented CPU diagonal-covariance Gaussian-to-voxel splatting.
- Implemented occupancy and semantic BEV reduction.
- Added unit tests and a synthetic demo.
- Added a reproducible BEV visualization and machine-readable demo summary.
- Added a bilingual, recruiter-facing README with explicit claim boundaries.
- Added GitHub Actions CI for tests and the synthetic smoke test.
- Added an ILGS PLY adapter that activates stored scale/opacity/quaternion parameters,
  preserves rotated covariance extent and supports explicit reconstruction-to-robot transforms.
- Ran the adapter on a real 18,223-Gaussian ILGS `ramen / pork belly` query export;
  published the hashed run summary, frozen filtering protocol and diagnostic figure.
- Converted the complete 854,507-Gaussian `ramen` scene and 256-way classifier into a
  memory-bounded hard-label semantic occupancy baseline with 47 active object IDs.
- Grounded classifier IDs 212 and 120 to the human-readable `pork belly` query using the
  hashed 18,223-Gaussian query export; other query labels remain explicitly unmapped.

Current claim level: a runnable reference baseline exists. Dataset-scale results, free-space ray supervision, dynamic reconstruction and ROS2 integration are not complete.

## 2026-09-17

- Completed ILGS query-mask exports for six `ramen` language queries and recorded their
  17 non-overlapping classifier object IDs.
- Projected the named IDs through the bounded full-scene hard-label BEV; all configured
  IDs are visible and together cover 9,097 of 36,196 cells with evidence.
- Added a reproducible three-panel named-query visualization and machine-readable summary.
- Flagged `chopsticks` as a diagnostic result because its 213,370-Gaussian export and
  7,152-cell footprint are unusually broad and may contain background leakage.
- Audited all six mappings against a fresh full-scene classifier pass: every export count
  matches its mapped-ID membership exactly. For `chopsticks`, ID 1 contributes 169,409
  Gaussians (79.40%) and is now the explicit source-mask review candidate.

Current claim level: the repository now demonstrates a reproducible open-vocabulary
query-to-object-ID-to-BEV bridge. It still does not provide metric robot-frame coordinates,
free-space supervision, semantic OCC ground truth, dynamic reconstruction or ROS2 closure.
