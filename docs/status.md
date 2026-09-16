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

Current claim level: a runnable reference baseline exists. Dataset-scale results, free-space ray supervision, dynamic reconstruction and ROS2 integration are not complete.
