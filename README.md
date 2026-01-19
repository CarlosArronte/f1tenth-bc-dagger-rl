# f1tenth-bc-dagger-rl
BC of a standard control (PP) + Dagger + RL
# Dataset Specification — Behavioral Cloning (BC) for Autonomous Racing

This document **formally and precisely** defines the structure of the **Dataset (DS)** used to train a **Behavioral Cloning (BC)** policy for autonomous racing (F1Tenth / AutoDrive).
It reflects the **corrected and validated design decisions** discussed previously and must be treated as the authoritative reference.

---

## 1. Dataset Objective

The dataset is designed to train a policy:

```
π(o_t) → a_t
```

that replicates the behavior of an **expert controller**, using only:

* Perception (LiDAR + IMU)
* Vehicle dynamic state
* Local geometric context
* Minimal temporal information

### BC Targets (outputs)

* `steering_cmd`
* `throttle_cmd`

> Internal variables of the expert controller are **explicitly excluded** from BC inputs to prevent information leakage.

---

## 2. High-Level Structure

The dataset is conceptually divided into the following blocks:

1. LiDAR (sectorized raw data + per-sector geometry)
2. Global LiDAR-derived statistics
3. IMU and gyroscope
4. Vehicle dynamic state
5. Local geometric context
6. Temporal information
7. Control targets
8. Metadata (outside CSV)

---

## 3. LiDAR

### 3.1 `lidar_ranges` (input)

Sectorized LiDAR range measurements.

* Raw LiDAR beams are grouped into angular sectors.
* Higher angular resolution in the frontal region.
* Example layouts:

  * `10 × 36`
  * `30 × 12`
  * `10 × 36`

Each sector contains the raw range values of its assigned beams.

---

### 3.2 `sector_statistics` (input)

Geometric features computed **per sector**:

* `min_range_s`
  Minimum distance to the closest obstacle in sector *s*.

* `mean_range_s`
  Mean distance in sector *s*.

* `std_range_s`
  Standard deviation of distances in sector *s*.

* Sector percentiles:

  * `p10`
  * `p25`
  * `p50` (median)
  * `p75`
  * `p90`

* `range_gradient`
  Temporal change in free space:

  ```
  mean_range_s(t-1) − mean_range_s(t)
  ```

* `valid_ratio_s`
  Ratio of valid beams (not `inf`, not `range_max`).

* `mean_inverse_range`

  ```
  mean(1 / r)
  ```

  Emphasizes nearby obstacles.

---

### 3.3 `lidar_meta` (metadata — NOT in CSV)

Static LiDAR configuration parameters, stored externally:

* `angle_min`
* `angle_max`
* `angle_increment`
* `range_min`
* `range_max`
* `scan_time`
* `n_beams`

> These parameters are **not included in the CSV** to avoid redundancy but are logged for reproducibility and future LiDAR-agnostic pipelines.

---

## 4. Global LiDAR Statistics (input)

Features derived from groups of sectors:

* `left_free_space`
  Mean of `p75` over left-side sectors.

* `right_free_space`
  Mean of `p75` over right-side sectors.

* `front_free_space`
  `p90` of frontal sectors.

* `free_space_ratio_left_right`

  ```
  left_free_space − right_free_space
  ```

* `free_space_front_left_vs_front_right`
  Same comparison but restricted to the frontal region.

* `track_width_estimate`

  ```
  p50_left + p50_right
  ```

* `center_offset_lidar`

  ```
  (p50_right − p50_left) / 2
  ```

---

## 5. IMU and Gyroscope

### 5.1 IMU (input)

* Raw accelerations:

  * `imu_ax_raw`
  * `imu_ay_raw`
  * `imu_az_raw`

* Filtered accelerations (LPF + gravity compensation):

  * `imu_ax_filt`
  * `imu_ay_filt`

* Angular velocity:

  * `imu_yaw_rate`
  * `imu_yaw_rate_filt`

---

### 5.2 Gyro / Orientation

* `gyro_z` (raw yaw rate)
  **Used only in RL, not in BC**

* `gyro_z_filtered`

* `yaw`

* `yaw_filtered`

---

## 6. Vehicle Dynamic State (input)

* `speed`

* `speed_filtered`

* `side_slip_estimated`
  Estimated lateral slip angle.

* `yaw_rate_over_speed`

  ```
  yaw_rate / speed
  ```

  Dynamic curvature proxy.

---

## 7. Local Geometric Context (input)

These variables are **not derived from the expert controller**.

### 7.1 `estimated_track_curvature`

Estimated local track curvature from frontal LiDAR geometry.

* Unit: `1/m`
* Represents **actual track geometry**
* **Must not be confused with `curvature_cmd`**

---

### 7.2 `local_complexity_metric`

Scalar metric describing local track geometric complexity.

Definition:

```
local_complexity_metric =
    w1 · Var(r_front)
  + w2 · Mean(|Δr_front|)
```

Where:

* `r_front` are LiDAR ranges in the frontal sector.

This metric captures:

* Spatial variability (curves, chicanes)
* Abrupt geometric changes

Used for:

* Context awareness
* Potential loss weighting or gating

---

## 8. Temporal Information (input)

Minimal temporal features to encode dynamics without explicit RNNs:

* `dt`
* `time_since_last_cmd` (effective control latency)

Short history:

* `prev_steering`

* `prev2_steering`

* `prev_delta_steering`

* `prev_speed`

* `prev2_speed`

* `prev_delta_speed`

---

## 9. Control

### 9.1 BC Targets (outputs)

**The only supervised outputs of the BC model:**

* `steering_cmd`
* `throttle_cmd`

---

### 9.2 Control Variables (NOT used as BC inputs)

These variables are **explicitly excluded** from BC inputs to avoid expert leakage:

* `Ld`
* `heading_error_pp`
* `lateral_error_pp`
* `steering_clip_ratio`
* `target_speed_pp`
* `speed_error_pp`
* `curvature_based_speed`
* `delta_steering_cmd`
* `delta_throttle_cmd`

Allowed usage:

* Analysis
* Debugging
* Offline or online RL stages

---

## 10. Design Principle (Key)

> **Behavioral Cloning must learn driving from perception and state, not from the internal logic of the expert controller.**

This dataset strictly enforces that principle.

---

## 11. Summary Table

| Block             | Role                         |
| ----------------- | ---------------------------- |
| Sectorized LiDAR  | Primary geometric perception |
| LiDAR statistics  | Spatial abstraction          |
| IMU / Gyro        | Vehicle dynamics             |
| Vehicle state     | Kinematics                   |
| Context features  | Implicit future geometry     |
| Temporal features | Control dynamics             |
| Targets           | Pure expert action           |

---

**This document defines the final, validated dataset structure and should be used as the reference for implementation, logging, and training.**
