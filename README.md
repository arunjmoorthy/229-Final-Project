# Learning Realistic LiDAR: Sim-to-Real Translation for 3D Perception

This project studies **Sim-to-Real translation for LiDAR data**, aiming to close the realism gap between **synthetic LiDAR scans** generated in simulators (CARLA / SynLiDAR) and **real-world sensor data** (nuScenes). We propose a **lightweight, two-stage pipeline** that combines **physics-inspired sensor calibration** with **neural residual refinement** in **range-view space**, achieving strong realism improvements at low computational cost.

---

## Motivation

Synthetic LiDAR is widely used to train 3D perception models due to the high cost of collecting real sensor data. However, simulated LiDAR differs substantially from real scans:

* Unrealistically dense point clouds
* Missing beam-level dropout and ring failures
* Uniform or inaccurate intensity distributions

These discrepancies create a large **simulation-to-real gap**, causing models trained on synthetic data to generalize poorly in real-world environments.

Our goal is to **translate synthetic LiDAR into realistic scans** that match real sensor statistics **without altering scene geometry**.

---

## Key Idea

Instead of using fully generative or diffusion-based models, we leverage the structure of LiDAR sensors:

> **Inject measured sensor noise first, then learn only the residual differences.**

This leads to a model that is:

* **More stable to train**
* **Computationally efficient**
* **Easier to interpret**
* **Less data-hungry**

---

## Method Overview

### 1. Range-View Representation

Each 3D LiDAR scan is projected onto a spherical range image, producing a tensor
**X ∈ ℝ³×⁶⁴×⁵¹²** with channels:

* **Range** (distance)
* **Intensity**
* **Validity mask**

This representation preserves beam structure and enables efficient 2D CNN processing.

---

### 2. Stage 1 — Calibrated Realism Module

We estimate **per-beam dropout rates** and **intensity noise distributions** from real nuScenes data and deterministically apply them to synthetic scans. This stage injects realistic sensor artifacts **without learning** and preserves underlying geometry.

---

### 3. Stage 2 — Residual U-Net Refinement

A compact **range-view U-Net** learns to correct higher-order, non-linear discrepancies:

[
\hat{x}*{real} = x*{calib} + U_\theta(x_{calib})
]

Key design choices:

* Circular padding for 360° azimuth continuity
* Masked L1 loss over valid returns
* Gradient loss to preserve geometric structure

---

## Results

We evaluate realism using **distributional and geometric metrics** between translated and real LiDAR:

| Metric           | Before | After | Improvement |
| ---------------- | ------ | ----- | ----------- |
| FRID ↓           | 25.04  | 0.031 | ~800×       |
| MMD (RBF) ↓      | 1.40   | 0.175 | 8×          |
| MMD (Linear) ↓   | 5.00   | 0.176 | 28×         |
| Range MAE ↓      | 13.38  | 0.060 | >200×       |
| Range Std Diff ↓ | 11.75  | 0.015 | >700×       |

Qualitative results show translated synthetic LiDAR closely matches real nuScenes scans in sparsity patterns, intensity falloff, and beam-level artifacts.

---

## Project Structure

```
.
├── scripts/
│   ├── preprocess_synlidar.py
│   ├── learn_sensor_profile.py
│   └── translate_batch.py
├── train/
│   └── trainer.py
├── eval/
│   └── metrics.py
├── outputs/
└── README.md
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Preprocess & Calibrate

```bash
python scripts/preprocess_synlidar.py
python scripts/learn_sensor_profile.py --data_root data/processed/nuscenes_mini
```

### Train

```bash
python train/trainer.py --stage direct
python train/trainer.py --stage diffusion --load_checkpoint outputs/sim2real/direct/checkpoints/best.pt
```

### Translate & Evaluate

```bash
python scripts/translate_batch.py --checkpoint outputs/sim2real/diffusion/checkpoints/best.pt
python eval/metrics.py --generated_dir translated --real_dir data/processed/nuscenes_mini
```

---

## Takeaways

* Strong LiDAR realism gains **without diffusion or GANs**
* Combines **physics-based calibration + neural learning**
* Efficient and stable even with limited real data
* Well-suited for downstream 3D perception tasks

---

## Authors

**Arun Moorthy**, **Krish Sharma**
CS 229 — Stanford University

---

## Future Work

* Distance- and environment-conditioned noise models
* Evaluation on downstream tasks (segmentation, detection)
* Extension to multi-return and multi-sensor LiDAR
