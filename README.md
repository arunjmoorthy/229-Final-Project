# LiDAR Sim2Real Translation via Range-View Diffusion

A physics-aware, range-view based approach to translate synthetic LiDAR data to realistic scans using calibrated diffusion models.

## Overview

This project implements a Sim→Real translation pipeline for LiDAR point clouds that:
- Operates in the range-view (2D) representation for efficiency
- Learns sensor-specific characteristics (dropout, intensity falloff, ring artifacts)
- Uses circular padding to handle 360° azimuth wrapping
- Validates improvements via downstream segmentation tasks

## Project Structure

```
.
├── data/                   # Data loading and preprocessing
│   ├── loaders.py         # Dataset classes
│   ├── range_projection.py # Point cloud → range view
│   └── sensor_profiles.py # Calibration utilities
├── models/                 # Neural network architectures
│   ├── unet.py            # Base UNet with circular padding
│   ├── diffusion.py       # Diffusion wrapper
│   └── segmentation.py    # Downstream RV segmentation
├── augment/               # Data augmentation
│   ├── calibration.py    # Physics-based augmentation
│   └── transforms.py     # Standard augmentations
├── train/                 # Training scripts
│   ├── trainer.py        # Main training loop
│   └── losses.py         # Loss functions
├── eval/                  # Evaluation
│   ├── metrics.py        # FRID, FPD, MMD
│   └── visualize.py      # Visualization tools
├── scripts/               # Utility scripts
│   ├── preprocess_synlidar.py
│   ├── learn_sensor_profile.py
│   ├── translate_batch.py
│   └── eval_downstream.py
├── modal_app.py           # Modal deployment
├── config.yaml            # Configuration
└── requirements.txt       # Dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Modal (if using):
```bash
modal token new
```

3. Download datasets:
   - SynLiDAR: https://github.com/xiaoaoran/SynLiDAR

## Usage

### Phase 1: Data Preprocessing
```bash
python scripts/preprocess_synlidar.py --data_root /path/to/synlidar
python scripts/learn_sensor_profile.py --data_root /path/to/processed/real_data
```

### Phase 2: Train Translator
```bash
# Stage A: Direct translator
python train/trainer.py --config config.yaml --stage direct

# Stage B: Upgrade to diffusion
python train/trainer.py --config config.yaml --stage diffusion --load_checkpoint checkpoints/direct_best.pt
```

### Phase 3: Translate & Evaluate
```bash
# Batch translation
python scripts/translate_batch.py --checkpoint checkpoints/diffusion_best.pt --input_dir synthetic --output_dir translated

# Compute metrics
python eval/metrics.py --generated_dir translated --real_dir real_val

# Downstream evaluation
python scripts/eval_downstream.py --synthetic_type [raw|translated]
```

## Key Features

- **Physics-Aware**: Incorporates beam angles, dropout patterns, and intensity falloff from real sensors
- **Circular Padding**: Handles 360° azimuth wrapping correctly
- **Masked Losses**: Ignores invalid pixels throughout training and evaluation
- **Two-Stage Training**: Quick UNet baseline → high-quality diffusion upgrade
- **Budget-Conscious**: Designed for <$1k compute budget on Modal/cloud GPUs

## Citation

If you use this code, please cite:
```
@article{lidar_sim2real_2025,
  title={Range-View Diffusion for LiDAR Sim2Real Translation},
  author={Your Name},
  year={2025}
}
```

