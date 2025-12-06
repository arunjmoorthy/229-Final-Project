# LiDAR Sim2Real Translation

Translate synthetic LiDAR data (SynLiDAR) to realistic scans (nuScenes) using a calibrated diffusion model.

## How It Works

1.  **Range View**: Converts 3D point clouds into 2D "range images" (like a panorama).
2.  **Calibration**: Learns noise and dropout patterns from real sensor data (nuScenes).
3.  **Translation**: Uses a diffusion model to refine synthetic scans to look real, preserving geometry while adding realistic sensor artifacts.
4.  **Verification**: Validates quality by training a segmentation model on the translated data.

## Quick Start

**1. Install & Setup**
```bash
pip install -r requirements.txt
```

**2. Preprocess Data**
```bash
python scripts/preprocess_synlidar.py
python scripts/learn_sensor_profile.py --data_root data/processed/nuscenes_mini
```

**3. Train**
```bash
python train/trainer.py --stage direct
python train/trainer.py --stage diffusion --load_checkpoint outputs/sim2real/direct/checkpoints/best.pt
```

**4. Translate & Eval**
```bash
python scripts/translate_batch.py --checkpoint outputs/sim2real/diffusion/checkpoints/best.pt
python eval/metrics.py --generated_dir translated --real_dir data/processed/nuscenes_mini
```
