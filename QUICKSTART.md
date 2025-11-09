# Quick Start Guide

This guide will help you get started with the LiDAR Sim2Real translation project.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: A100 with 80GB VRAM)
- 100GB+ storage for datasets

## Installation

1. **Clone the repository and install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up Modal (optional, for cloud training):**

```bash
pip install modal
modal token new
```

## Step-by-Step Workflow

### Phase 1: Data Preparation

#### 1.1 Download SemanticKITTI

1. Register at http://semantic-kitti.org/
2. Download:
   - Velodyne point clouds (80 GB)
   - Semantic labels (200 MB)
3. Extract to `./data/raw/SemanticKITTI/`

Expected structure:
```
data/raw/SemanticKITTI/
  sequences/
    00/
      velodyne/  (*.bin files)
      labels/    (*.label files)
    01/
    ...
```

#### 1.2 Download SynLiDAR (or use CARLA)

- **Option A: SynLiDAR**
  - Get from: https://github.com/xiaoaoran/SynLiDAR
  - Extract to `./data/raw/SynLiDAR/`

- **Option B: CARLA**
  - Set up CARLA simulator
  - Use regular LiDAR sensor (not semantic)
  - Generate scans and save as `.bin` files

#### 1.3 Preprocess Datasets

```bash
# Preprocess SemanticKITTI
python scripts/preprocess_semantickitti.py \
  --data_root ./data/raw/SemanticKITTI \
  --output_root ./data/processed

# Preprocess SynLiDAR (similar format)
# Modify the script for your specific synthetic data format
```

This converts point clouds to range-view format (64×1024 tensors).

#### 1.4 Learn Sensor Profile

Extract real sensor characteristics (dropout patterns, intensity falloff):

```bash
python scripts/learn_sensor_profile.py \
  --data_root ./data/processed \
  --output sensor_profile.json \
  --num_samples 500
```

This creates a `sensor_profile.json` file and visualization plot.

### Phase 2: Train Translator

#### 2.1 Stage A - Direct Translator (UNet)

Train the base UNet translator:

```bash
# Local training
python main.py \
  --config config.yaml \
  --stage direct \
  --output_dir ./outputs

# Or use Modal for cloud GPU
modal run modal_app.py --command train --stage direct
```

This takes ~6-12 hours on an A100.

**What it does:**
- Learns to map synthetic → real-style range images
- Uses L1 + perceptual + gradient losses
- Saves checkpoints every 5 epochs

#### 2.2 Stage B - Diffusion Upgrade (Optional)

Upgrade to diffusion for higher quality:

```bash
# Local training
python main.py \
  --config config.yaml \
  --stage diffusion \
  --checkpoint ./outputs/direct/checkpoints/best.pt \
  --output_dir ./outputs

# Or use Modal
modal run modal_app.py --command train --stage diffusion
```

This takes ~12-24 hours on an A100.

**What it does:**
- Wraps the UNet in a diffusion process
- Learns to denoise at multiple timesteps
- Better captures fine-grained details

### Phase 3: Generate & Evaluate

#### 3.1 Batch Translation

Translate synthetic validation set:

```bash
python scripts/translate_batch.py \
  --checkpoint ./outputs/diffusion/checkpoints/best.pt \
  --input_dir ./data/processed/synlidar_val \
  --output_dir ./outputs/translated \
  --batch_size 16 \
  --diffusion \
  --num_steps 50
```

#### 3.2 Compute Distribution Metrics

Evaluate FRID, FPD, MMD:

```bash
python eval/metrics.py \
  --real_dir ./data/processed/semantickitti_val \
  --generated_dir ./outputs/translated
```

**Success Criteria:**
- FRID (Translated) < FRID (Raw Synthetic)
- Lower is better; aim for 30-50% reduction

#### 3.3 Downstream Segmentation

Train segmentation models in different regimes:

```bash
# Run all three regimes
python scripts/eval_downstream.py \
  --config config.yaml \
  --all \
  --synthetic_dir ./outputs/translated \
  --output_dir ./outputs/segmentation
```

**Compares:**
1. Real-only baseline
2. Real + Raw Synthetic
3. Real + Translated Synthetic

**Success Criteria:**
- mIoU (Real + Translated) > mIoU (Real + Raw)
- mIoU (Real + Translated) ≥ mIoU (Real-only)

### Phase 4: Analysis & Ablations

#### 4.1 Visualize Results

```python
from eval.visualize import compare_range_views
import numpy as np

# Load samples
synthetic = np.load('./data/processed/synlidar_val/000000.npz')
real = np.load('./data/processed/semantickitti_val/000000.npz')
translated = np.load('./outputs/translated/000000.npz')

# Compare
compare_range_views(
    synthetic=dict(synthetic),
    real=dict(real),
    translated=dict(translated),
    save_path='./outputs/comparison.png'
)
```

#### 4.2 Run Ablations

Modify `config.yaml` to disable features one at a time:
- Remove circular padding: `use_circular_padding: false`
- Remove intensity normalization
- Remove calibrated augmentation

Re-run training and evaluation to measure impact.

## Configuration

Key parameters in `config.yaml`:

### Sensor Parameters
```yaml
sensor:
  n_rings: 64          # Vertical beams
  n_azimuth: 1024      # Horizontal resolution
  fov_up: 3.0          # Upper FOV (degrees)
  fov_down: -25.0      # Lower FOV (degrees)
  max_range: 80.0      # Max range (meters)
```

### Model Architecture
```yaml
model:
  base_channels: 64
  channel_multipliers: [1, 2, 4, 8]
  num_res_blocks: 2
  use_circular_padding: true  # Critical for 360° handling
```

### Training
```yaml
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0002
  mixed_precision: true
```

## Expected Results

**Metrics (after full training):**
- FRID: 50-150 (lower is better, depends on baseline)
- MMD: 0.1-0.5
- Segmentation mIoU improvement: +2-5% over raw synthetic

**Compute Budget:**
- Preprocessing: ~2 hours (CPU)
- Stage A training: ~8 hours (A100)
- Stage B training: ~16 hours (A100)
- Evaluation: ~2 hours (A100)
- **Total: ~28 hours @ $2.50/hr = ~$70**

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `proj_w` to 512 instead of 1024
- Use gradient checkpointing (add to model)

### Poor Results
- Check that synthetic and real data are properly aligned in range-view
- Verify sensor profile was learned correctly (check plots)
- Ensure circular padding is enabled
- Try longer training (more epochs)

### Data Loading Issues
- Verify file paths in config
- Check that preprocessing completed successfully
- Ensure `.npz` files contain expected keys: range, intensity, mask

## Next Steps

1. **Experiment with CARLA**: Generate edge cases (rain, fog, night)
2. **Try different architectures**: Larger models, attention mechanisms
3. **Domain randomization**: Add more synthetic variety before translation
4. **Real-time inference**: Optimize for deployment (quantization, pruning)

## Citation

If you use this code, please cite:
```bibtex
@article{lidar_sim2real_2025,
  title={Range-View Diffusion for LiDAR Sim2Real Translation},
  author={Your Name},
  year={2025}
}
```

## Support

For issues or questions:
1. Check the main README.md
2. Review config.yaml comments
3. Inspect logs in `./outputs/*/logs/`

