# Complete Setup & Training Guide

## 🎯 Two Options: Local GPU or Modal (Cloud)

### Option A: Local Training (If you have a GPU)
**Best for:** RTX 3090, RTX 4090, or better
**Cost:** $0 (uses your hardware)
**Time:** Same as cloud

### Option B: Modal Training (Recommended)
**Best for:** No GPU or want A100 performance
**Cost:** ~$2.50/hour (A100), total ~$70-100 for full project
**Time:** Faster training on A100

---

## 📦 Step 1: Initial Setup (Both Options)

### 1.1 Install Python Dependencies

```bash
cd "/Users/arunmoorthy/Junior Year/CS 229/Final Project"

# Install requirements
pip install -r requirements.txt

# Test installation
python test_installation.py
```

**Expected output:** All tests should pass ✓

---

## 📊 Step 2: Download Datasets

### 2.1 SemanticKITTI (Real LiDAR Data) - REQUIRED

**Size:** ~80 GB for velodyne, ~200 MB for labels

1. **Register & Download:**
   - Go to: http://semantic-kitti.org/
   - Create account and agree to terms
   - Download:
     - `data_odometry_velodyne.zip` (80 GB)
     - `data_odometry_labels.zip` (200 MB)

2. **Extract to:**
   ```bash
   mkdir -p data/raw/SemanticKITTI
   # Extract both zips to data/raw/SemanticKITTI/
   ```

3. **Verify structure:**
   ```
   data/raw/SemanticKITTI/
   └── sequences/
       ├── 00/
       │   ├── velodyne/
       │   │   ├── 000000.bin
       │   │   ├── 000001.bin
       │   │   └── ...
       │   └── labels/
       │       ├── 000000.label
       │       └── ...
       ├── 01/
       ├── 02/
       └── ...
   ```

### 2.2 SynLiDAR (Synthetic Data) - REQUIRED

**Size:** ~20-40 GB

1. **Get SynLiDAR:**
   - Visit: https://github.com/xiaoaoran/SynLiDAR
   - Follow their download instructions (usually Google Drive or similar)
   - Or generate your own with CARLA (see Alternative below)

2. **Extract to:**
   ```bash
   mkdir -p data/raw/SynLiDAR
   # Extract to data/raw/SynLiDAR/
   ```

3. **Expected structure:**
   ```
   data/raw/SynLiDAR/
   ├── train/
   │   └── velodyne/
   │       ├── 000000.bin
   │       └── ...
   └── val/
       └── velodyne/
           └── ...
   ```

**Alternative: CARLA Simulator**

If you can't get SynLiDAR, generate with CARLA:

```bash
# Install CARLA (0.9.13+)
pip install carla

# Use CARLA's Python API to:
# 1. Spawn vehicle with LiDAR sensor
# 2. Drive around
# 3. Save point clouds as .bin files (same format as KITTI)
```

---

## 🔧 Step 3: Preprocess Data

This converts 3D point clouds → 2D range views for training:

```bash
# Preprocess SemanticKITTI (takes ~30-60 minutes)
python scripts/preprocess_semantickitti.py \
  --data_root ./data/raw/SemanticKITTI \
  --output_root ./data/processed

# Preprocess SynLiDAR (takes ~20-40 minutes)
python scripts/preprocess_synlidar.py \
  --data_root ./data/raw/SynLiDAR \
  --output_root ./data/processed

# Learn sensor profile from real data (takes ~5 minutes)
python scripts/learn_sensor_profile.py \
  --data_root ./data/processed \
  --output sensor_profile.json \
  --num_samples 500
```

**Verify preprocessing worked:**
```bash
ls data/processed/semantickitti/00/*.npz  # Should see .npz files
ls data/processed/synlidar/train/*.npz    # Should see .npz files
ls sensor_profile.json                     # Should exist
```

---

## 🚀 Step 4A: Training - LOCAL (No Modal)

### Update config.yaml

```bash
# Edit config.yaml and set paths:
nano config.yaml
```

Update these lines:
```yaml
data:
  semantickitti_root: "/Users/arunmoorthy/Junior Year/CS 229/Final Project/data/processed"
  synlidar_root: "/Users/arunmoorthy/Junior Year/CS 229/Final Project/data/processed"
  output_root: "/Users/arunmoorthy/Junior Year/CS 229/Final Project/data/processed"
```

### Train Stage A (Direct Translator)

```bash
python main.py \
  --config config.yaml \
  --stage direct \
  --output_dir ./outputs \
  --device cuda
```

**Time:** ~8-12 hours on RTX 3090/4090, ~4-6 hours on A100

**What happens:**
- Trains UNet translator
- Saves checkpoints every 5 epochs to `./outputs/direct/checkpoints/`
- Logs to `./outputs/direct/logs/` (view with TensorBoard)
- Best model saved as `best.pt`

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir ./outputs/direct/logs
# Open http://localhost:6006
```

### Train Stage B (Diffusion - Optional)

After Stage A completes:

```bash
python main.py \
  --config config.yaml \
  --stage diffusion \
  --checkpoint ./outputs/direct/checkpoints/best.pt \
  --output_dir ./outputs \
  --device cuda
```

**Time:** ~16-24 hours on RTX 3090/4090, ~12-16 hours on A100

---

## 🚀 Step 4B: Training - MODAL (Cloud GPU)

### Modal Setup (One-time)

```bash
# Install Modal
pip install modal

# Create account and get token
modal token new

# This opens browser - follow instructions to authenticate
```

### Update modal_app.py paths

The Modal app needs to know where your data is. You have two options:

**Option 1: Upload to Modal Volumes (Recommended)**

```bash
# Create and upload data to Modal volumes
modal volume create lidar-data
modal volume create lidar-checkpoints  
modal volume create lidar-eval

# Upload preprocessed data
modal volume put lidar-data data/processed/semantickitti/ /data/SemanticKITTI/
modal volume put lidar-data data/processed/synlidar/ /data/SynLiDAR/
modal volume put lidar-data sensor_profile.json /data/
```

**Option 2: Mount from Cloud Storage**

If data is in S3/GCS, add to `modal_app.py`:
```python
# Add secret for cloud credentials
secrets=[
    modal.Secret.from_name("aws-secret"),  # or gcs-secret
]
```

### Train on Modal

```bash
# Stage A (Direct)
modal run modal_app.py --command train --stage direct

# Stage B (Diffusion) - after Stage A completes
modal run modal_app.py --command train --stage diffusion --checkpoint /checkpoints/direct/checkpoints/best.pt
```

**What happens:**
- Spins up A100 GPU instance
- Mounts data from volumes
- Trains model
- Saves checkpoints to volume
- Auto-shuts down when done

**Monitor Modal jobs:**
- Go to: https://modal.com/
- Click on your app
- View logs in real-time

**Download results:**
```bash
# Download trained model
modal volume get lidar-checkpoints /checkpoints/direct/checkpoints/best.pt ./outputs/

# Download all checkpoints
modal volume get lidar-checkpoints /checkpoints/ ./outputs/
```

---

## 📊 Step 5: Evaluate Results

### 5.1 Translate Synthetic Data

```bash
# Local
python scripts/translate_batch.py \
  --checkpoint ./outputs/direct/checkpoints/best.pt \
  --input_dir ./data/processed/synlidar/val \
  --output_dir ./outputs/translated \
  --batch_size 16

# Modal
modal run modal_app.py --command translate \
  --checkpoint /checkpoints/direct/checkpoints/best.pt
```

### 5.2 Compute Metrics

```bash
# Create evaluation script
python -c "
from eval.metrics import MetricsEvaluator
import torch
import numpy as np
from pathlib import Path

# Load real scans
real_files = sorted(Path('data/processed/semantickitti/08').glob('*.npz'))[:1000]
real_scans = []
for f in real_files:
    data = np.load(f)
    scan = np.stack([data['range'], data['intensity'], data['mask'].astype(float)], axis=0)
    real_scans.append(scan)
real_scans = torch.from_numpy(np.stack(real_scans))

# Load translated scans
trans_files = sorted(Path('outputs/translated').glob('*.npz'))[:1000]
trans_scans = []
for f in trans_files:
    data = np.load(f)
    scan = np.stack([data['range'], data['intensity'], data['mask'].astype(float)], axis=0)
    trans_scans.append(scan)
trans_scans = torch.from_numpy(np.stack(trans_scans))

# Compute metrics
evaluator = MetricsEvaluator()
metrics = evaluator.compute_metrics(real_scans, trans_scans)
print('Metrics:', metrics)
"
```

### 5.3 Downstream Evaluation

```bash
# Compare all three regimes
python scripts/eval_downstream.py \
  --config config.yaml \
  --all \
  --synthetic_dir ./outputs/translated \
  --output_dir ./outputs/segmentation
```

**This will:**
1. Train segmentation on Real-only
2. Train segmentation on Real + Raw Synthetic
3. Train segmentation on Real + Translated
4. Compare mIoU scores

---

## ⚡ Quick Command Reference

### Full Pipeline (Local)
```bash
# 1. Setup
pip install -r requirements.txt
python test_installation.py

# 2. Preprocess (after downloading data)
python scripts/preprocess_semantickitti.py --data_root ./data/raw/SemanticKITTI --output_root ./data/processed
python scripts/preprocess_synlidar.py --data_root ./data/raw/SynLiDAR --output_root ./data/processed
python scripts/learn_sensor_profile.py --data_root ./data/processed --output sensor_profile.json

# 3. Train
python main.py --stage direct --config config.yaml

# 4. Translate
python scripts/translate_batch.py --checkpoint ./outputs/direct/checkpoints/best.pt --input_dir ./data/processed/synlidar/val --output_dir ./outputs/translated

# 5. Evaluate
python scripts/eval_downstream.py --all --synthetic_dir ./outputs/translated
```

### Full Pipeline (Modal)
```bash
# 1. Setup Modal
modal token new
modal volume create lidar-data
modal volume create lidar-checkpoints
modal volume create lidar-eval

# 2. Upload data (after preprocessing locally)
modal volume put lidar-data data/processed/semantickitti/ /data/SemanticKITTI/
modal volume put lidar-data data/processed/synlidar/ /data/SynLiDAR/

# 3. Train
modal run modal_app.py --command train --stage direct

# 4. Download and evaluate locally
modal volume get lidar-checkpoints /checkpoints/direct/checkpoints/best.pt ./outputs/
python scripts/translate_batch.py --checkpoint ./outputs/best.pt --input_dir ./data/processed/synlidar/val --output_dir ./outputs/translated
python scripts/eval_downstream.py --all
```

---

## 💰 Cost Estimate

### Local Training (Free)
- Prerequisites: RTX 3090+ GPU, 32GB+ RAM
- Time: ~30 hours total
- Cost: $0

### Modal Training (Paid)
- A100-80GB: $2.50/hour
- Stage A: ~6 hours = $15
- Stage B: ~12 hours = $30
- Evaluation: ~2 hours = $5
- **Total: ~$50-70**

---

## 🐛 Troubleshooting

### "FileNotFoundError: data/raw/SemanticKITTI"
→ Download SemanticKITTI first (see Step 2.1)

### "CUDA out of memory"
→ Reduce batch_size in config.yaml:
```yaml
training:
  batch_size: 8  # or even 4
```

### "Modal authentication failed"
→ Run: `modal token new`

### "No module named 'data'"
→ Make sure you're in project root:
```bash
cd "/Users/arunmoorthy/Junior Year/CS 229/Final Project"
```

### Preprocessing takes forever
→ Normal! SemanticKITTI has ~23K scans. Takes 30-60 min.

### Training not starting
→ Check that preprocessed data exists:
```bash
ls data/processed/semantickitti/00/*.npz
ls data/processed/synlidar/train/*.npz
```

---

## 📈 What to Expect

### During Training
- **Epoch 1-10:** Loss drops rapidly
- **Epoch 10-50:** Gradual improvement
- **Epoch 50-100:** Fine-tuning

### Results
- **FRID:** Should see 30-50% reduction vs raw synthetic
- **mIoU:** +2-5% improvement with translated data

---

## ✅ Checklist

- [ ] Installed requirements
- [ ] Test passed (test_installation.py)
- [ ] Downloaded SemanticKITTI
- [ ] Downloaded SynLiDAR (or CARLA)
- [ ] Preprocessed both datasets
- [ ] Learned sensor profile
- [ ] Updated config.yaml paths
- [ ] Chose training method (local or Modal)
- [ ] (If Modal) Set up Modal account
- [ ] Started Stage A training
- [ ] Monitored training (TensorBoard)
- [ ] Translated validation set
- [ ] Evaluated metrics
- [ ] Ran downstream evaluation

---

## 🆘 Need Help?

1. Check `test_installation.py` passes
2. Review error messages in terminal
3. Check logs: `./outputs/*/logs/`
4. View TensorBoard: `tensorboard --logdir ./outputs`

**Common issues are in Troubleshooting section above!**

