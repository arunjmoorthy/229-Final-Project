# nuScenes Mini Setup Guide

## 🎯 What You Need (Total: ~3-6 GB)

Since SemanticKITTI is too big (80GB), we're using **nuScenes mini** instead (~3-6 GB total).

### Required Downloads:

1. **nuScenes mini** (~2-3 GB)
   - LiDAR point clouds (samples folder)
   - Metadata (v1.0-mini folder)

2. **nuScenes-lidarseg mini** (~1-2 GB) 
   - Semantic labels for point clouds
   - Required for downstream segmentation evaluation

3. **nuScenes devkit** (already in requirements.txt)
   - Python package for reading nuScenes data
   - Already added to requirements.txt

---

## 📥 Step-by-Step Download Instructions

### Step 1: Create Account & Download

1. **Go to nuScenes download page:**
   - Visit: https://www.nuscenes.org/download
   - Create account (free, academic email helps)
   - Agree to Terms of Use

2. **Download nuScenes mini:**
   - Look for "nuScenes mini" section
   - Download these files:
     - `v1.0-mini.tgz` (metadata, ~50 MB)
     - `samples-01.tgz` (LiDAR samples, ~2-3 GB)
     - `sweeps-01.tgz` (optional, intermediate frames, ~1-2 GB) - **You can skip this for testing**

3. **Download nuScenes-lidarseg mini:**
   - Look for "nuScenes-lidarseg" section
   - Download:
     - `v1.0-mini.tgz` (metadata, ~10 MB)
     - `lidarseg-01.tgz` (semantic labels, ~1-2 GB)

**Total downloads: ~3-6 GB** (much smaller than SemanticKITTI's 80GB!)

---

## 📁 Where to Put the Data

### Directory Structure:

```
/Users/arunmoorthy/Junior Year/CS 229/Final Project/
└── data/
    └── raw/
        └── nuscenes/          ← CREATE THIS FOLDER
            ├── samples/       ← Extract samples-01.tgz here
            ├── sweeps/        ← Extract sweeps-01.tgz here (optional)
            ├── lidarseg/      ← Extract lidarseg-01.tgz here
            └── v1.0-mini/     ← Extract both v1.0-mini.tgz files here
                                (they'll merge automatically)
```

### Extraction Steps:

```bash
# Navigate to project
cd "/Users/arunmoorthy/Junior Year/CS 229/Final Project"

# Create directory
mkdir -p data/raw/nuscenes

# Extract nuScenes mini (do this first)
cd data/raw/nuscenes

# Extract metadata
tar -xzf /path/to/v1.0-mini.tgz

# Extract samples (LiDAR point clouds)
tar -xzf /path/to/samples-01.tgz

# Extract sweeps (optional - skip if you want to save space)
# tar -xzf /path/to/sweeps-01.tgz

# Extract lidarseg metadata (will merge with v1.0-mini)
tar -xzf /path/to/lidarseg/v1.0-mini.tgz

# Extract lidarseg labels
tar -xzf /path/to/lidarseg/lidarseg-01.tgz
```

**Important:** Extract all archives into the **same folder** (`data/raw/nuscenes/`). The folder structures will merge automatically.

---

## ✅ Verify Your Setup

After extraction, verify you have this structure:

```bash
cd "/Users/arunmoorthy/Junior Year/CS 229/Final Project"

# Check structure
ls -la data/raw/nuscenes/
# Should see: samples/, sweeps/, lidarseg/, v1.0-mini/

# Check samples exist
ls data/raw/nuscenes/samples/LIDAR_TOP/ | head -5
# Should see .pcd.bin files

# Check labels exist
ls data/raw/nuscenes/lidarseg/v1.0-mini/ | head -5
# Should see .bin files

# Check metadata
ls data/raw/nuscenes/v1.0-mini/
# Should see: sample.json, scene.json, etc.
```

---

## 🚀 Next Steps After Download

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `nuscenes-devkit` automatically.

### 2. Preprocess nuScenes to Range View

```bash
python scripts/preprocess_nuscenes.py \
  --data_root ./data/raw/nuscenes \
  --output_root ./data/processed \
  --split mini
```

This converts nuScenes point clouds → range-view format (32×1024 images) using the Velodyne HDL-32E geometry.

> **Tip:** Use `config_nuscenes.yaml` for full training runs so the loader expects the 32-ring geometry and automatically picks up the preprocessed NPZ splits.

**Time:** ~10-20 minutes for mini dataset

### 3. Test Training (1 epoch on CPU)

```bash
python test_training.py --config config_test.yaml --num_samples 50
```

Or train downstream segmentation:

```bash
python scripts/eval_downstream.py \
  --config config_test.yaml \
  --regime real \
  --dataset_type nuscenes_npz \
  --real_root ./data/processed/nuscenes_mini
```

---

## 📊 What You Get

After preprocessing, you'll have:

- **~300-400 preprocessed scans** (nuScenes mini)
- **Range-view format** (32×1024 by default; adjust `--proj_h/--proj_w` if needed)
- **Semantic labels** for segmentation evaluation
- **Ready for training!**

---

## 🔄 Alternative: Use Even Smaller Subset

If you want to test with even less data:

```bash
# After preprocessing, you can limit samples in config_test.yaml
# Or use --num_samples flag in test_training.py
python test_training.py --num_samples 20  # Only 20 scans
```

---

## ❓ Troubleshooting

### "nuscenes-devkit not found"
```bash
pip install nuscenes-devkit
```

### "No samples found"
- Check you extracted `samples-01.tgz` to `data/raw/nuscenes/samples/`
- Verify files exist: `ls data/raw/nuscenes/samples/LIDAR_TOP/`

### "No labels found"
- Check you extracted `lidarseg-01.tgz` to `data/raw/nuscenes/lidarseg/`
- Verify metadata: `ls data/raw/nuscenes/v1.0-mini/`

### "Permission denied"
```bash
chmod -R 755 data/raw/nuscenes
```

---

## 📝 Summary

**What to download:**
1. ✅ nuScenes mini (~3 GB)
2. ✅ nuScenes-lidarseg mini (~2 GB)
3. ✅ Total: ~5 GB (vs 80 GB for SemanticKITTI!)

**Where to put it:**
- `data/raw/nuscenes/` (extract all archives here)

**What happens next:**
- Preprocess → converts to range-view format
- Train → 1 epoch test on CPU
- Then move to GPU for full training

---

**Once you've downloaded and extracted, let me know and we'll run the preprocessing!**

