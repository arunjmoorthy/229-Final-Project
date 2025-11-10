# 🎯 Getting Synthetic LiDAR Data

## What You Need

For true Sim→Real LiDAR translation, you need:

1. ✅ **Real LiDAR** (target domain) - **YOU HAVE THIS** (nuScenes mini)
2. ⏳ **Synthetic LiDAR** (source domain) - **YOU NEED THIS**

---

## 📥 OPTION 1: SynLiDAR Dataset (RECOMMENDED)

### Quick Info
- **Source:** Nanyang Technological University
- **Size:** 5 GB (subset) to 245 GB (full)
- **Format:** `.bin` files (same as KITTI)
- **License:** Free for research
- **Quality:** High-quality, realistic geometry

### Download Links

**Option A: Mini Subset (~5 GB, 1000 scans)**
```bash
# Download from Google Drive or official source
# I'll provide exact commands below
```

**Option B: Small Subset (~25 GB, 5000 scans)**
```bash
# Better for full training
```

### Steps to Get SynLiDAR

1. **Visit:** https://github.com/xiaoaoran/SynLiDAR
2. **Register:** Fill out form for download access
3. **Download:** Get the subset you need
4. **Preprocess:** Convert to range-view format (I'll provide script)

---

## 🎮 OPTION 2: CARLA Simulator (DIY)

### Quick Info
- **Source:** Open-source driving simulator
- **Size:** ~10 GB simulator + your generated data
- **Format:** Custom (you generate it)
- **License:** MIT (fully open)
- **Quality:** Depends on your configuration

### Steps to Use CARLA

1. **Install CARLA:**
```bash
# Download CARLA 0.9.15
cd ~/Downloads
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Mac/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15
```

2. **Run CARLA:**
```bash
./CarlaUE4.sh -quality-level=Low
```

3. **Generate Data:** (I'll provide Python script)
```python
# Script to drive around and collect LiDAR scans
# Saves to .bin format
```

4. **Preprocess:** Convert to range-view format

---

## ⚡ OPTION 3: Quick Start (Use nuScenes Twice)

This is a **temporary hack** to validate your pipeline immediately.

### What This Does
- Uses nuScenes for BOTH input and target
- Model learns identity function (not useful for real Sim→Real)
- Good for: Pipeline validation, debugging, testing

### How to Use
**Already set up in your code!** Just run:
```bash
arch -arm64 python3 train_nuscenes.py --config config.yaml --stage direct --device cpu
```

### When to Use
- ✅ Right now: Validate pipeline works
- ✅ Before getting synthetic data
- ✅ Quick experiments
- ❌ Final results (not real Sim→Real)

---

## 🎯 RECOMMENDED PATH FOR YOU

### Today (1 hour): Validate Pipeline
```bash
# Use nuScenes as pseudo-synthetic (Option 3)
arch -arm64 python3 train_nuscenes.py --config config.yaml \
  --stage direct --device cpu --num_samples 100

# Train for 2-3 epochs to see it work
# Goal: MSE drops, PSNR increases
```

### Tomorrow (3-4 hours): Get Real Synthetic Data

**Option 1A: SynLiDAR Mini (EASIEST)**
1. Go to https://github.com/xiaoaoran/SynLiDAR
2. Request download access
3. Download ~5 GB subset
4. Run preprocessing script (I'll provide)
5. Train with real synthetic→real

**Option 1B: CARLA (MORE FLEXIBLE)**
1. Install CARLA (~1 hour)
2. Run data generation script (~1 hour)
3. Generate 1000 scans (~1 hour)
4. Preprocess to range-view format
5. Train with real synthetic→real

---

## 📊 Data Requirements Summary

| Phase | Real Data | Synthetic Data | Training Mode | Result |
|-------|-----------|----------------|---------------|--------|
| 1. Validation | nuScenes mini (323) | None (use nuScenes twice) | Real→Real | Pipeline validated |
| 2. Sim→Real | nuScenes mini (323) | SynLiDAR (1000) | **Synthetic→Real** | **Actual project** |
| 3. Full Training | nuScenes mini (323) | SynLiDAR (5000) | Synthetic→Real | Best results |

---

## 💾 Storage Requirements

```
data/
├── raw/
│   ├── nuscenes/          # 10 GB (already have)
│   └── synlidar/          # 5-25 GB (need to download)
│
└── processed/
    ├── nuscenes_mini/     # 500 MB (already have)
    │   ├── mini_train/    # 323 files
    │   └── mini_val/      # 81 files
    │
    └── synlidar/          # 1-5 GB (need to create)
        ├── train/         # 800 files
        └── val/           # 200 files
```

**Total additional storage needed:** 6-30 GB

---

## 🚀 Next Steps

**Tell me which option you want:**

**A)** SynLiDAR mini subset (5 GB) - I'll help you download and preprocess

**B)** CARLA simulator - I'll help you install and generate data

**C)** Quick validation first (nuScenes twice) - Already set up, just run training

**My recommendation: Start with C (5 minutes), then do A (tomorrow)**

