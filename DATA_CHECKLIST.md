# Data Checklist - What You Need

## ✅ Complete Checklist

### For Real LiDAR Data (Target Domain):

- [ ] **nuScenes mini dataset** (~3 GB)
  - [ ] Downloaded `v1.0-mini.tgz` (metadata)
  - [ ] Downloaded `samples-01.tgz` (LiDAR point clouds)
  - [ ] Extracted to `data/raw/nuscenes/`

- [ ] **nuScenes-lidarseg mini** (~2 GB)
  - [ ] Downloaded `v1.0-mini.tgz` (lidarseg metadata)
  - [ ] Downloaded `lidarseg-01.tgz` (semantic labels)
  - [ ] Extracted to `data/raw/nuscenes/`

**Total Real Data: ~5 GB** ✅

---

### For Synthetic LiDAR Data (Source Domain) - LATER:

- [ ] **SynLiDAR** (~20-40 GB) - **NOT NEEDED FOR INITIAL TEST**
  - We'll add this after verifying the pipeline works
  - Or use CARLA simulator to generate synthetic scans

**Note:** For now, we're testing with **real data only** to verify the pipeline works!

---

## 🎯 Current Goal: Test Pipeline with Real Data Only

You only need **nuScenes mini** right now to:
1. ✅ Verify preprocessing works
2. ✅ Test training loop (1 epoch)
3. ✅ Check evaluation metrics
4. ✅ Validate the entire pipeline

**After this works**, we'll add synthetic data for Sim→Real translation.

---

## 📍 Where Each File Goes

```
data/raw/nuscenes/
├── samples/              ← From samples-01.tgz
│   └── LIDAR_TOP/
│       ├── n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
│       └── ...
├── lidarseg/            ← From lidarseg-01.tgz
│   └── v1.0-mini/
│       ├── n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
│       └── ...
└── v1.0-mini/           ← From both v1.0-mini.tgz files (merged)
    ├── sample.json
    ├── scene.json
    ├── sample_data.json
    └── ...
```

---

## 🚀 Quick Start Commands

### 1. Download (manual - go to website)
- Visit: https://www.nuscenes.org/download
- Create account
- Download nuScenes mini + lidarseg mini

### 2. Extract
```bash
cd "/Users/arunmoorthy/Junior Year/CS 229/Final Project"
mkdir -p data/raw/nuscenes
cd data/raw/nuscenes

# Extract all archives here (they'll merge)
tar -xzf /path/to/v1.0-mini.tgz
tar -xzf /path/to/samples-01.tgz
tar -xzf /path/to/lidarseg/v1.0-mini.tgz
tar -xzf /path/to/lidarseg/lidarseg-01.tgz
```

### 3. Verify
```bash
# Check structure
ls data/raw/nuscenes/
# Should see: samples/, lidarseg/, v1.0-mini/

# Check files exist
ls data/raw/nuscenes/samples/LIDAR_TOP/ | wc -l
# Should show ~300-400 files
```

### 4. Preprocess
```bash
cd "/Users/arunmoorthy/Junior Year/CS 229/Final Project"
python scripts/preprocess_nuscenes.py \
  --data_root ./data/raw/nuscenes \
  --output_root ./data/processed \
  --split mini
```

### 5. Test Training
```bash
python test_training.py --config config_test.yaml --num_samples 50
```

---

## 📊 Data Size Comparison

| Dataset | Size | Status |
|---------|------|--------|
| SemanticKITTI | 80 GB | ❌ Too big |
| nuScenes mini | ~5 GB | ✅ Perfect! |
| nuScenes full | ~300 GB | ❌ Too big |
| SynLiDAR | ~40 GB | ⏳ Later |

---

## ✅ You're Ready When:

- [x] nuScenes mini downloaded
- [x] Extracted to `data/raw/nuscenes/`
- [x] Verified files exist
- [x] Ready to preprocess!

**Once you've downloaded nuScenes mini, run the preprocessing script and we'll test training!**

