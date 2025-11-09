# What Data Do I Need? (Simple Answer)

## 🎯 Short Answer

You need **ONE dataset** to test the pipeline:

### **nuScenes mini** (~5 GB total)

This replaces SemanticKITTI (which was 80GB - too big).

---

## 📥 What to Download

Go to: **https://www.nuscenes.org/download**

Create account → Download these 4 files:

1. **nuScenes mini metadata** (~50 MB)
   - File: `v1.0-mini.tgz`

2. **nuScenes mini LiDAR samples** (~2-3 GB)
   - File: `samples-01.tgz`

3. **nuScenes-lidarseg mini metadata** (~10 MB)
   - File: `v1.0-mini.tgz` (from lidarseg section)

4. **nuScenes-lidarseg mini labels** (~1-2 GB)
   - File: `lidarseg-01.tgz`

**Total: ~5 GB** (vs 80 GB for SemanticKITTI!)

---

## 📁 Where to Put It

Extract **all 4 files** into this folder:

```
/Users/arunmoorthy/Junior Year/CS 229/Final Project/data/raw/nuscenes/
```

**Quick command:**
```bash
cd "/Users/arunmoorthy/Junior Year/CS 229/Final Project"
mkdir -p data/raw/nuscenes
cd data/raw/nuscenes

# Extract all archives here (they'll merge automatically)
tar -xzf /path/to/v1.0-mini.tgz
tar -xzf /path/to/samples-01.tgz
tar -xzf /path/to/lidarseg/v1.0-mini.tgz
tar -xzf /path/to/lidarseg/lidarseg-01.tgz
```

---

## ✅ Verify It Worked

After extraction, check:

```bash
ls data/raw/nuscenes/
# Should see: samples/, lidarseg/, v1.0-mini/

ls data/raw/nuscenes/samples/LIDAR_TOP/ | wc -l
# Should show ~300-400 files
```

---

## 🚀 Then Run This

```bash
# 1. Preprocess (converts to range-view format)
python scripts/preprocess_nuscenes.py \
  --dataroot ./data/raw/nuscenes \
  --version v1.0-mini \
  --output_root ./data/processed/nuscenes_mini

# 2. Test training (1 epoch on CPU)
python test_training.py --config config_test.yaml --num_samples 50
```

---

## 📊 That's It!

**You only need nuScenes mini** (~5 GB) to:
- ✅ Test preprocessing
- ✅ Test training loop
- ✅ Verify pipeline works
- ✅ Run 1 epoch on CPU

**Later** (after testing works), we'll add:
- SynLiDAR or CARLA for synthetic data
- Then train the Sim→Real translator

---

## ❓ Still Confused?

1. **Go to:** https://www.nuscenes.org/download
2. **Download:** nuScenes mini + lidarseg mini (4 files total)
3. **Extract:** All to `data/raw/nuscenes/`
4. **Run:** Preprocessing script (above)

**That's all you need right now!**

