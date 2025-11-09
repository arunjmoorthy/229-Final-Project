# 🚀 START HERE

## Quick Answer to Your Questions

### ❓ "Where is the data?"

**You need to download it:**

1. **SemanticKITTI** (Real LiDAR - REQUIRED, ~80GB)
   - Go to: http://semantic-kitti.org/
   - Register & download velodyne + labels
   - Extract to: `data/raw/SemanticKITTI/`

2. **SynLiDAR** (Synthetic LiDAR - REQUIRED, ~20-40GB)
   - Go to: https://github.com/xiaoaoran/SynLiDAR
   - Follow download instructions
   - Extract to: `data/raw/SynLiDAR/`

### ❓ "Do I need Modal?"

**NO, you have TWO options:**

**Option A: Train Locally (FREE)**
- Need: GPU (RTX 3090+), 32GB+ RAM
- Cost: $0
- Use if: You have a good GPU

**Option B: Train on Modal (PAID)**
- Need: Internet connection
- Cost: ~$50-70 total (A100 @ $2.50/hr)
- Use if: No GPU or want faster training

### ❓ "How do I start training?"

## 🎯 THREE SIMPLE STEPS

### **Step 1: Setup** (5 minutes)

**If training locally:**
```bash
./setup_local.sh
```

**If using Modal (cloud):**
```bash
./setup_local.sh  # Still need to do preprocessing locally first
```

### **Step 2: Get & Preprocess Data** (1-2 hours)

1. Download datasets (see "Where is the data?" above)
2. Run preprocessing:
```bash
./preprocess_data.sh
```

### **Step 3: Train** (6-24 hours depending on GPU)

**If training locally:**
```bash
python3 main.py --stage direct --config config.yaml
```

**If using Modal:**
```bash
./setup_modal.sh  # One-time setup
modal run modal_app.py --command train --stage direct
```

---

## 📋 Complete Commands (Copy & Paste)

### Local Training (Full Pipeline)
```bash
# 1. Setup
./setup_local.sh

# 2. Download data manually (see links above)
#    - SemanticKITTI → data/raw/SemanticKITTI/
#    - SynLiDAR → data/raw/SynLiDAR/

# 3. Preprocess
./preprocess_data.sh

# 4. Train
python3 main.py --stage direct --config config.yaml

# 5. Monitor (in another terminal)
tensorboard --logdir ./outputs
```

### Modal Training (Full Pipeline)
```bash
# 1. Setup
./setup_local.sh

# 2. Download data manually (see links above)

# 3. Preprocess locally
./preprocess_data.sh

# 4. Setup Modal & upload data
./setup_modal.sh

# 5. Train on cloud
modal run modal_app.py --command train --stage direct

# 6. Download results
modal volume get lidar-checkpoints /checkpoints/direct/checkpoints/best.pt ./outputs/
```

---

## 📚 Detailed Documentation

- **SETUP_GUIDE.md** - Complete setup instructions with troubleshooting
- **QUICKSTART.md** - Full project walkthrough
- **config.yaml** - All settings (edit this to tune hyperparameters)

---

## ⏱️ Time & Cost Estimates

### Local Training
- Setup: 5 min
- Download data: 2-4 hours (depends on internet)
- Preprocessing: 1-2 hours
- Training Stage A: 8-12 hours (RTX 3090/4090)
- **Total: ~12-18 hours**
- **Cost: $0**

### Modal Training
- Setup: 10 min
- Download data: 2-4 hours
- Preprocessing: 1-2 hours (local)
- Upload to Modal: 30 min
- Training Stage A: 6 hours (A100)
- **Total: ~10-13 hours**
- **Cost: ~$15-20 for Stage A**

---

## ✅ What You Get

After training, you'll have:
- ✅ Trained Sim→Real translator
- ✅ Translated synthetic scans (look realistic)
- ✅ Evaluation metrics (FRID, MMD, etc.)
- ✅ Downstream task results (segmentation mIoU)
- ✅ All for your CS 229 final project!

---

## 🆘 Help!

### "Command not found: python3"
Try `python` instead:
```bash
python main.py --stage direct --config config.yaml
```

### "No such file or directory: data/raw/SemanticKITTI"
You need to download the data first (see "Where is the data?" above)

### "CUDA out of memory"
Edit `config.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Other issues?
Read **SETUP_GUIDE.md** for detailed troubleshooting

---

## 🎓 For Your CS 229 Project

This gives you:
1. ✅ Novel approach (range-view diffusion)
2. ✅ Complete implementation
3. ✅ Rigorous evaluation
4. ✅ Reproducible results
5. ✅ Full documentation

Good luck! 🚀

