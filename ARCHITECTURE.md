# Project Architecture

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION PHASE                           │
└─────────────────────────────────────────────────────────────────────────┘

Raw Point Clouds                    Sensor Profile Learning
┌──────────────┐                   ┌────────────────────┐
│ SemanticKITTI│                   │  Real Scans (500+) │
│  .bin files  │                   │                    │
└──────┬───────┘                   └─────────┬──────────┘
       │                                     │
       │ RangeProjection()                   │ learn_from_scans()
       │                                     │
       ↓                                     ↓
┌──────────────┐                   ┌────────────────────┐
│  Range Views │                   │ Sensor Profile     │
│ 64×1024×3    │                   │ - dropout_vs_range │
│ [R, I, M]    │                   │ - intensity_stats  │
└──────┬───────┘                   │ - ring_dropouts    │
       │                           └─────────┬──────────┘
       │                                     │
       │                                     │ save as JSON
┌──────┴───────┐                   ┌─────────┴──────────┐
│ SynLiDAR     │                   │sensor_profile.json │
│  .bin files  │                   └────────────────────┘
└──────┬───────┘
       │
       │ RangeProjection()
       │
       ↓
┌──────────────┐
│  Range Views │
│ 64×1024×3    │
└──────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                                   │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────────┐
                        │ Sensor Profile     │
                        │ (JSON)             │
                        └─────────┬──────────┘
                                  │
                                  ↓
Synthetic Scans          Calibrated Augmentation       Real Scans
┌─────────────┐         ┌──────────────────────┐      ┌─────────────┐
│ Range View  │────────→│ - Apply dropout      │      │ Range View  │
│ [R, I, M]   │         │ - Intensity falloff  │      │ [R, I, M]   │
│             │         │ - Ring artifacts     │      │             │
└─────────────┘         │ - Range noise        │      └──────┬──────┘
                        └──────────┬───────────┘             │
                                   │                         │
                                   ↓                         │
                           ┌──────────────┐                  │
                           │ Augmented    │                  │
                           │ Synthetic    │                  │
                           └──────┬───────┘                  │
                                  │                          │
                                  └─────────┬────────────────┘
                                            │
                                            ↓
                                  ┌──────────────────┐
                                  │                  │
                           ┌──────┤   TRANSLATOR     ├──────┐
                           │      │                  │      │
                           │      └──────────────────┘      │
                           │                                │
                    Stage A│                                │Stage B
                    (Direct)                                │(Diffusion)
                           │                                │
                           ↓                                ↓
              ┌────────────────────────┐      ┌────────────────────────┐
              │     UNet Translator    │      │   Diffusion Model      │
              │  ┌──────────────────┐  │      │  ┌──────────────────┐  │
              │  │ Input: [R,I,M,θ] │  │      │  │ UNet Denoiser    │  │
              │  │                  │  │      │  │ + Time Embedding │  │
              │  │ Encoder (Down)   │  │      │  │ + CFG            │  │
              │  │  ↓ Circular Pad  │  │      │  │                  │  │
              │  │ Middle (Attn)    │  │      │  │ T=1000 steps     │  │
              │  │  ↓ Circular Pad  │  │      │  │ Sampling: 50     │  │
              │  │ Decoder (Up)     │  │      │  └──────────────────┘  │
              │  │                  │  │      │                        │
              │  │ Output: [R,I,M]  │  │      └────────────────────────┘
              │  └──────────────────┘  │
              │                        │
              │ Loss: L1 + Perceptual  │
              │       + Gradient       │
              └────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PHASE                                 │
└─────────────────────────────────────────────────────────────────────────┘

                        Trained Translator
                        ┌────────────────┐
                        │  best.pt       │
                        └────────┬───────┘
                               │
                Batch Translation
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ↓                      ↓                      ↓
┌───────────────┐    ┌────────────────┐    ┌────────────────┐
│Raw Synthetic  │    │Translated      │    │Real            │
│Range Views    │    │Range Views     │    │Range Views     │
└───────┬───────┘    └────────┬───────┘    └────────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                Distribution Metrics
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ↓                           ↓
    ┌───────────────────┐       ┌───────────────────┐
    │ Feature Extractor │       │ Point Cloud Comp  │
    │ (CNN)             │       │ (Chamfer)         │
    └─────────┬─────────┘       └─────────┬─────────┘
              │                           │
              ↓                           ↓
    ┌───────────────────┐       ┌───────────────────┐
    │ FRID: Fréchet     │       │ FPD: Point Cloud  │
    │ MMD: Kernel       │       │ Distance          │
    └───────────────────┘       └───────────────────┘

                Downstream Task Evaluation
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ↓                     ↓                     ↓
┌───────────────┐    ┌────────────────┐    ┌────────────────┐
│Train on:      │    │Train on:       │    │Train on:       │
│Real only      │    │Real + Raw Syn  │    │Real + Trans    │
│               │    │                │    │                │
│RangeNet       │    │RangeNet        │    │RangeNet        │
└───────┬───────┘    └────────┬───────┘    └────────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    Validate on Real
                              │
                              ↓
                    ┌─────────────────┐
                    │ mIoU Comparison │
                    │                 │
                    │ Goal:           │
                    │ Trans > Raw     │
                    └─────────────────┘
```

## Component Interaction Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                        Training Loop                            │
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                    │
│  │  DataLoader  │────────→│   Model      │                    │
│  │  (Synthetic  │         │  (UNet or    │                    │
│  │   + Real)    │         │   Diffusion) │                    │
│  └──────────────┘         └──────┬───────┘                    │
│         ↑                        │                             │
│         │                        ↓                             │
│         │                 ┌──────────────┐                    │
│         │                 │  Loss Fn     │                    │
│         │                 │ (Masked L1)  │                    │
│         │                 └──────┬───────┘                    │
│         │                        │                             │
│         │                        ↓                             │
│  ┌──────┴──────┐          ┌──────────────┐                   │
│  │ Augmentation│          │  Optimizer   │                   │
│  │ (Calibrated)│          │  (AdamW)     │                   │
│  └─────────────┘          └──────┬───────┘                   │
│         ↑                        │                             │
│         │                        ↓                             │
│  ┌──────┴──────────┐      ┌──────────────┐                   │
│  │ Sensor Profile  │      │  Scheduler   │                   │
│  │ (Learned)       │      │  (Cosine)    │                   │
│  └─────────────────┘      └──────────────┘                   │
│                                  │                             │
│                                  ↓                             │
│                           ┌──────────────┐                    │
│                           │ Checkpoint   │                    │
│                           │ Save/Load    │                    │
│                           └──────────────┘                    │
└────────────────────────────────────────────────────────────────┘
```

## Model Architecture Details

### UNet with Circular Padding

```
Input: [B, 4, 64, 1024]  (Range, Intensity, Mask, BeamAngle)
│
├─ Conv_in [64 channels]
│
├─ Encoder
│  ├─ DownBlock 1: 64  → 128  (CircPad + Conv + Pool) + Skip
│  ├─ DownBlock 2: 128 → 256  (CircPad + Conv + Pool) + Skip
│  ├─ DownBlock 3: 256 → 512  (CircPad + Conv + Pool) + Skip
│  └─ DownBlock 4: 512 → 1024 (CircPad + Conv + Pool) + Skip
│
├─ Middle
│  ├─ ResBlock (CircPad + Conv)
│  ├─ Attention (Self-attention)
│  └─ ResBlock (CircPad + Conv)
│
├─ Decoder
│  ├─ UpBlock 1: 1024 + 1024 → 512  (Upsample + Concat + CircPad)
│  ├─ UpBlock 2: 512  + 512  → 256  (Upsample + Concat + CircPad)
│  ├─ UpBlock 3: 256  + 256  → 128  (Upsample + Concat + CircPad)
│  └─ UpBlock 4: 128  + 128  → 64   (Upsample + Concat + CircPad)
│
└─ Conv_out [3 channels]
   │
   Output: [B, 3, 64, 1024]  (Range, Intensity, Mask)
```

### Circular Padding Detail

```
Regular Padding (WRONG for 360° LiDAR):
┌─────────────────────┐
│ 0 0 | Data | 0 0   │  ← Artificial discontinuity at 0°/360°
└─────────────────────┘

Circular Padding (CORRECT):
┌─────────────────────┐
│ ...end | Data | start... │  ← Seamless wrap-around
└─────────────────────┘

Implementation:
left = tensor[..., -pad_width:]   # Last columns
right = tensor[..., :pad_width]   # First columns
padded = cat([left, tensor, right], dim=-1)
```

## File Dependencies

```
main.py
├─ data/loaders.py
│  ├─ data/range_projection.py
│  └─ data/sensor_profiles.py
├─ models/unet.py
├─ models/diffusion.py (optional)
└─ train/trainer.py
   ├─ train/losses.py
   └─ eval/metrics.py

modal_app.py
├─ main.py (reuses everything)
└─ modal SDK

scripts/preprocess_*.py
└─ data/range_projection.py

scripts/learn_sensor_profile.py
└─ data/sensor_profiles.py

scripts/translate_batch.py
├─ models/unet.py
└─ models/diffusion.py

scripts/eval_downstream.py
├─ models/segmentation.py
└─ data/loaders.py
```

## Key Design Decisions

### 1. Why Range View?
✅ **Pros:**
- Regular 2D grid → fast CNNs
- Matches sensor geometry
- Efficient for large scans (millions of points → 64K pixels)
- Natural for diffusion models

❌ **Cons:**
- Loses some 3D structure
- Non-uniform point density

**Decision:** Range view for translation, can unproject for 3D tasks

### 2. Why Two-Stage Training?
- **Stage A (UNet):** Fast baseline, proves concept (~8 hours)
- **Stage B (Diffusion):** High quality, better details (~16 hours)

**Rationale:** Stage A alone may be sufficient; diffusion is an upgrade, not a requirement

### 3. Why Circular Padding?
LiDAR scans 360° → column 0 is adjacent to column 1023
```
Standard Conv: treats boundaries as edges → artifacts
Circular Pad:  seamless wrap → correct geometry
```

### 4. Why Masked Losses?
Many pixels are invalid (no return) → shouldn't penalize model
```python
loss = (pred - target)**2 * mask
loss = loss.sum() / mask.sum()  # Only valid pixels
```

## Scalability & Extensions

### Easy Extensions
1. **New Sensors:** Update config, re-learn profile
2. **More Data:** Just add to preprocessing
3. **Better Models:** Larger channels, more layers
4. **New Losses:** Add to `train/losses.py`

### Challenging Extensions
1. **Temporal (Video):** Need 3D+T convolutions
2. **Multi-Modal:** LiDAR + Camera fusion
3. **Real-time:** Model compression, TensorRT

## Performance Characteristics

### Memory Usage (A100-80GB)
```
Batch Size 16, 64×1024:
- UNet:      ~30 GB
- Diffusion: ~45 GB
- With Amp:  ~20/30 GB
```

### Training Speed
```
Stage A: ~100 steps/min (A100)
Stage B: ~30 steps/min  (A100, 50 diffusion steps)
```

### Inference Speed
```
UNet:      ~50 scans/sec  (A100)
Diffusion: ~2 scans/sec   (A100, 50 steps)
           ~10 scans/sec  (10 steps, quality trade-off)
```

## Error Handling

```
Data Loading Error
└─ Check file paths in config
   └─ Verify preprocessing completed

OOM Error
└─ Reduce batch_size
   └─ Reduce proj_w (1024→512)
      └─ Enable mixed_precision

Poor Quality
└─ Check sensor profile learned
   └─ Verify circular_padding enabled
      └─ Train longer
         └─ Use diffusion upgrade

Slow Training
└─ Enable mixed_precision
   └─ Use larger GPU
      └─ Reduce num_res_blocks
```

---

This architecture is **modular**, **extensible**, and **production-ready**. Each component can be swapped or upgraded independently.

