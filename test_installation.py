"""
Test script to verify installation and basic functionality.
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("Testing LiDAR Sim2Real Installation")
print("=" * 80)

# Test 1: Check dependencies
print("\n1. Checking dependencies...")
try:
    import yaml
    import matplotlib
    import scipy
    import sklearn
    print("   ✓ All required packages installed")
except ImportError as e:
    print(f"   ✗ Missing package: {e}")
    sys.exit(1)

# Test 2: Check CUDA
print("\n2. Checking CUDA...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
else:
    print("   ⚠ CUDA not available (CPU only)")

# Test 3: Test range projection
print("\n3. Testing range projection...")
try:
    from data.range_projection import RangeProjection
    
    # Create fake point cloud
    points = np.random.randn(10000, 3) * 10
    intensity = np.random.rand(10000)
    
    projection = RangeProjection(
        fov_up=3.0,
        fov_down=-25.0,
        proj_h=64,
        proj_w=1024,
    )
    
    result = projection.project(points, intensity)
    
    assert result['range'].shape == (64, 1024)
    assert result['intensity'].shape == (64, 1024)
    assert result['mask'].shape == (64, 1024)
    
    print("   ✓ Range projection works")
except Exception as e:
    print(f"   ✗ Range projection failed: {e}")
    sys.exit(1)

# Test 4: Test UNet model
print("\n4. Testing UNet model...")
try:
    from models.unet import RangeViewUNet
    
    model = RangeViewUNet(
        in_channels=4,
        out_channels=3,
        base_channels=32,
        channel_multipliers=[1, 2, 4],
        use_circular_padding=True,
    )
    
    # Test forward pass
    x = torch.randn(2, 4, 64, 256)
    mask = torch.ones(2, 1, 64, 256)
    
    with torch.no_grad():
        y = model(x, mask)
    
    assert y.shape == (2, 3, 64, 256)
    
    print(f"   ✓ UNet model works ({sum(p.numel() for p in model.parameters()):,} parameters)")
except Exception as e:
    print(f"   ✗ UNet model failed: {e}")
    sys.exit(1)

# Test 5: Test diffusion model
print("\n5. Testing diffusion model...")
try:
    from models.unet import RangeViewUNet
    from models.diffusion import DiffusionModel
    
    unet = RangeViewUNet(
        in_channels=4 + 3 + 1,
        out_channels=3,
        base_channels=16,
        channel_multipliers=[1, 2],
        use_circular_padding=True,
    )
    
    diffusion = DiffusionModel(unet, timesteps=100)
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 128)
    condition = torch.randn(2, 4, 64, 128)
    mask = torch.ones(2, 1, 64, 128)
    
    with torch.no_grad():
        noise_pred, noise_gt = diffusion(x, condition, mask)
    
    assert noise_pred.shape == noise_gt.shape
    
    print("   ✓ Diffusion model works")
except Exception as e:
    print(f"   ✗ Diffusion model failed: {e}")
    sys.exit(1)

# Test 6: Test sensor profile
print("\n6. Testing sensor profile...")
try:
    from data.sensor_profiles import SensorProfile
    
    profile = SensorProfile()
    
    # Create fake data
    range_views = [np.random.rand(64, 1024) for _ in range(10)]
    intensity_views = [np.random.rand(64, 1024) for _ in range(10)]
    mask_views = [np.random.rand(64, 1024) > 0.1 for _ in range(10)]
    
    profile.learn_from_scans(range_views, intensity_views, mask_views, num_range_bins=10)
    
    assert profile.dropout_vs_range is not None
    assert profile.intensity_vs_range is not None
    
    print("   ✓ Sensor profile works")
except Exception as e:
    print(f"   ✗ Sensor profile failed: {e}")
    sys.exit(1)

# Test 7: Test metrics
print("\n7. Testing evaluation metrics...")
try:
    from eval.metrics import compute_frid, compute_mmd
    
    features1 = np.random.randn(100, 512)
    features2 = np.random.randn(100, 512)
    
    frid = compute_frid(features1, features2)
    mmd = compute_mmd(features1, features2)
    
    assert isinstance(frid, float)
    assert isinstance(mmd, float)
    
    print(f"   ✓ Metrics work (FRID: {frid:.2f}, MMD: {mmd:.4f})")
except Exception as e:
    print(f"   ✗ Metrics failed: {e}")
    sys.exit(1)

# Test 8: Test losses
print("\n8. Testing loss functions...")
try:
    from train.losses import MaskedL1Loss, CombinedLoss
    
    l1_loss = MaskedL1Loss()
    combined_loss = CombinedLoss()
    
    pred = torch.randn(2, 3, 64, 128)
    target = torch.randn(2, 3, 64, 128)
    mask = torch.ones(2, 1, 64, 128)
    
    loss1 = l1_loss(pred, target, mask)
    loss2 = combined_loss(pred, target, mask)
    
    assert loss1.item() >= 0
    assert loss2.item() >= 0
    
    print("   ✓ Loss functions work")
except Exception as e:
    print(f"   ✗ Loss functions failed: {e}")
    sys.exit(1)

# Test 9: Test augmentation
print("\n9. Testing augmentation...")
try:
    from augment.calibration import StandardAugmentation
    from data.sensor_profiles import SensorProfile
    
    aug = StandardAugmentation()
    
    sample = {
        'range': torch.randn(64, 128),
        'intensity': torch.randn(64, 128),
        'mask': torch.ones(64, 128, dtype=torch.bool),
        'beam_angle': torch.linspace(0, 1, 64).unsqueeze(1).repeat(1, 128),
    }
    
    augmented = aug(sample)
    
    assert augmented['range'].shape == sample['range'].shape
    
    print("   ✓ Augmentation works")
except Exception as e:
    print(f"   ✗ Augmentation failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print("\nYour installation is ready. Next steps:")
print("1. Download datasets (see QUICKSTART.md)")
print("2. Preprocess data: python scripts/preprocess_semantickitti.py")
print("3. Learn sensor profile: python scripts/learn_sensor_profile.py")
print("4. Train model: python main.py --stage direct")
print("=" * 80)

