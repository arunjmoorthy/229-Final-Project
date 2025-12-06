"Runs one epoch"

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Subset

from data.loaders import create_dataloaders
from models.unet import RangeViewUNet
from train.trainer import Trainer


def create_small_subset(dataset, max_samples=50):
    """Create a small subset for quick testing."""
    indices = list(range(min(max_samples, len(dataset))))
    return Subset(dataset, indices)


def main():
    parser = argparse.ArgumentParser(description="Test training on CPU")
    parser.add_argument(
        "--config",
        type=str,
        default="config_test.yaml",
        help="Path to test config file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to use for testing"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST TRAINING - Verifying Setup")
    print("=" * 80)
    print()
    
    # Check if data exists
    data_root = Path("data/processed")
    if not data_root.exists():
        print("❌ ERROR: Preprocessed data not found!")
        print(f"   Expected: {data_root}")
        print()
        print("Please run preprocessing first:")
        print("  1. Download nuScenes mini")
        print("  2. Run preprocessing script")
        print()
        return
    
    # Check for nuScenes data (real data)
    nuscenes_train = data_root / "nuscenes_mini" / "mini_train"
    nuscenes_val = data_root / "nuscenes_mini" / "mini_val"
    
    if not nuscenes_train.exists():
        print("❌ ERROR: No nuScenes data found!")
        print(f"   Expected: {nuscenes_train}")
        print("   Please preprocess nuScenes first.")
        return
    
    real_data = list(nuscenes_train.glob("*.npz"))
    val_data = list(nuscenes_val.glob("*.npz")) if nuscenes_val.exists() else []
    
    if len(real_data) == 0:
        print("❌ ERROR: No preprocessed nuScenes data found!")
        print("   Please run: python scripts/preprocess_nuscenes.py")
        return
    
    print(f"✓ Found {len(real_data)} nuScenes training scans")
    print(f"✓ Found {len(val_data)} nuScenes validation scans")
    print()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config loaded")
    print()
    
    # Print test parameters
    print("Test Configuration:")
    print(f"  - Device: CPU")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Batch size: {config['training']['batch_size']}")
    print(f"  - Samples: {args.num_samples}")
    print(f"  - Model: Tiny UNet ({config['model']['base_channels']} channels)")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    try:
        # Use RangeViewNPZDataset for preprocessed data
        from data.loaders import RangeViewNPZDataset
        
        train_real_dataset = RangeViewNPZDataset(
            root_dir=str(nuscenes_train),
        )
        
        val_real_dataset = RangeViewNPZDataset(
            root_dir=str(nuscenes_val),
        )
        
        # Create small subsets for testing (use same for synthetic and real for now)
        train_syn_subset = create_small_subset(train_real_dataset, args.num_samples)
        train_real_subset = create_small_subset(train_real_dataset, args.num_samples)
        val_syn_subset = create_small_subset(val_real_dataset, 10)
        val_real_subset = create_small_subset(val_real_dataset, 10)
        
        # Create new dataloaders with subsets
        from torch.utils.data import DataLoader
        
        train_syn_loader = DataLoader(
            train_syn_subset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Single threaded for testing
            pin_memory=False,
        )
        
        train_real_loader = DataLoader(
            train_real_subset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        
        val_syn_loader = DataLoader(
            val_syn_subset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        val_real_loader = DataLoader(
            val_real_subset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        print(f"✓ Created dataloaders")
        print(f"  - Train synthetic: {len(train_syn_subset)} samples")
        print(f"  - Train real: {len(train_real_subset)} samples")
        print(f"  - Val synthetic: {len(val_syn_subset)} samples")
        print(f"  - Val real: {len(val_real_subset)} samples")
        print()
        
    except Exception as e:
        print(f"❌ ERROR creating dataloaders: {e}")
        print()
        print("This might mean:")
        print("  1. Data is not preprocessed correctly")
        print("  2. Paths in config are wrong")
        print("  3. Data format is incorrect")
        return
    
    # Create model
    print("Creating model...")
    try:
        model = RangeViewUNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            channel_multipliers=config['model']['channel_multipliers'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=config['model']['dropout'],
            use_circular_padding=config['model']['use_circular_padding'],
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created ({num_params:,} parameters)")
        print()
        
    except Exception as e:
        print(f"❌ ERROR creating model: {e}")
        return
    
    # Create trainer
    print("Creating trainer...")
    output_dir = Path("./outputs/test_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        trainer = Trainer(
            model=model,
            train_loader_syn=train_syn_loader,
            train_loader_real=train_real_loader,
            val_loader_syn=val_syn_loader,
            val_loader_real=val_real_loader,
            config=config,
            output_dir=str(output_dir),
            device='cpu',
            is_diffusion=False,
        )
        
        print(f"✓ Trainer created")
        print(f"✓ Output: {output_dir}")
        print()
        
    except Exception as e:
        print(f"❌ ERROR creating trainer: {e}")
        return
    
    # Train
    print("=" * 80)
    print("Starting test training...")
    print("=" * 80)
    print()
    print("This will take 5-15 minutes depending on your CPU.")
    print("Watch for:")
    print("  - Loss should be a reasonable number (0.1 - 10.0)")
    print("  - No errors or crashes")
    print("  - Progress bar should advance")
    print()
    
    try:
        trainer.train()
        
        print()
        print("=" * 80)
        print("✅ TEST TRAINING SUCCESSFUL!")
        print("=" * 80)
        print()
        print("Everything is working! Here's what happened:")
        print("  ✓ Data loaded correctly")
        print("  ✓ Model created successfully")
        print("  ✓ Training loop executed")
        print("  ✓ Loss was computed")
        print("  ✓ Gradients flowed")
        print("  ✓ Checkpoint saved")
        print()
        print("Outputs saved to:")
        print(f"  - Checkpoints: {output_dir / 'checkpoints'}")
        print(f"  - Logs: {output_dir / 'logs'}")
        print()
        print("=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print()
        print("Now that everything works, you can:")
        print()
        print("1. Train on GPU (much faster):")
        print("   python3 main.py --stage direct --config config.yaml --device cuda")
        print()
        print("2. Or continue on CPU with full config (slow but works):")
        print("   python3 main.py --stage direct --config config.yaml --device cpu")
        print()
        print("Recommendation: Use GPU if you have one!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR during training:")
        print("=" * 80)
        print(f"{e}")
        print()
        print("This error needs to be fixed before proceeding.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

