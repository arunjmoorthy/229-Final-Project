#!/usr/bin/env python3
"""
Train Sim→Real LiDAR Translation Model

This script trains a model to translate SYNTHETIC LiDAR scans to REALISTIC LiDAR scans.

Input:  Synthetic LiDAR (from SynLiDAR) - "too clean", unrealistic
Output: Real LiDAR (from nuScenes) - realistic sensor characteristics

The goal is to make synthetic data look more realistic so it can be used for training
perception models (segmentation, detection, etc.) without domain gap.
"""

import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

from data.loaders import RangeViewNPZDataset
from models.unet import RangeViewUNet
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Sim→Real LiDAR translation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="direct",
        choices=["direct", "diffusion"],
        help="Training stage: direct (UNet) or diffusion",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Use subset of data for quick testing (None = use all)",
    )
    
    args = parser.parse_args()
    
    # Load config
    print("=" * 60)
    print("🚀 Sim→Real LiDAR Translation Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Stage:  {args.stage}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Data paths
    print("\n📂 Loading datasets...")
    
    # SYNTHETIC data (input)
    synlidar_root = Path(config['data']['output_root']) / 'synlidar'
    syn_train_path = synlidar_root / 'train'
    syn_val_path = synlidar_root / 'val'
    
    # REAL data (target)
    nuscenes_root = Path(config['data']['nuscenes_npz_root'])
    real_train_path = nuscenes_root / 'mini_train'
    real_val_path = nuscenes_root / 'mini_val'
    
    # Check paths exist
    print(f"\n✓ Checking data paths:")
    for name, path in [
        ("Synthetic Train", syn_train_path),
        ("Synthetic Val", syn_val_path),
        ("Real Train", real_train_path),
        ("Real Val", real_val_path),
    ]:
        if not path.exists():
            print(f"  ❌ {name}: {path} NOT FOUND")
            return
        else:
            num_files = len(list(path.glob("*.npz")))
            print(f"  ✓ {name}: {num_files} scans")
    
    # Load datasets
    print("\n📊 Creating datasets...")
    
    # Training data
    train_synthetic = RangeViewNPZDataset(root_dir=str(syn_train_path))
    train_real = RangeViewNPZDataset(root_dir=str(real_train_path))
    
    # Validation data
    val_synthetic = RangeViewNPZDataset(root_dir=str(syn_val_path))
    val_real = RangeViewNPZDataset(root_dir=str(real_val_path))
    
    print(f"  Training:   {len(train_synthetic)} synthetic, {len(train_real)} real scans")
    print(f"  Validation: {len(val_synthetic)} synthetic, {len(val_real)} real scans")
    
    # Optionally subsample for quick testing
    if args.num_samples is not None:
        print(f"\n⚠️  Using subset of {args.num_samples} samples for testing")
        train_synthetic = Subset(train_synthetic, range(min(args.num_samples, len(train_synthetic))))
        train_real = Subset(train_real, range(min(args.num_samples, len(train_real))))
        val_synthetic = Subset(val_synthetic, range(min(args.num_samples // 4, len(val_synthetic))))
        val_real = Subset(val_real, range(min(args.num_samples // 4, len(val_real))))
    
    # Create dataloaders
    print("\n🔄 Creating dataloaders...")
    batch_size = config['training']['batch_size']
    
    # Use num_workers=0 for CPU to avoid multiprocessing issues on Mac
    num_workers = 0 if args.device == 'cpu' else config['training']['num_workers']
    
    train_loader_syn = DataLoader(
        train_synthetic,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(args.device == 'cuda'),
        drop_last=True,
    )
    
    train_loader_real = DataLoader(
        train_real,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(args.device == 'cuda'),
        drop_last=True,
    )
    
    val_loader_syn = DataLoader(
        val_synthetic,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    
    val_loader_real = DataLoader(
        val_real,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    
    print(f"  Batch size: {batch_size}")
    print(f"  Workers:    {num_workers}")
    print(f"  Train batches: {len(train_loader_syn)} synthetic, {len(train_loader_real)} real")
    print(f"  Val batches:   {len(val_loader_syn)} synthetic, {len(val_loader_real)} real")
    
    # Create model
    print("\n🏗️  Building model...")
    model_cfg = config['model']
    
    if args.stage == "direct":
        model = RangeViewUNet(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            base_channels=model_cfg['base_channels'],
            channel_multipliers=model_cfg['channel_multipliers'],
            num_res_blocks=model_cfg['num_res_blocks'],
            attention_resolutions=model_cfg['attention_resolutions'],
            dropout=model_cfg['dropout'],
        )
        print(f"  Model: RangeViewUNet (direct translation)")
    else:
        raise NotImplementedError("Diffusion stage not yet implemented")
    
    model = model.to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create trainer
    print("\n🎓 Setting up trainer...")
    
    output_dir = Path("outputs") / "sim2real" / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader_syn=train_loader_syn,
        train_loader_real=train_loader_real,
        val_loader_syn=val_loader_syn,
        val_loader_real=val_loader_real,
        config=config,
        output_dir=str(output_dir),
        device=args.device,
        is_diffusion=(args.stage == "diffusion"),
    )
    
    print(f"  Optimizer:  AdamW")
    print(f"  LR:         {config['training']['learning_rate']}")
    print(f"  Epochs:     {config['training']['num_epochs']}")
    print(f"  Output:     {output_dir}")
    
    # Start training
    print("\n" + "=" * 60)
    print("🏃 Starting training...")
    print("=" * 60)
    print("\n💡 WHAT THIS DOES:")
    print("  • Input:  SYNTHETIC LiDAR (unrealistic, too clean)")
    print("  • Output: REALISTIC LiDAR (with real sensor noise/dropout)")
    print("  • Goal:   Make synthetic data usable for training perception")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"Outputs saved to: {output_dir}")
    print(f"Checkpoints:      {output_dir / 'checkpoints'}")
    print(f"Logs:             {output_dir / 'logs'}")
    print("\n💡 Next steps:")
    print("  1. Run inference: python test_model.py --checkpoint <path>")
    print("  2. Visualize results")
    print("  3. Evaluate on downstream task (segmentation)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

