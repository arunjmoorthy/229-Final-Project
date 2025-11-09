"""
Training script specifically for nuScenes preprocessed NPZ data.
Use this instead of main.py when training on nuScenes mini.
"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from data.loaders import RangeViewNPZDataset
from models.unet import RangeViewUNet
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train on nuScenes mini")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--stage", type=str, default="direct", choices=["direct", "diffusion"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit training samples (for testing)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"Training on nuScenes mini - Stage: {args.stage}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Data paths
    train_path = Path("data/processed/nuscenes_mini/mini_train")
    val_path = Path("data/processed/nuscenes_mini/mini_val")
    
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Please run: python scripts/preprocess_nuscenes.py first")
        return
    
    # Create datasets
    print("Loading datasets...")
    print(f"  Train path: {train_path}")
    print(f"  Val path: {val_path}")
    
    train_dataset = RangeViewNPZDataset(root_dir=str(train_path))
    print(f"  Loaded {len(train_dataset)} training files")
    
    val_dataset = RangeViewNPZDataset(root_dir=str(val_path))
    print(f"  Loaded {len(val_dataset)} validation files")
    
    # Limit samples if requested (for testing)
    if args.num_samples:
        print(f"  Limiting to {args.num_samples} training samples")
        train_dataset = Subset(train_dataset, range(min(args.num_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(args.num_samples // 4, len(val_dataset))))
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print()
    
    # Create dataloaders (use same dataset for both syn and real since we only have real data)
    print("Creating dataloaders...")
    batch_size = config['training']['batch_size']
    print(f"  Batch size: {batch_size}")
    
    train_loader_syn = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if args.device == 'cpu' else config['training']['num_workers'],
        pin_memory=(args.device == 'cuda'),
    )
    
    train_loader_real = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0 if args.device == 'cpu' else config['training']['num_workers'],
        pin_memory=(args.device == 'cuda'),
    )
    
    val_loader_syn = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == 'cuda'),
    )
    
    val_loader_real = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == 'cuda'),
    )
    
    print(f"✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader_syn)}")
    print(f"  Val batches: {len(val_loader_syn)}")
    print()
    
    # Create model
    print("Creating model...")
    if args.stage == "direct":
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
        is_diffusion = False
        
    elif args.stage == "diffusion":
        from models.diffusion import DiffusionModel
        
        unet = RangeViewUNet(
            in_channels=config['model']['in_channels'] + config['model']['out_channels'] + 1,
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            channel_multipliers=config['model']['channel_multipliers'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=config['model']['dropout'],
            use_circular_padding=config['model']['use_circular_padding'],
        )
        
        model = DiffusionModel(
            denoise_model=unet,
            timesteps=config['diffusion']['timesteps'],
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            cfg_dropout=0.1,
        )
        is_diffusion = True
        
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
    
    print(f"✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader_syn=train_loader_syn,
        train_loader_real=train_loader_real,
        val_loader_syn=val_loader_syn,
        val_loader_real=val_loader_real,
        config=config,
        output_dir=str(output_dir),
        device=args.device,
        is_diffusion=is_diffusion,
    )
    
    print(f"✓ Trainer created")
    print(f"✓ Output: {output_dir}")
    print()
    
    # Train
    print("=" * 80)
    print("Starting training...")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Device: {args.device}")
    print("=" * 80)
    print()
    
    import sys
    sys.stdout.flush()  # Ensure output is visible
    
    trainer.train()
    
    print()
    print("=" * 80)
    print("Training completed!")
    print(f"Best model: {output_dir / 'checkpoints' / 'best.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

