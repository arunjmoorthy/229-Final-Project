"""
Main training script for LiDAR Sim2Real translation.
Can run locally or on Modal.
"""

import argparse
import yaml
import torch
from pathlib import Path

from data.loaders import create_dataloaders
from models.unet import RangeViewUNet
from models.diffusion import DiffusionModel
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train LiDAR Sim2Real Translator")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["direct", "diffusion"],
        default="direct",
        help="Training stage: 'direct' for UNet, 'diffusion' for diffusion model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print configuration
    print("=" * 80)
    print(f"Training Stage: {args.stage}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        config=config,
        synthetic_root=config['data'].get('synlidar_root'),
        real_root=config['data'].get('nuscenes_npz_root'),
        num_workers=config['training'].get('num_workers', 4),
    )
    
    def _describe_loader(name: str, pretty: str):
        loader = dataloaders.get(name)
        if loader is None:
            print(f"{pretty}: unavailable (check config paths)")
        else:
            try:
                length = len(loader.dataset)
            except AttributeError:
                length = len(loader)
            print(f"{pretty}: {length} samples")

    _describe_loader('train_syn', 'Train synthetic')
    _describe_loader('train_real', 'Train real')
    _describe_loader('val_syn', 'Val synthetic')
    _describe_loader('val_real', 'Val real')
    
    # Create model
    print("\nCreating model...")
    
    if args.stage == "direct":
        # Direct UNet translator
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
        output_dir = Path(args.output_dir) / "direct"
        
    elif args.stage == "diffusion":
        # Diffusion model
        unet = RangeViewUNet(
            in_channels=config['model']['in_channels'] + config['model']['out_channels'] + 1,  # +1 for time
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
        output_dir = Path(args.output_dir) / "diffusion"
        
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_loader_syn=dataloaders['train_syn'],
        train_loader_real=dataloaders['train_real'],
        val_loader_syn=dataloaders['val_syn'],
        val_loader_real=dataloaders['val_real'],
        config=config,
        output_dir=str(output_dir),
        device=args.device,
        is_diffusion=is_diffusion,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model saved at: {output_dir / 'checkpoints' / 'best.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

