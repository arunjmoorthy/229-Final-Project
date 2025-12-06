"""
Evaluate downstream segmentation performance with different data regimes.
Compares: Real-only, Real+Raw-Synthetic, Real+Translated-Synthetic.
"""

import sys
import argparse
import torch
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders import SynLiDARDataset, RangeViewNPZDataset
from data.range_projection import RangeProjection
from models.segmentation import RangeNetSegmentation, SegmentationTrainer
from torch.utils.data import DataLoader, ConcatDataset


def train_segmentation(
    config_path: str,
    data_regime: str = 'real',  # 'real', 'real+raw', 'real+translated'
    synthetic_dir: str = None,
    dataset: str = 'nuscenes_npz',  # 'nuscenes_npz'
    output_dir: str = './outputs/segmentation',
):
    """
    Train segmentation model with different data regimes.
    
    Args:
        config_path: Path to config file
        data_regime: Data regime ('real', 'real+raw', 'real+translated')
        synthetic_dir: Directory with synthetic or translated data
        output_dir: Output directory
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create projection
    sensor_cfg = config['sensor']
    projection = RangeProjection(
        fov_up=sensor_cfg['fov_up'],
        fov_down=sensor_cfg['fov_down'],
        proj_h=sensor_cfg['n_rings'],
        proj_w=sensor_cfg['n_azimuth'],
        max_range=sensor_cfg['max_range'],
        min_range=sensor_cfg['min_range'],
    )
    
    if dataset == 'nuscenes_npz':
        # Expect preprocessed NPZs at config['data']['nuscenes_npz_root']
        print(f"Loading real data (nuScenes NPZ)...")
        npz_root = config['data'].get('nuscenes_npz_root', './data/processed/nuscenes')
        # Use mini splits by default
        real_train = RangeViewNPZDataset(root_dir=f"{npz_root}/mini_train")
        real_val = RangeViewNPZDataset(root_dir=f"{npz_root}/mini_val")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"Real train: {len(real_train)} scans")
    print(f"Real val: {len(real_val)} scans")
    
    # Prepare training dataset based on regime
    if data_regime == 'real':
        train_dataset = real_train
        print("Using: Real data only")
        
    elif data_regime == 'real+raw':
        if synthetic_dir is None:
            synthetic_dir = config['data']['synlidar_root']
        
        print(f"Loading raw synthetic data from {synthetic_dir}...")
        syn_train = SynLiDARDataset(
            root=synthetic_dir,
            split='train',
            projection=projection,
        )
        
        # Limit synthetic to match real data size (for fair comparison)
        syn_train = torch.utils.data.Subset(syn_train, range(min(len(syn_train), len(real_train))))
        
        train_dataset = ConcatDataset([real_train, syn_train])
        print(f"Using: Real ({len(real_train)}) + Raw Synthetic ({len(syn_train)})")
        
    elif data_regime == 'real+translated':
        if synthetic_dir is None:
            raise ValueError("Must provide synthetic_dir for translated data")
        
        print(f"Loading translated synthetic data from {synthetic_dir}...")
        syn_train = SynLiDARDataset(
            root=synthetic_dir,
            split='train',
            projection=projection,
        )
        
        syn_train = torch.utils.data.Subset(syn_train, range(min(len(syn_train), len(real_train))))
        
        train_dataset = ConcatDataset([real_train, syn_train])
        print(f"Using: Real ({len(real_train)}) + Translated ({len(syn_train)})")
        
    else:
        raise ValueError(f"Unknown data regime: {data_regime}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['segmentation']['batch_size'] if 'segmentation' in config else 8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        real_val,
        batch_size=config['segmentation']['batch_size'] if 'segmentation' in config else 8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    print("Creating segmentation model...")
    model = RangeNetSegmentation(
        in_channels=3,
        num_classes=config['segmentation']['num_classes'] if 'segmentation' in config else 19,
        base_channels=32,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config['segmentation']['num_classes'] if 'segmentation' in config else 19,
        learning_rate=config['segmentation']['learning_rate'] if 'segmentation' in config else 0.001,
        device=device,
    )
    
    # Train
    print(f"\nTraining segmentation model - Regime: {data_regime}")
    num_epochs = config['segmentation']['num_epochs'] if 'segmentation' in config else 50
    best_miou = trainer.train(num_epochs=num_epochs)
    
    print(f"\n{'='*80}")
    print(f"Data Regime: {data_regime}")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"{'='*80}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"results_{data_regime.replace('+', '_')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Data Regime: {data_regime}\n")
        f.write(f"Best mIoU: {best_miou:.4f}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Validation samples: {len(real_val)}\n")
    
    print(f"Results saved to: {results_file}")
    
    return best_miou


def main():
    parser = argparse.ArgumentParser(description="Evaluate downstream segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--regime",
        type=str,
        choices=['real', 'real+raw', 'real+translated'],
        default='real',
        help="Data regime to use"
    )
    parser.add_argument(
        "--synthetic_dir",
        type=str,
        default=None,
        help="Directory with synthetic or translated data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['nuscenes_npz'],
        default='nuscenes_npz',
        help="Real dataset source"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/segmentation",
        help="Output directory"
    )
    parser.add_argument(
        "--all",
        action='store_true',
        help="Run all three regimes"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Run all regimes for comparison
        print("Running all data regimes for comparison...")
        
        results = {}
        
        # Real only
        print("\n" + "="*80)
        print("REGIME 1: Real Only")
        print("="*80)
        results['real'] = train_segmentation(
            config_path=args.config,
            data_regime='real',
            dataset=args.dataset,
            output_dir=args.output_dir,
        )
        
        # Real + Raw Synthetic
        print("\n" + "="*80)
        print("REGIME 2: Real + Raw Synthetic")
        print("="*80)
        results['real+raw'] = train_segmentation(
            config_path=args.config,
            data_regime='real+raw',
            dataset=args.dataset,
            synthetic_dir=args.synthetic_dir,
            output_dir=args.output_dir,
        )
        
        # Real + Translated Synthetic
        print("\n" + "="*80)
        print("REGIME 3: Real + Translated Synthetic")
        print("="*80)
        results['real+translated'] = train_segmentation(
            config_path=args.config,
            data_regime='real+translated',
            dataset=args.dataset,
            synthetic_dir=args.synthetic_dir,
            output_dir=args.output_dir,
        )
        
        # Print comparison
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        for regime, miou in results.items():
            print(f"{regime:20s}: mIoU = {miou:.4f}")
        print("="*80)
        
    else:
        # Run single regime
        train_segmentation(
            config_path=args.config,
            data_regime=args.regime,
            dataset=args.dataset,
            synthetic_dir=args.synthetic_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

