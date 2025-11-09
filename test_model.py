"""
Simple script to test a trained model and visualize results.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from models.unet import RangeViewUNet
from data.loaders import RangeViewNPZDataset


def load_model(checkpoint_path, config, device='cpu'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = RangeViewUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['best_val_loss']:.4f})")
    return model


def compute_metrics(pred, target, mask):
    """Compute reconstruction metrics."""
    # Ensure mask has the right shape [1, 1, H, W] or broadcast to [1, 3, H, W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [1, H, W] -> [1, 1, H, W]
    
    # Broadcast mask to match pred/target channels
    mask = mask.expand_as(pred)  # [1, 1, H, W] -> [1, 3, H, W]
    
    # Only compute on valid pixels
    pred_valid = pred[mask]
    target_valid = target[mask]
    
    # MSE
    mse = torch.mean((pred_valid - target_valid) ** 2).item()
    
    # PSNR (peak signal-to-noise ratio)
    max_val = target_valid.max().item() if len(target_valid) > 0 else 1.0
    psnr = 10 * np.log10(max_val ** 2 / (mse + 1e-8))
    
    # MAE
    mae = torch.mean(torch.abs(pred_valid - target_valid)).item()
    
    return {'mse': mse, 'psnr': psnr, 'mae': mae}


def visualize_sample(input_data, output_data, sample_idx, save_dir):
    """Visualize input vs output for one sample."""
    # Extract channels
    input_range = input_data[0, 0].cpu().numpy()  # [H, W]
    input_intensity = input_data[0, 1].cpu().numpy()
    input_mask = input_data[0, 2].cpu().numpy().astype(bool)
    
    output_range = output_data[0, 0].cpu().numpy()
    output_intensity = output_data[0, 1].cpu().numpy()
    output_mask = output_data[0, 2].cpu().numpy().astype(bool)
    
    # Compute difference (error map)
    range_diff = np.abs(output_range - input_range)
    intensity_diff = np.abs(output_intensity - input_intensity)
    
    # Mask invalid pixels (make them white)
    input_range_vis = input_range.copy()
    input_range_vis[~input_mask] = np.nan
    
    output_range_vis = output_range.copy()
    output_range_vis[~output_mask] = np.nan
    
    range_diff_vis = range_diff.copy()
    range_diff_vis[~input_mask] = np.nan
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Range
    axes[0, 0].imshow(input_range_vis, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Input Range', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(output_range_vis, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Output Range', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    im = axes[0, 2].imshow(range_diff_vis, cmap='hot', aspect='auto')
    axes[0, 2].set_title('Range Difference (Error)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Intensity
    input_intensity_vis = input_intensity.copy()
    input_intensity_vis[~input_mask] = np.nan
    
    output_intensity_vis = output_intensity.copy()
    output_intensity_vis[~output_mask] = np.nan
    
    intensity_diff_vis = intensity_diff.copy()
    intensity_diff_vis[~input_mask] = np.nan
    
    axes[1, 0].imshow(input_intensity_vis, cmap='gray', aspect='auto')
    axes[1, 0].set_title('Input Intensity', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(output_intensity_vis, cmap='gray', aspect='auto')
    axes[1, 1].set_title('Output Intensity', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    im = axes[1, 2].imshow(intensity_diff_vis, cmap='hot', aspect='auto')
    axes[1, 2].set_title('Intensity Difference (Error)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    # Row 3: Masks and statistics
    axes[2, 0].imshow(input_mask, cmap='gray', aspect='auto')
    axes[2, 0].set_title('Input Mask', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(output_mask, cmap='gray', aspect='auto')
    axes[2, 1].set_title('Output Mask', fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    
    # Statistics text
    axes[2, 2].axis('off')
    stats_text = f"""
    Statistics:
    
    Valid pixels: {input_mask.sum():,}
    
    Range:
      Mean error: {range_diff[input_mask].mean():.4f}
      Max error: {range_diff[input_mask].max():.4f}
    
    Intensity:
      Mean error: {intensity_diff[input_mask].mean():.4f}
      Max error: {intensity_diff[input_mask].max():.4f}
    """
    axes[2, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.suptitle(f'Sample {sample_idx}: Model Input vs Output Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f'sample_{sample_idx:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test trained model")
    parser.add_argument('--checkpoint', type=str, default='outputs/direct/checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output_dir', type=str, default='test_outputs',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Testing Trained Model")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print()
    
    # Load model
    model = load_model(args.checkpoint, config, args.device)
    print()
    
    # Load validation dataset
    val_path = Path("data/processed/nuscenes_mini/mini_val")
    print(f"Loading validation data from {val_path}...")
    val_dataset = RangeViewNPZDataset(root_dir=str(val_path))
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    print()
    
    # Test on samples
    num_samples = min(args.num_samples, len(val_dataset))
    print(f"Testing on {num_samples} samples...")
    print()
    
    all_metrics = []
    
    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}:")
        
        # Get sample
        sample = val_dataset[i]
        
        # Prepare input (add batch dimension)
        input_tensor = torch.stack([
            sample['range'],
            sample['intensity'],
            sample['mask'].float(),
            sample['beam_angle'],
        ], dim=0).unsqueeze(0).to(args.device)  # [1, 4, H, W]
        
        mask = sample['mask'].unsqueeze(0).unsqueeze(0).to(args.device)  # [1, 1, H, W]
        
        # Run model
        with torch.no_grad():
            output = model(input_tensor, mask)  # [1, 3, H, W]
        
        # Compute metrics
        target = torch.stack([
            sample['range'],
            sample['intensity'],
            sample['mask'].float(),
        ], dim=0).unsqueeze(0).to(args.device)
        
        metrics = compute_metrics(output, target, sample['mask'].to(args.device))
        all_metrics.append(metrics)
        
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  MAE: {metrics['mae']:.6f}")
        
        # Visualize
        visualize_sample(input_tensor, output, i, output_dir)
        print()
    
    # Average metrics
    print("=" * 80)
    print("Average Metrics:")
    print("=" * 80)
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    
    print(f"MSE:  {avg_mse:.6f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"MAE:  {avg_mae:.6f}")
    print()
    print(f"✓ All visualizations saved to: {output_dir}")
    print()
    print("=" * 80)
    print("WHAT THESE METRICS MEAN:")
    print("=" * 80)
    print("• MSE (Mean Squared Error): Lower is better. Measures pixel-level difference.")
    print("  - Good: < 0.01")
    print("  - Okay: 0.01 - 0.1")
    print("  - Poor: > 0.1")
    print()
    print("• PSNR (Peak Signal-to-Noise Ratio): Higher is better. Measures reconstruction quality.")
    print("  - Excellent: > 40 dB")
    print("  - Good: 30-40 dB")
    print("  - Acceptable: 20-30 dB")
    print()
    print("• MAE (Mean Absolute Error): Lower is better. Average pixel error.")
    print("  - Good: < 0.05")
    print("  - Okay: 0.05 - 0.15")
    print("  - Poor: > 0.15")
    print()


if __name__ == '__main__':
    main()

