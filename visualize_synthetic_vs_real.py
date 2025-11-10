#!/usr/bin/env python3
"""
Visualize the difference between SYNTHETIC and REAL LiDAR data.

This shows WHY we need Sim→Real translation:
- Synthetic: too clean, unrealistic intensity, no sensor noise
- Real: realistic sensor dropout, noise, intensity falloff
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from data.loaders import RangeViewNPZDataset


def visualize_comparison(syn_sample, real_sample, save_path=None):
    """Create side-by-side comparison of synthetic vs real LiDAR."""
    
    # Extract data
    syn_range = syn_sample['range'].numpy() if isinstance(syn_sample['range'], torch.Tensor) else syn_sample['range']
    syn_intensity = syn_sample['intensity'].numpy() if isinstance(syn_sample['intensity'], torch.Tensor) else syn_sample['intensity']
    syn_mask = syn_sample['mask'].numpy() if isinstance(syn_sample['mask'], torch.Tensor) else syn_sample['mask']
    
    real_range = real_sample['range'].numpy() if isinstance(real_sample['range'], torch.Tensor) else real_sample['range']
    real_intensity = real_sample['intensity'].numpy() if isinstance(real_sample['intensity'], torch.Tensor) else real_sample['intensity']
    real_mask = real_sample['mask'].numpy() if isinstance(real_sample['mask'], torch.Tensor) else real_sample['mask']
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('SYNTHETIC vs REAL LiDAR Comparison\n(Why We Need Sim→Real Translation)', 
                 fontsize=16, fontweight='bold')
    
    # Column titles
    axes[0, 0].set_title('SYNTHETIC (Input)\n❌ Unrealistic', fontsize=14, fontweight='bold', color='red')
    axes[0, 1].set_title('REAL (Target)\n✓ Realistic', fontsize=14, fontweight='bold', color='green')
    
    # Row 1: Range images
    im1 = axes[0, 0].imshow(syn_range, cmap='viridis', aspect='auto', vmin=0, vmax=80)
    axes[0, 0].set_ylabel('Range (m)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 1].imshow(real_range, cmap='viridis', aspect='auto', vmin=0, vmax=80)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Row 2: Intensity images
    im3 = axes[1, 0].imshow(syn_intensity, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_ylabel('Intensity', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im4 = axes[1, 1].imshow(real_intensity, cmap='gray', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Row 3: Valid mask (shows dropouts)
    im5 = axes[2, 0].imshow(syn_mask, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[2, 0].set_ylabel('Valid Returns\n(Green = detected)', fontsize=12)
    axes[2, 0].set_xlabel('Azimuth (horizontal scan)', fontsize=10)
    plt.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    im6 = axes[2, 1].imshow(real_mask, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[2, 1].set_xlabel('Azimuth (horizontal scan)', fontsize=10)
    plt.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)
    
    # Compute statistics
    syn_dropout_rate = 1 - syn_mask.sum() / syn_mask.size
    real_dropout_rate = 1 - real_mask.sum() / real_mask.size
    
    # Add text annotations
    textstr_syn = f'Dropout: {syn_dropout_rate*100:.1f}%\nMean range: {syn_range[syn_mask].mean():.1f}m'
    textstr_real = f'Dropout: {real_dropout_rate*100:.1f}%\nMean range: {real_range[real_mask].mean():.1f}m'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[2, 0].text(0.02, 0.98, textstr_syn, transform=axes[2, 0].transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)
    axes[2, 1].text(0.02, 0.98, textstr_real, transform=axes[2, 1].transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic vs real LiDAR")
    parser.add_argument(
        "--synthetic_root",
        type=str,
        default="data/processed/synlidar/train",
        help="Path to synthetic data",
    )
    parser.add_argument(
        "--real_root",
        type=str,
        default="data/processed/nuscenes_mini/mini_train",
        help="Path to real data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="SYNTHETIC_VS_REAL_COMPARISON.png",
        help="Output file path",
    )
    parser.add_argument(
        "--syn_idx",
        type=int,
        default=0,
        help="Index of synthetic sample to visualize",
    )
    parser.add_argument(
        "--real_idx",
        type=int,
        default=0,
        help="Index of real sample to visualize",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 Visualizing Synthetic vs Real LiDAR")
    print("=" * 60)
    
    # Load datasets
    print(f"\nLoading synthetic data from: {args.synthetic_root}")
    syn_dataset = RangeViewNPZDataset(root_dir=args.synthetic_root)
    print(f"  Found {len(syn_dataset)} synthetic scans")
    
    print(f"\nLoading real data from: {args.real_root}")
    real_dataset = RangeViewNPZDataset(root_dir=args.real_root)
    print(f"  Found {len(real_dataset)} real scans")
    
    # Get samples
    syn_sample = syn_dataset[args.syn_idx]
    real_sample = real_dataset[args.real_idx]
    
    print(f"\nComparing:")
    print(f"  Synthetic sample {args.syn_idx}")
    print(f"  Real sample {args.real_idx}")
    
    # Create visualization
    print(f"\nGenerating comparison...")
    visualize_comparison(syn_sample, real_sample, save_path=args.output)
    
    print("\n" + "=" * 60)
    print("💡 KEY DIFFERENCES:")
    print("=" * 60)
    print("1. RANGE:")
    print("   • Synthetic: smoother, fewer dropouts")
    print("   • Real: more gaps, distance-dependent dropouts")
    print()
    print("2. INTENSITY:")
    print("   • Synthetic: uniform, unrealistic")
    print("   • Real: realistic falloff, surface-dependent")
    print()
    print("3. VALID RETURNS (bottom row):")
    print("   • Synthetic: almost all green (very few missing)")
    print("   • Real: more red areas (sensor dropouts, occlusions)")
    print()
    print("=" * 60)
    print("🎯 PROJECT GOAL:")
    print("=" * 60)
    print("Train a model to TRANSFORM synthetic → real")
    print("So that synthetic data can be used for training")
    print("perception models without domain gap!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

