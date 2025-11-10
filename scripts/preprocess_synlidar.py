#!/usr/bin/env python3
"""
Preprocess SynLiDAR dataset to range-view NPZ format.

SynLiDAR has the same format as SemanticKITTI:
- velodyne/*.bin files (N x 4: x, y, z, intensity)
- labels/*.label files (N x 1: semantic labels)

This script projects point clouds to range view and saves as NPZ files.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml


class RangeProjection:
    """Project 3D point clouds to 2D range images."""
    
    def __init__(
        self,
        proj_h: int = 64,
        proj_w: int = 1024,
        fov_up: float = 3.0,
        fov_down: float = -25.0,
        max_range: float = 80.0,
        min_range: float = 0.5,
    ):
        self.proj_h = proj_h
        self.proj_w = proj_w
        self.fov_up = fov_up * np.pi / 180.0
        self.fov_down = fov_down * np.pi / 180.0
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.max_range = max_range
        self.min_range = min_range

    def project(self, points: np.ndarray, labels: np.ndarray = None):
        """
        Project point cloud to range image.
        
        Args:
            points: (N, 4) array of [x, y, z, intensity]
            labels: (N,) array of semantic labels (optional)
            
        Returns:
            Dictionary with range, intensity, mask, beam_angle, and optionally labels
        """
        # Extract coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3]
        
        # Compute range
        depth = np.sqrt(x**2 + y**2 + z**2)
        
        # Filter by range
        valid_mask = (depth >= self.min_range) & (depth <= self.max_range)
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        depth = depth[valid_mask]
        intensity = intensity[valid_mask]
        if labels is not None:
            labels = labels[valid_mask]
        
        # Compute angles
        yaw = -np.arctan2(y, x)  # Azimuth angle
        pitch = np.arcsin(z / depth)  # Elevation angle
        
        # Project to image coordinates
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # [0, 1]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov  # [0, 1]
        
        # Scale to image size
        proj_x = np.floor(proj_x * self.proj_w).astype(np.int32)
        proj_y = np.floor(proj_y * self.proj_h).astype(np.int32)
        
        # Clamp to valid range
        proj_x = np.clip(proj_x, 0, self.proj_w - 1)
        proj_y = np.clip(proj_y, 0, self.proj_h - 1)
        
        # Create range image (initialize with zeros)
        range_image = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        intensity_image = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        mask = np.zeros((self.proj_h, self.proj_w), dtype=bool)
        beam_angle = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        
        if labels is not None:
            label_image = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)
        
        # Fill in the range image (keep closest point per pixel)
        # Sort by depth (furthest first) so closest points overwrite
        order = np.argsort(depth)[::-1]
        proj_x = proj_x[order]
        proj_y = proj_y[order]
        depth = depth[order]
        intensity = intensity[order]
        pitch = pitch[order]
        if labels is not None:
            labels = labels[order]
        
        # Fill images
        range_image[proj_y, proj_x] = depth
        intensity_image[proj_y, proj_x] = intensity
        mask[proj_y, proj_x] = True
        beam_angle[proj_y, proj_x] = pitch
        if labels is not None:
            label_image[proj_y, proj_x] = labels
        
        result = {
            'range': range_image,
            'intensity': intensity_image,
            'mask': mask,
            'beam_angle': beam_angle,
        }
        
        if labels is not None:
            result['labels'] = label_image
        
        return result


def load_synlidar_scan(bin_path: Path):
    """Load a SynLiDAR .bin file."""
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return points


def load_synlidar_labels(label_path: Path):
    """Load a SynLiDAR .label file."""
    if label_path.exists():
        labels = np.fromfile(str(label_path), dtype=np.uint32)
        # Extract semantic label (lower 16 bits)
        labels = labels & 0xFFFF
        return labels.astype(np.int32)
    return None


def preprocess_sequence(
    seq_dir: Path,
    output_dir: Path,
    projection: RangeProjection,
    max_scans: int = None,
):
    """Preprocess a single SynLiDAR sequence."""
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    
    if not velodyne_dir.exists():
        print(f"  ⚠️  Velodyne directory not found: {velodyne_dir}")
        return 0
    
    # Get all .bin files
    bin_files = sorted(velodyne_dir.glob("*.bin"))
    
    if max_scans is not None:
        bin_files = bin_files[:max_scans]
    
    print(f"  Processing {len(bin_files)} scans from {seq_dir.name}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_processed = 0
    for bin_file in tqdm(bin_files, desc=f"  Seq {seq_dir.name}"):
        # Load point cloud
        points = load_synlidar_scan(bin_file)
        
        # Load labels if available
        label_file = labels_dir / bin_file.name.replace(".bin", ".label")
        labels = load_synlidar_labels(label_file)
        
        # Project to range view
        range_data = projection.project(points, labels)
        
        # Save as NPZ
        output_file = output_dir / f"{seq_dir.name}_{bin_file.stem}.npz"
        np.savez_compressed(output_file, **range_data)
        
        num_processed += 1
    
    return num_processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess SynLiDAR to range-view NPZ")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--synlidar_root",
        type=str,
        default=None,
        help="Path to SynLiDAR raw data (overrides config)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Path to output directory (overrides config)",
    )
    parser.add_argument(
        "--train_sequences",
        type=str,
        nargs="+",
        default=["00", "01", "02"],
        help="Sequence IDs for training set",
    )
    parser.add_argument(
        "--val_sequences",
        type=str,
        nargs="+",
        default=["03"],
        help="Sequence IDs for validation set",
    )
    parser.add_argument(
        "--max_scans_per_seq",
        type=int,
        default=None,
        help="Maximum scans per sequence (None = all)",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths
    if args.synlidar_root:
        synlidar_root = Path(args.synlidar_root)
    else:
        synlidar_root = Path(config['data'].get('synlidar_root', 'data/raw/synlidar'))
    
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = Path(config['data']['output_root']) / "synlidar"
    
    # Check if synlidar_root exists
    if not synlidar_root.exists():
        print(f"❌ SynLiDAR root not found: {synlidar_root}")
        return
    
    print(f"\n{'='*60}")
    print(f"SynLiDAR Preprocessing")
    print(f"{'='*60}")
    print(f"Input:  {synlidar_root}")
    print(f"Output: {output_root}")
    print(f"Train sequences: {args.train_sequences}")
    print(f"Val sequences:   {args.val_sequences}")
    if args.max_scans_per_seq:
        print(f"Max scans/seq:   {args.max_scans_per_seq}")
    print(f"{'='*60}\n")
    
    # Create projection
    sensor_cfg = config['sensor']
    projection = RangeProjection(
        proj_h=sensor_cfg['n_rings'],
        proj_w=sensor_cfg['n_azimuth'],
        fov_up=sensor_cfg['fov_up'],
        fov_down=sensor_cfg['fov_down'],
        max_range=sensor_cfg['max_range'],
        min_range=sensor_cfg['min_range'],
    )
    
    # Process training sequences
    print("\n📦 Processing TRAINING sequences...")
    train_output = output_root / "train"
    total_train = 0
    
    for seq_id in args.train_sequences:
        # Try in root directory first, then in sequences/ subdirectory
        seq_dir = synlidar_root / seq_id
        if not seq_dir.exists():
            seq_dir = synlidar_root / "sequences" / seq_id
        
        if seq_dir.exists():
            num = preprocess_sequence(
                seq_dir,
                train_output,
                projection,
                args.max_scans_per_seq,
            )
            total_train += num
        else:
            print(f"  ⚠️  Sequence {seq_id} not found")
    
    # Process validation sequences
    print("\n📦 Processing VALIDATION sequences...")
    val_output = output_root / "val"
    total_val = 0
    
    for seq_id in args.val_sequences:
        # Try in root directory first, then in sequences/ subdirectory
        seq_dir = synlidar_root / seq_id
        if not seq_dir.exists():
            seq_dir = synlidar_root / "sequences" / seq_id
        
        if seq_dir.exists():
            num = preprocess_sequence(
                seq_dir,
                val_output,
                projection,
                args.max_scans_per_seq,
            )
            total_val += num
        else:
            print(f"  ⚠️  Sequence {seq_id} not found")
    
    print(f"\n{'='*60}")
    print(f"✅ Preprocessing complete!")
    print(f"{'='*60}")
    print(f"Training scans:   {total_train}")
    print(f"Validation scans: {total_val}")
    print(f"Total:            {total_train + total_val}")
    print(f"Output location:  {output_root}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
