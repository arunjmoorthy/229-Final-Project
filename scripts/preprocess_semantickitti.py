"""
Preprocess SemanticKITTI dataset to range-view format.
Downloads if needed and projects all scans to range images.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.range_projection import RangeProjection, load_kitti_point_cloud, load_kitti_labels


def download_semantickitti(data_root: str):
    """
    Instructions for downloading SemanticKITTI.
    Note: Actual download requires manual registration and agreement to terms.
    """
    print("=" * 80)
    print("SemanticKITTI Download Instructions:")
    print("=" * 80)
    print()
    print("1. Register at: http://semantic-kitti.org/")
    print("2. Download the following:")
    print("   - Velodyne point clouds (80 GB)")
    print("   - Semantic labels (200 MB)")
    print("3. Extract to:", data_root)
    print()
    print("Expected structure:")
    print(f"{data_root}/")
    print("  sequences/")
    print("    00/")
    print("      velodyne/  (*.bin files)")
    print("      labels/    (*.label files)")
    print("    01/")
    print("    ...")
    print()
    print("=" * 80)


def preprocess_semantickitti(
    data_root: str,
    output_root: str,
    sequences: list = None,
    config: dict = None,
):
    """
    Preprocess SemanticKITTI to range-view format.
    
    Args:
        data_root: Path to SemanticKITTI root
        output_root: Path to save preprocessed data
        sequences: List of sequences to process (default: all train/val)
        config: Configuration dict with sensor parameters
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        download_semantickitti(data_root)
        return
    
    # Default sequences (train + val)
    if sequences is None:
        sequences = [f"{i:02d}" for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    # Create projection
    if config is None:
        config = {
            'fov_up': 3.0,
            'fov_down': -25.0,
            'proj_h': 64,
            'proj_w': 1024,
            'max_range': 80.0,
            'min_range': 0.5,
        }
    
    projection = RangeProjection(
        fov_up=config.get('fov_up', 3.0),
        fov_down=config.get('fov_down', -25.0),
        proj_h=config.get('proj_h', 64),
        proj_w=config.get('proj_w', 1024),
        max_range=config.get('max_range', 80.0),
        min_range=config.get('min_range', 0.5),
    )
    
    print(f"Preprocessing SemanticKITTI from {data_root}")
    print(f"Output: {output_root}")
    print(f"Sequences: {sequences}")
    print(f"Projection: {config['proj_h']}x{config['proj_w']}")
    
    # Process each sequence
    total_scans = 0
    
    for seq in sequences:
        seq_path = data_root / "sequences" / seq
        
        if not seq_path.exists():
            print(f"Warning: Sequence {seq} not found at {seq_path}")
            continue
        
        # Create output directory
        output_seq = output_root / "semantickitti" / seq
        output_seq.mkdir(parents=True, exist_ok=True)
        
        # Get scan files
        velodyne_path = seq_path / "velodyne"
        label_path = seq_path / "labels"
        
        scan_files = sorted(velodyne_path.glob("*.bin"))
        
        print(f"\nProcessing sequence {seq}: {len(scan_files)} scans")
        
        for scan_file in tqdm(scan_files, desc=f"Seq {seq}"):
            # Load point cloud
            points, intensity = load_kitti_point_cloud(str(scan_file))
            
            # Load labels if available
            label_file = label_path / f"{scan_file.stem}.label"
            labels = None
            if label_file.exists():
                labels = load_kitti_labels(str(label_file))
            
            # Project to range view
            range_data = projection.project(points, intensity, labels)
            
            # Save
            output_file = output_seq / f"{scan_file.stem}.npz"
            np.savez_compressed(output_file, **range_data)
            
            total_scans += 1
    
    print(f"\nPreprocessing completed!")
    print(f"Total scans processed: {total_scans}")
    print(f"Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SemanticKITTI dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to SemanticKITTI root directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/processed",
        help="Path to save preprocessed data"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequences to process (default: train+val sequences)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = None
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get('sensor', {})
    
    preprocess_semantickitti(
        data_root=args.data_root,
        output_root=args.output_root,
        sequences=args.sequences,
        config=config,
    )


if __name__ == "__main__":
    main()

