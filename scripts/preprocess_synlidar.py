"""
Preprocess SynLiDAR dataset to range-view format.
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

from data.range_projection import RangeProjection, load_kitti_point_cloud


def preprocess_synlidar(
    data_root: str,
    output_root: str,
    splits: list = ['train', 'val'],
    config: dict = None,
):
    """
    Preprocess SynLiDAR to range-view format.
    
    Args:
        data_root: Path to SynLiDAR root
        output_root: Path to save preprocessed data
        splits: List of splits to process
        config: Configuration dict with sensor parameters
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        print("\nSynLiDAR Download Instructions:")
        print("=" * 80)
        print("1. Visit: https://github.com/xiaoaoran/SynLiDAR")
        print("2. Follow download instructions")
        print("3. Extract to:", data_root)
        print("=" * 80)
        return
    
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
    
    print(f"Preprocessing SynLiDAR from {data_root}")
    print(f"Output: {output_root}")
    print(f"Splits: {splits}")
    print(f"Projection: {config['proj_h']}x{config['proj_w']}")
    
    # Process each split
    total_scans = 0
    
    for split in splits:
        # Try different possible directory structures
        possible_paths = [
            data_root / split / "velodyne",
            data_root / "velodyne" / split,
            data_root / split,
            data_root,
        ]
        
        velodyne_path = None
        for path in possible_paths:
            if path.exists() and list(path.glob("*.bin")):
                velodyne_path = path
                break
        
        if velodyne_path is None:
            print(f"Warning: Could not find velodyne data for split {split}")
            continue
        
        # Create output directory
        output_split = output_root / "synlidar" / split
        output_split.mkdir(parents=True, exist_ok=True)
        
        # Get scan files
        scan_files = sorted(velodyne_path.glob("*.bin"))
        
        print(f"\nProcessing split {split}: {len(scan_files)} scans")
        print(f"Source: {velodyne_path}")
        
        for scan_file in tqdm(scan_files, desc=f"Split {split}"):
            try:
                # Load point cloud
                points, intensity = load_kitti_point_cloud(str(scan_file))
                
                # Project to range view
                range_data = projection.project(points, intensity)
                
                # Save
                output_file = output_split / f"{scan_file.stem}.npz"
                np.savez_compressed(output_file, **range_data)
                
                total_scans += 1
                
            except Exception as e:
                print(f"\nError processing {scan_file}: {e}")
                continue
    
    print(f"\nPreprocessing completed!")
    print(f"Total scans processed: {total_scans}")
    print(f"Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SynLiDAR dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to SynLiDAR root directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/processed",
        help="Path to save preprocessed data"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=['train', 'val'],
        help="Splits to process"
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
    
    preprocess_synlidar(
        data_root=args.data_root,
        output_root=args.output_root,
        splits=args.splits,
        config=config,
    )


if __name__ == "__main__":
    main()

