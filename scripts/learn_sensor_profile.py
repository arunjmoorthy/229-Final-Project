"""
Learn sensor profile (dropout, intensity patterns) from real data.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sensor_profiles import SensorProfile


def learn_sensor_profile(
    data_root: str,
    output_path: str,
    num_samples: int = 500,
    sequences: list = None,
):
    """
    Learn sensor profile from SemanticKITTI data.
    
    Args:
        data_root: Path to preprocessed SemanticKITTI
        output_path: Path to save sensor profile JSON
        num_samples: Number of scans to use
        sequences: Sequences to use (default: train sequences)
    """
    data_root = Path(data_root)
    
    if sequences is None:
        sequences = [f"{i:02d}" for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]
    
    print(f"Learning sensor profile from {data_root}")
    print(f"Sequences: {sequences}")
    print(f"Samples: {num_samples}")
    
    # Collect range views
    range_views = []
    intensity_views = []
    mask_views = []
    
    samples_per_seq = max(1, num_samples // len(sequences))
    
    for seq in sequences:
        seq_path = data_root / "semantickitti" / seq
        
        if not seq_path.exists():
            print(f"Warning: Sequence {seq} not found at {seq_path}")
            continue
        
        # Get files
        files = sorted(seq_path.glob("*.npz"))
        
        # Sample uniformly
        step = max(1, len(files) // samples_per_seq)
        sampled_files = files[::step][:samples_per_seq]
        
        print(f"Sequence {seq}: Using {len(sampled_files)} scans")
        
        for file in tqdm(sampled_files, desc=f"Loading {seq}", leave=False):
            data = np.load(file)
            range_views.append(data['range'])
            intensity_views.append(data['intensity'])
            mask_views.append(data['mask'])
    
    print(f"\nLoaded {len(range_views)} scans")
    
    # Learn profile
    profile = SensorProfile()
    profile.learn_from_scans(
        range_views=range_views,
        intensity_views=intensity_views,
        mask_views=mask_views,
        num_range_bins=50,
        num_incidence_bins=20,
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.save(str(output_path))
    
    # Plot
    plot_path = output_path.parent / f"{output_path.stem}_plot.png"
    profile.plot(str(plot_path))
    
    print(f"\nSensor profile saved to: {output_path}")
    print(f"Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Learn sensor profile from real data")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to preprocessed data root"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sensor_profile.json",
        help="Path to save sensor profile"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of scans to use"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequences to use (default: train sequences)"
    )
    
    args = parser.parse_args()
    
    learn_sensor_profile(
        data_root=args.data_root,
        output_path=args.output,
        num_samples=args.num_samples,
        sequences=args.sequences,
    )


if __name__ == "__main__":
    main()

