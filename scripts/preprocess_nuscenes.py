import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.range_projection import RangeProjection


def has_lidarseg(dataroot: Path) -> bool:
    return (dataroot / "lidarseg").exists()


def preprocess_split(
    dataroot: Path,
    version: str,
    split_name: str,
    output_dir: Path,
    proj_h: int,
    proj_w: int,
    fov_up: float,
    fov_down: float,
    max_range: float,
    min_range: float,
):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)

    # Scenes for split
    split_scenes = create_splits_scenes()[split_name]
    scene_tokens = [s["token"] for s in nusc.scene if s["name"] in split_scenes]

    # Prepare lidarseg if present
    lidarseg_available = has_lidarseg(dataroot)
    if lidarseg_available:
        # Lidarseg files are in dataroot/lidarseg/version/
        lidarseg_root = dataroot / "lidarseg" / version
    else:
        lidarseg_root = None

    projection = RangeProjection(
        fov_up=fov_up,
        fov_down=fov_down,
        proj_h=proj_h,
        proj_w=proj_w,
        max_range=max_range,
        min_range=min_range,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    num_saved = 0

    for scene_token in scene_tokens:
        scene = nusc.get("scene", scene_token)
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            # Get LIDAR_TOP data
            lidar_sd_token = sample["data"]["LIDAR_TOP"]
            sd_rec = nusc.get("sample_data", lidar_sd_token)
            lidar_filepath = dataroot / sd_rec["filename"]  # samples/LIDAR_TOP/....
            if not lidar_filepath.exists():
                sample_token = sample["next"]
                continue

            # Load points
            pc = LidarPointCloud.from_file(str(lidar_filepath))
            # pc.points: shape (5, N): x, y, z, intensity, ring_index
            points = pc.points[:3, :].T.astype(np.float32)
            intensity = pc.points[3, :].astype(np.float32)
            rings = pc.points[4, :].astype(np.int32) if pc.points.shape[0] > 4 else None

            # Load labels if available
            labels = None
            if lidarseg_root is not None:
                try:
                    # Lidarseg labels are stored as <sample_data_token>_lidarseg.bin
                    label_file = lidarseg_root / f"{lidar_sd_token}_lidarseg.bin"
                    if label_file.exists():
                        labels = np.fromfile(str(label_file), dtype=np.uint8).astype(np.int32)
                except Exception:
                    labels = None

            # Project
            proj = projection.project(
                points=points,
                intensity=intensity,
                labels=labels,
                ring_indices=rings,
            )

            # Save
            out_name = f"{sd_rec['token']}.npz"
            np.savez_compressed(output_dir / out_name, **proj)
            num_saved += 1

            sample_token = sample["next"]

    print(f"Saved {num_saved} NPZ files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess nuScenes to range-view NPZs")
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="nuScenes root (contains samples/, sweeps/, v1.0-*/)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="Dataset version: v1.0-mini or v1.0-trainval",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/processed/nuscenes",
        help="Directory to write NPZs",
    )
    parser.add_argument(
        "--proj_h", type=int, default=32, help="Range view height (rings)"
    )
    parser.add_argument(
        "--proj_w", type=int, default=1024, help="Range view width (azimuth)"
    )
    parser.add_argument("--fov_up", type=float, default=10.67)
    parser.add_argument("--fov_down", type=float, default=-30.67)
    parser.add_argument("--max_range", type=float, default=80.0)
    parser.add_argument("--min_range", type=float, default=0.5)
    parser.add_argument(
        "--mini_splits",
        action="store_true",
        help="Use mini_train / mini_val when version is v1.0-mini.",
    )

    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.version == "v1.0-mini":
        # nuScenes mini has two scene splits: mini_train, mini_val
        splits = [("mini_train", out_root / "mini_train"), ("mini_val", out_root / "mini_val")]
    else:
        # Use train/val scenes as defined by devkit
        splits = [("train", out_root / "train"), ("val", out_root / "val")]

    for split_name, split_out in splits:
        print(f"\nProcessing split: {split_name} (version={args.version})")
        preprocess_split(
            dataroot=dataroot,
            version=args.version,
            split_name=split_name,
            output_dir=split_out,
            proj_h=args.proj_h,
            proj_w=args.proj_w,
            fov_up=args.fov_up,
            fov_down=args.fov_down,
            max_range=args.max_range,
            min_range=args.min_range,
        )

    print("\nnuScenes preprocessing complete.")


if __name__ == "__main__":
    main()


