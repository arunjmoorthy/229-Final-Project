import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.loaders import RangeViewNPZDataset  # type: ignore  # noqa: E402
from data.range_projection import RangeProjection, load_bin_point_cloud, load_bin_labels  # type: ignore  # noqa: E402
from eval.visualize import (  # type: ignore  # noqa: E402
    visualize_3d_point_cloud,
    visualize_range_view,
    compare_range_views,
)
from models.unet import RangeViewUNet  # type: ignore  # noqa: E402


def find_example_synlidar_scan(
    raw_root: Path, seq: str
) -> Path:
    """
    Pick one SynLiDAR raw .bin scan from a sequence.

    Tries two layouts:
      - raw_root/<seq>/velodyne/*.bin
      - raw_root/sequences/<seq>/velodyne/*.bin
    """
    for base in [raw_root / seq, raw_root / "sequences" / seq]:
        velodyne_dir = base / "velodyne"
        if velodyne_dir.exists():
            bins = sorted(velodyne_dir.glob("*.bin"))
            if bins:
                return bins[0]
    raise FileNotFoundError(
        f"No SynLiDAR velodyne .bin files found for sequence {seq} under {raw_root}"
    )


def matching_processed_synlidar_npz(
    processed_root: Path, seq: str, bin_path: Path
) -> Path:
    """
    Map a raw SynLiDAR bin path to the NPZ produced by preprocess_synlidar.py.

    preprocess_synlidar saves files as:
        <output_root>/train/<seq>_<scan_stem>.npz
    """
    stem = bin_path.stem
    candidate = processed_root / "train" / f"{seq}_{stem}.npz"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected processed NPZ not found: {candidate}\n"
            "Did you run scripts/preprocess_synlidar.py with the same config?"
        )
    return candidate


def load_real_nuscenes_example(nuscenes_root: Path) -> dict:
    """
    Load a single real nuScenes range-view NPZ using RangeViewNPZDataset.
    """
    ds = RangeViewNPZDataset(root_dir=str(nuscenes_root))
    return ds[0]


def maybe_load_translated_from_checkpoint(
    checkpoint_path: Path,
    syn_npz: dict,
    device: str = "cpu",
) -> dict:
    """
    If a checkpoint is provided, run the trained UNet on the synthetic scan
    and return a translated range/intensity/mask dict.
    """
    print(f"\n[Stage 4] Loading translator from checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    in_channels = model_cfg.get("in_channels", 4)
    out_channels = model_cfg.get("out_channels", 3)
    base_channels = model_cfg.get("base_channels", 64)
    channel_multipliers = model_cfg.get("channel_multipliers", [1, 2, 4, 8])
    num_res_blocks = model_cfg.get("num_res_blocks", 2)
    attention_resolutions = model_cfg.get("attention_resolutions", [16, 8])
    dropout = 0.0  # no dropout at inference

    model = RangeViewUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        use_circular_padding=model_cfg.get("use_circular_padding", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build model input: [range, intensity, mask, beam_angle]
    range_t = torch.from_numpy(syn_npz["range"]).to(device)
    intensity_t = torch.from_numpy(syn_npz["intensity"]).to(device)
    mask_t = torch.from_numpy(syn_npz["mask"]).to(device)
    beam_angle_t = torch.from_numpy(syn_npz["beam_angle"]).to(device)

    x = torch.stack(
        [range_t, intensity_t, mask_t.float(), beam_angle_t], dim=0
    ).unsqueeze(0)  # [1, 4, H, W]
    mask_input = mask_t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    with torch.no_grad():
        pred = model(x, mask_input)  # [1, 3, H, W]

    pred_np = pred.squeeze(0).cpu().numpy()
    translated = {
        "range": pred_np[0],
        "intensity": pred_np[1] if pred_np.shape[0] > 1 else syn_npz["intensity"],
        "mask": (pred_np[2] > 0.5) if pred_np.shape[0] > 2 else syn_npz["mask"],
    }
    return translated


def main():
    parser = argparse.ArgumentParser(
        description="Visualize raw synthetic → range-view → real comparison (and optional translation)."
    )
    parser.add_argument(
        "--synlidar_raw_root",
        type=str,
        default="data/raw/synlidar",
        help="Root directory of raw SynLiDAR data.",
    )
    parser.add_argument(
        "--synlidar_processed_root",
        type=str,
        default="data/processed/synlidar",
        help="Root directory of processed SynLiDAR NPZs.",
    )
    parser.add_argument(
        "--nuscenes_npz_root",
        type=str,
        default="data/processed/nuscenes_mini/mini_train",
        help="Directory with real nuScenes-mini NPZs.",
    )
    parser.add_argument(
        "--syn_sequence",
        type=str,
        default="04",
        help="SynLiDAR sequence ID to visualize (e.g., 04, 06, 07, 09).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a trained UNet checkpoint to visualize translated output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for optional model inference.",
    )

    args = parser.parse_args()

    syn_raw_root = Path(args.synlidar_raw_root)
    syn_processed_root = Path(args.synlidar_processed_root)
    nuscenes_root = Path(args.nuscenes_npz_root)

    print("=" * 80)
    print("Sim→Real LiDAR Data Pipeline Visualization")
    print("=" * 80)
    print("\nGoal: Take SYNTHETIC LiDAR and learn to make it look like REAL sensor data.")
    print("This script shows the data at each stage so you can see what is happening.\n")

    # -------------------------------------------------------------------------
    # 1. RAW SYNTHETIC POINT CLOUD (SynLiDAR)
    # -------------------------------------------------------------------------
    print("[Stage 1] Raw SynLiDAR point cloud (XYZ + intensity)")
    bin_path = find_example_synlidar_scan(syn_raw_root, args.syn_sequence)
    print(f"  Using raw scan: {bin_path}")

    xyz, intensity = load_bin_point_cloud(str(bin_path))

    print(f"  Num points: {xyz.shape[0]}")
    print("  Visualizing 3D point cloud...")
    visualize_3d_point_cloud(
        xyz,
        colors=None,
        title=f"Raw SynLiDAR Point Cloud ({args.syn_sequence}, {bin_path.stem})",
    )

    # -------------------------------------------------------------------------
    # 2. SYNTHETIC RANGE VIEW (projection of that raw scan)
    # -------------------------------------------------------------------------
    print("\n[Stage 2] Synthetic range-view (projection of the same scan)")

    # Use the same projection as preprocess_synlidar (configured inside that script).
    # We reuse its RangeProjection class but with default settings; for exact
    # match you can pass sensor config via args if needed.
    projection = RangeProjection()
    labels = None
    label_path = bin_path.parent.parent / "labels" / f"{bin_path.stem}.label"
    if label_path.exists():
        labels = load_bin_labels(str(label_path))

    proj = projection.project(xyz, intensity=intensity, labels=labels)

    visualize_range_view(
        range_img=proj["range"],
        intensity_img=proj["intensity"],
        mask=proj["mask"],
        labels=proj.get("labels", None),
        title="Synthetic Range View (from raw SynLiDAR)",
    )

    # Also try to load the cached NPZ produced by the preprocessing script and
    # confirm it looks the same.
    try:
        syn_npz_path = matching_processed_synlidar_npz(
            syn_processed_root, args.syn_sequence, bin_path
        )
        print(f"  Matching processed NPZ: {syn_npz_path}")
        syn_npz = dict(np.load(syn_npz_path))
        visualize_range_view(
            range_img=syn_npz["range"],
            intensity_img=syn_npz["intensity"],
            mask=syn_npz["mask"],
            labels=syn_npz.get("labels", None),
            title="Synthetic Range View (loaded from processed NPZ)",
        )
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        syn_npz = proj  # fall back to on-the-fly projection

    # -------------------------------------------------------------------------
    # 3. REAL RANGE VIEW (nuScenes-mini)
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Real range-view from nuScenes-mini")
    if not nuscenes_root.exists():
        print(f"  Skipping real visualization – nuScenes NPZ root not found: {nuscenes_root}")
        real_sample = None
    else:
        real_sample = load_real_nuscenes_example(nuscenes_root)
        visualize_range_view(
            range_img=real_sample["range"],
            intensity_img=real_sample["intensity"],
            mask=real_sample["mask"],
            labels=real_sample.get("labels", None),
            title="Real Range View (nuScenes-mini)",
        )

        # Side-by-side comparison synthetic vs real
        print("  Showing side-by-side synthetic vs real range views...")
        compare_range_views(
            synthetic={
                "range": syn_npz["range"],
                "intensity": syn_npz["intensity"],
                "mask": syn_npz["mask"],
            },
            real={
                "range": real_sample["range"],
                "intensity": real_sample["intensity"],
                "mask": real_sample["mask"],
            },
            translated=None,
        )

    # -------------------------------------------------------------------------
    # 4. OPTIONAL: TRANSLATED OUTPUT FROM TRAINED UNET
    # -------------------------------------------------------------------------
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"\n[Stage 4] Skipped – checkpoint not found: {checkpoint_path}")
        else:
            translated = maybe_load_translated_from_checkpoint(
                checkpoint_path=checkpoint_path,
                syn_npz=syn_npz,
                device=args.device,
            )

            print("\n[Stage 4] Comparing synthetic vs translated (and real, if available)")
            compare_range_views(
                synthetic={
                    "range": syn_npz["range"],
                    "intensity": syn_npz["intensity"],
                    "mask": syn_npz["mask"],
                },
                real=translated,
                translated=None,
            )

            if real_sample is not None:
                # Optional 3-way comparison: synthetic vs real vs translated
                compare_range_views(
                    synthetic={
                        "range": syn_npz["range"],
                        "intensity": syn_npz["intensity"],
                        "mask": syn_npz["mask"],
                    },
                    real={
                        "range": real_sample["range"],
                        "intensity": real_sample["intensity"],
                        "mask": real_sample["mask"],
                    },
                    translated=translated,
                )

    print("\nDone. You just saw:")
    print("  1) Raw synthetic 3D point cloud.")
    print("  2) Its range/intensity/mask image (what the model actually sees).")
    print("  3) A real nuScenes range-view for comparison.")
    print("  4) Optionally, the model’s translated output approximating the real scan.\n")


if __name__ == "__main__":
    main()


