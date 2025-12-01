#!/usr/bin/env python3
"""
Create side-by-side figures for synthetic vs translated vs real range views.

Example:
    python3 scripts/visualize_translation_examples.py \
        --synthetic_dir data/processed/synlidar/val \
        --translated_dir outputs/sim2real/direct/translated_synlidar_val \
        --real_dir data/processed/nuscenes_mini/mini_val \
        --num_examples 4 \
        --output_dir outputs/sim2real/direct/example_comparisons
"""

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loaders import RangeViewNPZDataset
from eval.visualize import compare_range_views


def tensor_to_numpy(sample, keys):
    return {k: sample[k].numpy() if hasattr(sample[k], "numpy") else sample[k] for k in keys}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize synthetic vs translated vs real range views."
    )
    parser.add_argument(
        "--synthetic_dir",
        required=True,
        help="Directory with raw synthetic NPZs.",
    )
    parser.add_argument(
        "--translated_dir",
        required=True,
        help="Directory with translated synthetic NPZs.",
    )
    parser.add_argument(
        "--real_dir",
        required=True,
        help="Directory with real NPZs (nuScenes mini).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=4,
        help="How many examples to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save comparison figures.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    syn_ds = RangeViewNPZDataset(args.synthetic_dir)
    trn_ds = RangeViewNPZDataset(args.translated_dir)
    real_ds = RangeViewNPZDataset(args.real_dir)

    n = min(len(syn_ds), len(trn_ds))
    if n == 0:
        raise ValueError("Synthetic or translated dataset is empty.")
    indices = random.sample(range(n), k=min(args.num_examples, n))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        syn_sample = syn_ds[idx]
        trn_sample = trn_ds[idx]
        real_sample = real_ds[idx % len(real_ds)]

        compare_range_views(
            synthetic=tensor_to_numpy(syn_sample, ["range", "intensity", "mask"]),
            real=tensor_to_numpy(real_sample, ["range", "intensity", "mask"]),
            translated=tensor_to_numpy(trn_sample, ["range", "intensity", "mask"]),
            save_path=out_dir / f"comparison_{idx:04d}.png",
        )
        print(f"Saved comparison_{idx:04d}.png")


if __name__ == "__main__":
    main()


