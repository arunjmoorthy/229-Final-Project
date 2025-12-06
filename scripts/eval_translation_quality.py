import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loaders import RangeViewNPZDataset
from eval.metrics import MetricsEvaluator


def _stack_batch(dataset: RangeViewNPZDataset, max_samples: int) -> torch.Tensor:
    """Load up to max_samples and stack into (N, 3, H, W): [range, intensity, mask]."""
    n = min(len(dataset), max_samples)
    samples = []
    for idx in range(n):
        item = dataset[idx]
        samples.append(
            torch.stack(
                [
                    item["range"].float(),
                    item["intensity"].float(),
                    item["mask"].float(),
                ],
                dim=0,
            )
        )
    if not samples:
        raise ValueError("Dataset had zero samples to stack.")
    return torch.stack(samples)


def _compute_basic_stats(batch: torch.Tensor) -> Dict[str, float]:
    """Compute mean/std for range & intensity on valid pixels."""
    range_chan = batch[:, 0]
    intensity_chan = batch[:, 1]
    mask = batch[:, 2] > 0.5

    range_vals = range_chan[mask].cpu().numpy()
    intensity_vals = intensity_chan[mask].cpu().numpy()

    return {
        "num_samples": int(batch.shape[0]),
        "num_valid_pixels": int(mask.sum().item()),
        "range_mean": float(range_vals.mean()) if range_vals.size else float("nan"),
        "range_std": float(range_vals.std()) if range_vals.size else float("nan"),
        "intensity_mean": float(intensity_vals.mean())
        if intensity_vals.size
        else float("nan"),
        "intensity_std": float(intensity_vals.std())
        if intensity_vals.size
        else float("nan"),
    }


def _plot_histograms(
    real_batch: torch.Tensor,
    other_batch: torch.Tensor,
    label_other: str,
    out_path: Path,
):
    """Save histograms comparing range/intensity distributions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flatten_valid(batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        mask = batch[:, 2] > 0.5
        range_vals = batch[:, 0][mask].cpu().numpy()
        intensity_vals = batch[:, 1][mask].cpu().numpy()
        return range_vals, intensity_vals

    real_range, real_intensity = flatten_valid(real_batch)
    other_range, other_intensity = flatten_valid(other_batch)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(real_range, bins=80, alpha=0.6, label="Real")
    axes[0].hist(other_range, bins=80, alpha=0.6, label=label_other)
    axes[0].set_title("Range distribution (valid pixels)")
    axes[0].set_xlabel("Normalized range")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(real_intensity, bins=80, alpha=0.6, label="Real")
    axes[1].hist(other_intensity, bins=80, alpha=0.6, label=label_other)
    axes[1].set_title("Intensity distribution (valid pixels)")
    axes[1].set_xlabel("Normalized intensity")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic→real translation quality."
    )
    parser.add_argument(
        "--real_dir",
        required=True,
        help="Directory with real NPZ files (e.g., nuScenes mini_val).",
    )
    parser.add_argument(
        "--synthetic_dir",
        required=True,
        help="Directory with raw synthetic NPZs (e.g., SynLiDAR val).",
    )
    parser.add_argument(
        "--translated_dir",
        required=True,
        help="Directory with translated synthetic NPZs.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum samples to use from each set (to control runtime/memory).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for feature extractor (cuda or cpu).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Optional directory to save histogram plots.",
    )

    args = parser.parse_args()

    real_ds = RangeViewNPZDataset(args.real_dir)
    syn_ds = RangeViewNPZDataset(args.synthetic_dir)
    trn_ds = RangeViewNPZDataset(args.translated_dir)

    print(f"Loading up to {args.max_samples} samples per set...")
    real_batch = _stack_batch(real_ds, args.max_samples)
    syn_batch = _stack_batch(syn_ds, args.max_samples)
    trn_batch = _stack_batch(trn_ds, args.max_samples)

    print("Computing FRID/MMD metrics (raw synthetic vs real)...")
    evaluator = MetricsEvaluator(device=args.device)
    metrics_raw = evaluator.compute_metrics(real_batch.to(args.device), syn_batch.to(args.device))

    print("\nComputing FRID/MMD metrics (translated vs real)...")
    metrics_trn = evaluator.compute_metrics(real_batch.to(args.device), trn_batch.to(args.device))

    stats = {
        "real": _compute_basic_stats(real_batch),
        "synthetic": _compute_basic_stats(syn_batch),
        "translated": _compute_basic_stats(trn_batch),
    }

    print("\n===== Metrics =====")
    print("Raw synthetic vs real:")
    for k, v in metrics_raw.items():
        print(f"  {k}: {v:.4f}")

    print("\nTranslated vs real:")
    for k, v in metrics_trn.items():
        print(f"  {k}: {v:.4f}")

    print("\n===== Basic stats (valid pixels) =====")
    for label, stat in stats.items():
        print(f"{label.capitalize()}:")
        for k, v in stat.items():
            print(f"  {k}: {v}")

    if args.output_json:
        payload = {
            "settings": {
                "real_dir": args.real_dir,
                "synthetic_dir": args.synthetic_dir,
                "translated_dir": args.translated_dir,
                "max_samples": args.max_samples,
            },
            "raw_vs_real": metrics_raw,
            "translated_vs_real": metrics_trn,
            "basic_stats": stats,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved metrics to {out_path}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        _plot_histograms(real_batch, syn_batch, "Synthetic", plot_dir / "hist_raw_vs_real.png")
        _plot_histograms(real_batch, trn_batch, "Translated", plot_dir / "hist_translated_vs_real.png")
        print(f"Saved histogram plots under {plot_dir}")


if __name__ == "__main__":
    main()


