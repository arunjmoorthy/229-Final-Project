"""
Sensor profile learning and calibration utilities.
Learns dropout patterns, intensity falloff, and ring artifacts from real data.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


class SensorProfile:
    """Stores and applies learned sensor characteristics."""
    
    def __init__(self):
        self.dropout_vs_range = None  # Dropout probability vs range
        self.dropout_vs_incidence = None  # Dropout probability vs incidence angle
        self.intensity_vs_range = None  # Mean intensity vs range
        self.intensity_std_vs_range = None  # Intensity std vs range
        self.ring_dropout_rates = None  # Per-ring dropout rates
        self.range_bins = None
        self.incidence_bins = None
        
    def learn_from_scans(
        self, 
        range_views: List[np.ndarray],
        intensity_views: List[np.ndarray],
        mask_views: List[np.ndarray],
        num_range_bins: int = 50,
        num_incidence_bins: int = 20,
    ):
        """
        Learn sensor characteristics from a collection of real scans.
        
        Args:
            range_views: List of (H, W) range images
            intensity_views: List of (H, W) intensity images
            mask_views: List of (H, W) boolean masks
            num_range_bins: Number of bins for range-dependent stats
            num_incidence_bins: Number of bins for incidence angle stats
        """
        print("Learning sensor profile from real data...")
        
        # Collect all valid measurements
        all_ranges = []
        all_intensities = []
        all_masks = []
        
        for range_view, intensity_view, mask_view in zip(range_views, intensity_views, mask_views):
            all_ranges.append(range_view[mask_view])
            all_intensities.append(intensity_view[mask_view])
            all_masks.append(mask_view)
        
        all_ranges = np.concatenate(all_ranges)
        all_intensities = np.concatenate(all_intensities)
        
        # Learn dropout vs range
        max_range = all_ranges.max()
        self.range_bins = np.linspace(0, max_range, num_range_bins + 1)
        
        # Compute dropout rate per range bin
        dropout_rates = []
        for i in range(len(self.range_bins) - 1):
            bin_mask = (all_ranges >= self.range_bins[i]) & (all_ranges < self.range_bins[i + 1])
            if bin_mask.sum() > 0:
                # Estimate dropout as inverse of point density
                dropout_rate = 1.0 / (bin_mask.sum() / len(all_ranges) + 1e-6)
                dropout_rate = min(0.5, max(0.01, dropout_rate / 100))  # Normalize
            else:
                dropout_rate = 0.1
            dropout_rates.append(dropout_rate)
        
        self.dropout_vs_range = np.array(dropout_rates)
        
        # Learn intensity vs range
        intensity_means, _, _ = binned_statistic(
            all_ranges, all_intensities, statistic='mean', bins=self.range_bins
        )
        intensity_stds, _, _ = binned_statistic(
            all_ranges, all_intensities, statistic='std', bins=self.range_bins
        )
        
        self.intensity_vs_range = np.nan_to_num(intensity_means, nan=0.5)
        self.intensity_std_vs_range = np.nan_to_num(intensity_stds, nan=0.1)
        
        # Learn per-ring dropout rates
        h = mask_views[0].shape[0]
        ring_valid_counts = np.zeros(h)
        ring_total_counts = np.zeros(h)
        
        for mask_view in all_masks:
            ring_valid_counts += mask_view.sum(axis=1)
            ring_total_counts += mask_view.shape[1]
        
        ring_valid_rates = ring_valid_counts / (ring_total_counts + 1e-6)
        self.ring_dropout_rates = 1.0 - ring_valid_rates
        
        print(f"Learned profile: range bins={len(self.range_bins)-1}, "
              f"mean dropout={self.dropout_vs_range.mean():.3f}, "
              f"mean intensity={self.intensity_vs_range.mean():.3f}")
    
    def get_dropout_prob(self, range_values: np.ndarray) -> np.ndarray:
        """Get dropout probability for given range values."""
        if self.dropout_vs_range is None:
            return np.zeros_like(range_values)
        
        bin_indices = np.digitize(range_values, self.range_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.dropout_vs_range) - 1)
        return self.dropout_vs_range[bin_indices]
    
    def get_intensity_stats(self, range_values: np.ndarray) -> tuple:
        """Get expected intensity mean and std for given range values."""
        if self.intensity_vs_range is None:
            return np.ones_like(range_values) * 0.5, np.ones_like(range_values) * 0.1
        
        bin_indices = np.digitize(range_values, self.range_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.intensity_vs_range) - 1)
        
        means = self.intensity_vs_range[bin_indices]
        stds = self.intensity_std_vs_range[bin_indices]
        return means, stds
    
    def save(self, path: str):
        """Save profile to JSON file."""
        data = {
            'dropout_vs_range': self.dropout_vs_range.tolist() if self.dropout_vs_range is not None else None,
            'intensity_vs_range': self.intensity_vs_range.tolist() if self.intensity_vs_range is not None else None,
            'intensity_std_vs_range': self.intensity_std_vs_range.tolist() if self.intensity_std_vs_range is not None else None,
            'ring_dropout_rates': self.ring_dropout_rates.tolist() if self.ring_dropout_rates is not None else None,
            'range_bins': self.range_bins.tolist() if self.range_bins is not None else None,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved sensor profile to {path}")
    
    def load(self, path: str):
        """Load profile from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.dropout_vs_range = np.array(data['dropout_vs_range']) if data['dropout_vs_range'] else None
        self.intensity_vs_range = np.array(data['intensity_vs_range']) if data['intensity_vs_range'] else None
        self.intensity_std_vs_range = np.array(data['intensity_std_vs_range']) if data['intensity_std_vs_range'] else None
        self.ring_dropout_rates = np.array(data['ring_dropout_rates']) if data['ring_dropout_rates'] else None
        self.range_bins = np.array(data['range_bins']) if data['range_bins'] else None
        
        print(f"Loaded sensor profile from {path}")
    
    def plot(self, save_path: Optional[str] = None):
        """Visualize the learned sensor profile."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Dropout vs range
        if self.dropout_vs_range is not None:
            bin_centers = (self.range_bins[:-1] + self.range_bins[1:]) / 2
            axes[0, 0].plot(bin_centers, self.dropout_vs_range)
            axes[0, 0].set_xlabel('Range (m)')
            axes[0, 0].set_ylabel('Dropout Probability')
            axes[0, 0].set_title('Dropout vs Range')
            axes[0, 0].grid(True)
        
        # Intensity vs range
        if self.intensity_vs_range is not None:
            bin_centers = (self.range_bins[:-1] + self.range_bins[1:]) / 2
            axes[0, 1].plot(bin_centers, self.intensity_vs_range, label='Mean')
            axes[0, 1].fill_between(
                bin_centers,
                self.intensity_vs_range - self.intensity_std_vs_range,
                self.intensity_vs_range + self.intensity_std_vs_range,
                alpha=0.3,
                label='±1 Std'
            )
            axes[0, 1].set_xlabel('Range (m)')
            axes[0, 1].set_ylabel('Intensity')
            axes[0, 1].set_title('Intensity vs Range')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Ring dropout rates
        if self.ring_dropout_rates is not None:
            axes[1, 0].plot(self.ring_dropout_rates)
            axes[1, 0].set_xlabel('Ring Index')
            axes[1, 0].set_ylabel('Dropout Rate')
            axes[1, 0].set_title('Per-Ring Dropout Rates')
            axes[1, 0].grid(True)
        
        # Summary stats
        axes[1, 1].axis('off')
        summary_text = f"Sensor Profile Summary\n\n"
        if self.dropout_vs_range is not None:
            summary_text += f"Dropout rate: {self.dropout_vs_range.mean():.3f} ± {self.dropout_vs_range.std():.3f}\n"
        if self.intensity_vs_range is not None:
            summary_text += f"Intensity: {self.intensity_vs_range.mean():.3f} ± {self.intensity_vs_range.std():.3f}\n"
        if self.ring_dropout_rates is not None:
            summary_text += f"Ring dropout: {self.ring_dropout_rates.mean():.3f} ± {self.ring_dropout_rates.std():.3f}\n"
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved profile plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

