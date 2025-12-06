import numpy as np
import torch
from typing import Dict, Optional


class CalibratedAugmentation:
    """Apply physics-based augmentation using learned sensor profile."""
    
    def __init__(
        self,
        sensor_profile,
        dropout_scale: float = 1.0,
        intensity_noise_scale: float = 1.0,
        range_noise_std: float = 0.02,
        ring_dropout_prob: float = 0.05,
        apply_prob: float = 0.8,
    ):
        """
        Args:
            sensor_profile: Learned SensorProfile instance
            dropout_scale: Multiplier for dropout probabilities
            intensity_noise_scale: Multiplier for intensity noise
            range_noise_std: Std of additive range noise (normalized)
            ring_dropout_prob: Probability of dropping entire ring
            apply_prob: Probability of applying augmentation
        """
        self.sensor_profile = sensor_profile
        self.dropout_scale = dropout_scale
        self.intensity_noise_scale = intensity_noise_scale
        self.range_noise_std = range_noise_std
        self.ring_dropout_prob = ring_dropout_prob
        self.apply_prob = apply_prob
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply calibrated augmentation to a range-view sample.
        
        Args:
            sample: Dictionary with keys 'range', 'intensity', 'mask', etc.
            
        Returns:
            Augmented sample
        """
        if np.random.random() > self.apply_prob:
            return sample
        
        # Work with numpy for easier manipulation
        range_img = sample['range'].numpy() if torch.is_tensor(sample['range']) else sample['range']
        intensity_img = sample['intensity'].numpy() if torch.is_tensor(sample['intensity']) else sample['intensity']
        mask_img = sample['mask'].numpy() if torch.is_tensor(sample['mask']) else sample['mask']
        
        h, w = range_img.shape
        
        # 1. Range-dependent dropout
        if self.sensor_profile.dropout_vs_range is not None:
            dropout_probs = self.sensor_profile.get_dropout_prob(range_img) * self.dropout_scale
            dropout_mask = np.random.random(range_img.shape) > dropout_probs
            mask_img = mask_img & dropout_mask
        
        # 2. Ring-level dropout (simulate sensor failures)
        if self.sensor_profile.ring_dropout_rates is not None and self.ring_dropout_prob > 0:
            for ring_idx in range(h):
                if np.random.random() < self.ring_dropout_prob:
                    # Use learned ring dropout rate
                    ring_drop_rate = self.sensor_profile.ring_dropout_rates[ring_idx]
                    ring_mask = np.random.random(w) > ring_drop_rate
                    mask_img[ring_idx] &= ring_mask
        
        # 3. Intensity augmentation (falloff + noise)
        if self.sensor_profile.intensity_vs_range is not None:
            # Get expected intensity statistics
            intensity_means, intensity_stds = self.sensor_profile.get_intensity_stats(range_img)
            
            # Apply intensity falloff with range
            intensity_img = intensity_img * (intensity_means + 0.5) / 1.5
            
            # Add calibrated noise
            noise = np.random.randn(*intensity_img.shape) * intensity_stds * self.intensity_noise_scale
            intensity_img = intensity_img + noise
            intensity_img = np.clip(intensity_img, 0, 1)
        
        # 4. Range noise (measurement uncertainty)
        if self.range_noise_std > 0:
            range_noise = np.random.randn(*range_img.shape) * self.range_noise_std
            range_img = range_img + range_noise * mask_img.astype(float)
            range_img = np.clip(range_img, 0, 1)
        
        # 5. Ring artifacts (horizontal streaks)
        if np.random.random() < 0.3:  # 30% chance
            affected_rings = np.random.choice(h, size=max(1, h // 20), replace=False)
            for ring_idx in affected_rings:
                # Add streak artifact
                streak_intensity = np.random.uniform(0.9, 1.1)
                intensity_img[ring_idx] *= streak_intensity
                intensity_img[ring_idx] = np.clip(intensity_img[ring_idx], 0, 1)
        
        # Update sample
        sample['range'] = torch.from_numpy(range_img) if not torch.is_tensor(sample['range']) else torch.tensor(range_img)
        sample['intensity'] = torch.from_numpy(intensity_img) if not torch.is_tensor(sample['intensity']) else torch.tensor(intensity_img)
        sample['mask'] = torch.from_numpy(mask_img) if not torch.is_tensor(sample['mask']) else torch.tensor(mask_img)
        
        return sample


class StandardAugmentation:
    """Standard data augmentation for range views."""
    
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        intensity_jitter: float = 0.1,
        range_jitter: float = 0.05,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.intensity_jitter = intensity_jitter
        self.range_jitter = range_jitter
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply standard augmentations."""
        # Horizontal flip (azimuth flip)
        if np.random.random() < self.horizontal_flip_prob:
            for key in ['range', 'intensity', 'mask', 'beam_angle', 'xyz']:
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=[-1])
        
        # Intensity jitter
        if self.intensity_jitter > 0 and 'intensity' in sample:
            jitter = 1.0 + np.random.uniform(-self.intensity_jitter, self.intensity_jitter)
            sample['intensity'] = torch.clamp(sample['intensity'] * jitter, 0, 1)
        
        # Range jitter
        if self.range_jitter > 0 and 'range' in sample:
            mask = sample.get('mask', torch.ones_like(sample['range'], dtype=torch.bool))
            jitter = torch.randn_like(sample['range']) * self.range_jitter * mask.float()
            sample['range'] = torch.clamp(sample['range'] + jitter, 0, 1)
        
        return sample


def compose_transforms(*transforms):
    """Compose multiple transforms."""
    def composed(sample):
        for transform in transforms:
            sample = transform(sample)
        return sample
    return composed

