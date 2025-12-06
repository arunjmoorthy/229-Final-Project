"""
Range-view projection utilities for LiDAR point clouds.
Handles projection from 3D point cloud to 2D range image with proper masking.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict


class RangeProjection:
    """Projects 3D LiDAR point clouds into range-view (2D) representation."""
    
    def __init__(
        self,
        fov_up: float = 3.0,
        fov_down: float = -25.0,
        proj_h: int = 64,
        proj_w: int = 1024,
        max_range: float = 80.0,
        min_range: float = 0.5,
    ):
        """
        Args:
            fov_up: Upper FOV in degrees
            fov_down: Lower FOV in degrees
            proj_h: Height of range image (number of rings/beams)
            proj_w: Width of range image (azimuth resolution)
            max_range: Maximum range to consider (meters)
            min_range: Minimum range to consider (meters)
        """
        self.fov_up = fov_up * np.pi / 180.0  # Convert to radians
        self.fov_down = fov_down * np.pi / 180.0
        self.fov = abs(self.fov_up) + abs(self.fov_down)
        
        self.proj_h = proj_h
        self.proj_w = proj_w
        self.max_range = max_range
        self.min_range = min_range
        
    def project(
        self,
        points: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        ring_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Project point cloud to range view.
        
        Args:
            points: (N, 3) array of [x, y, z] coordinates
            intensity: (N,) array of intensity values (optional)
            labels: (N,) array of semantic labels (optional)
            ring_indices: (N,) array of laser ring indices (optional). If provided,
                these override the pitch-based vertical mapping and allow sensors
                such as the nuScenes HDL-32E to align perfectly with the output grid.
            
        Returns:
            Dictionary containing:
                - range: (H, W) range values
                - intensity: (H, W) intensity values
                - mask: (H, W) valid pixel mask
                - beam_angle: (H, W) normalized beam angle (ring index / H)
                - labels: (H, W) semantic labels (if provided)
                - xyz: (H, W, 3) xyz coordinates in range view
        """
        # Calculate range
        depth = np.linalg.norm(points[:, :3], ord=2, axis=1)
        
        # Filter by range
        valid_range = (depth >= self.min_range) & (depth <= self.max_range)
        points = points[valid_range]
        depth = depth[valid_range]
        
        if intensity is not None:
            intensity = intensity[valid_range]
        if labels is not None:
            labels = labels[valid_range]
        if ring_indices is not None:
            ring_indices = ring_indices[valid_range]
            
        # Calculate pitch and yaw
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        yaw = -np.arctan2(y, x)  # Azimuth angle
        pitch = np.arcsin(z / depth)  # Elevation angle
        
        # Map to range image coordinates
        # Yaw: -pi to pi -> 0 to W-1
        proj_x = 0.5 * (yaw / np.pi + 1.0) * (self.proj_w - 1)
        proj_x = np.floor(proj_x).astype(np.int32)
        proj_x = np.clip(proj_x, 0, self.proj_w - 1)
        
        if ring_indices is not None:
            # Map provided ring indices to projection rows. nuScenes provides
            # integer ring IDs (0-indexed). We rescale them if the requested
            # projection height differs from the number of unique rings.
            ring_indices = ring_indices.astype(np.float32)
            ring_min = ring_indices.min()
            ring_max = ring_indices.max()
            if ring_max > ring_min:
                ring_norm = (ring_indices - ring_min) / (ring_max - ring_min)
            else:
                ring_norm = np.zeros_like(ring_indices)
            proj_y = np.floor(ring_norm * (self.proj_h - 1)).astype(np.int32)
            proj_y = np.clip(proj_y, 0, self.proj_h - 1)
        else:
            # Pitch: fov_down to fov_up -> 0 to H-1
            proj_y = (1.0 - (pitch - self.fov_down) / self.fov) * (self.proj_h - 1)
            proj_y = np.floor(proj_y).astype(np.int32)
            proj_y = np.clip(proj_y, 0, self.proj_h - 1)
        
        # Initialize range image
        range_img = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        intensity_img = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        mask_img = np.zeros((self.proj_h, self.proj_w), dtype=np.bool_)
        xyz_img = np.zeros((self.proj_h, self.proj_w, 3), dtype=np.float32)
        
        if labels is not None:
            label_img = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)
        
        # Fill range image (keep closest point per pixel)
        order = np.argsort(depth)[::-1]  # Sort by depth (far to near)
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        depth = depth[order]
        points = points[order]
        if intensity is not None:
            intensity = intensity[order]
        if labels is not None:
            labels = labels[order]
        
        # Fill the images
        range_img[proj_y, proj_x] = depth
        mask_img[proj_y, proj_x] = True
        xyz_img[proj_y, proj_x] = points[:, :3]
        
        if intensity is not None:
            intensity_img[proj_y, proj_x] = intensity
        else:
            intensity_img[proj_y, proj_x] = depth  # Use depth as proxy
            
        if labels is not None:
            label_img[proj_y, proj_x] = labels
        
        # Normalize range to [0, 1]
        range_img_normalized = range_img / self.max_range
        
        # Normalize intensity to [0, 1]
        if intensity is not None and intensity_img[mask_img].max() > 0:
            intensity_img = intensity_img / intensity_img[mask_img].max()
        
        # Create beam angle channel (normalized ring index)
        beam_angle = np.repeat(
            np.arange(self.proj_h).reshape(-1, 1) / self.proj_h, 
            self.proj_w, 
            axis=1
        ).astype(np.float32)
        
        result = {
            'range': range_img_normalized,
            'intensity': intensity_img,
            'mask': mask_img,
            'beam_angle': beam_angle,
            'xyz': xyz_img,
        }
        
        if labels is not None:
            result['labels'] = label_img
            
        return result
    
    def unproject(
        self, 
        range_img: np.ndarray, 
        return_spherical: bool = False
    ) -> np.ndarray:
        """
        Convert range image back to 3D point cloud.
        
        Args:
            range_img: (H, W) range image (normalized 0-1 or absolute values)
            return_spherical: If True, return (range, azimuth, elevation)
            
        Returns:
            (N, 3) array of points [x, y, z] or [range, azimuth, elevation]
        """
        # Get pixel coordinates
        rows, cols = np.meshgrid(
            np.arange(self.proj_h), 
            np.arange(self.proj_w), 
            indexing='ij'
        )
        
        # Convert to angles
        yaw = (cols.astype(np.float32) / (self.proj_w - 1) - 0.5) * 2.0 * np.pi
        pitch = self.fov_down + (1.0 - rows.astype(np.float32) / (self.proj_h - 1)) * self.fov
        
        # Denormalize range if needed
        if range_img.max() <= 1.0:
            range_vals = range_img * self.max_range
        else:
            range_vals = range_img
            
        if return_spherical:
            return np.stack([range_vals, yaw, pitch], axis=-1)
        
        # Convert to Cartesian
        x = range_vals * np.cos(pitch) * np.cos(yaw)
        y = range_vals * np.cos(pitch) * np.sin(yaw)
        z = range_vals * np.sin(pitch)
        
        return np.stack([x, y, z], axis=-1)


def circular_pad_2d(tensor: torch.Tensor, pad_width: int) -> torch.Tensor:
    """
    Apply circular padding on the width dimension (azimuth).
    
    Args:
        tensor: (B, C, H, W) tensor
        pad_width: Number of pixels to pad on each side
        
    Returns:
        Padded tensor of shape (B, C, H, W + 2*pad_width)
    """
    # Pad width dimension circularly (360° wrap)
    left = tensor[..., -pad_width:]
    right = tensor[..., :pad_width]
    return torch.cat([left, tensor, right], dim=-1)


def load_bin_point_cloud(bin_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from binary file (KITTI/SynLiDAR format).
    
    Args:
        bin_path: Path to .bin file
        
    Returns:
        points: (N, 3) xyz coordinates
        intensity: (N,) intensity values
    """
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]
    intensity = scan[:, 3]
    return points, intensity


def load_bin_labels(label_path: str) -> np.ndarray:
    """
    Load semantic labels from binary file (KITTI/SynLiDAR format).
    
    Args:
        label_path: Path to .label file
        
    Returns:
        labels: (N,) semantic class labels
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels & 0xFFFF
    return labels

