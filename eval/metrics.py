"""
Evaluation metrics for LiDAR distribution comparison.
Implements FRID (Fréchet Range Image Distance), FPD (Fréchet Point Distance), and MMD.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from typing import Tuple, Optional
from tqdm import tqdm


class SimpleFeatureExtractor(nn.Module):
    """
    Simple CNN feature extractor for range images.
    Used for computing FRID (similar to FID for images).
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) range images
            
        Returns:
            (B, 512) feature vectors
        """
        x = self.features(x)
        return x.view(x.size(0), -1)


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """
    Calculate Fréchet distance between two Gaussians.
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small value for numerical stability
        
    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    # Calculate mean difference
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical errors
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of features.
    
    Args:
        features: (N, D) feature array
        
    Returns:
        mu: Mean vector
        sigma: Covariance matrix
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_frid(
    real_features: np.ndarray,
    generated_features: np.ndarray,
) -> float:
    """
    Compute Fréchet Range Image Distance (FRID).
    Similar to FID but for range images.
    
    Args:
        real_features: (N1, D) features from real scans
        generated_features: (N2, D) features from generated scans
        
    Returns:
        FRID score (lower is better)
    """
    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(generated_features)
    
    frid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return float(frid)


def compute_mmd(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    kernel: str = 'rbf',
    bandwidth: float = 1.0,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD).
    
    Args:
        real_features: (N1, D) features from real scans
        generated_features: (N2, D) features from generated scans
        kernel: Kernel type ('rbf' or 'linear')
        bandwidth: Bandwidth for RBF kernel
        
    Returns:
        MMD score (lower is better)
    """
    n_real = real_features.shape[0]
    n_gen = generated_features.shape[0]
    
    if kernel == 'rbf':
        # RBF kernel
        gamma = 1.0 / (2 * bandwidth ** 2)
        
        # K(real, real)
        xx = cdist(real_features, real_features, 'sqeuclidean')
        kxx = np.exp(-gamma * xx)
        
        # K(gen, gen)
        yy = cdist(generated_features, generated_features, 'sqeuclidean')
        kyy = np.exp(-gamma * yy)
        
        # K(real, gen)
        xy = cdist(real_features, generated_features, 'sqeuclidean')
        kxy = np.exp(-gamma * xy)
        
    elif kernel == 'linear':
        # Linear kernel
        kxx = real_features @ real_features.T
        kyy = generated_features @ generated_features.T
        kxy = real_features @ generated_features.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # MMD^2
    mmd2 = (kxx.sum() - np.trace(kxx)) / (n_real * (n_real - 1))
    mmd2 += (kyy.sum() - np.trace(kyy)) / (n_gen * (n_gen - 1))
    mmd2 -= 2 * kxy.sum() / (n_real * n_gen)
    
    return float(np.sqrt(max(mmd2, 0)))


class MetricsEvaluator:
    """Evaluator for computing distribution metrics."""
    
    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            feature_extractor: Feature extraction model
            device: Device to use
        """
        if feature_extractor is None:
            feature_extractor = SimpleFeatureExtractor(in_channels=3)
        
        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()
        self.device = device
    
    @torch.no_grad()
    def extract_features(
        self,
        range_views: torch.Tensor,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract features from range views.
        
        Args:
            range_views: (N, C, H, W) range images
            batch_size: Batch size for processing
            
        Returns:
            (N, D) feature array
        """
        n_samples = range_views.shape[0]
        all_features = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting features", leave=False):
            batch = range_views[i:i + batch_size].to(self.device)
            features = self.feature_extractor(batch)
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def compute_metrics(
        self,
        real_range_views: torch.Tensor,
        generated_range_views: torch.Tensor,
        batch_size: int = 32,
    ) -> dict:
        """
        Compute all metrics.
        
        Args:
            real_range_views: (N1, C, H, W) real range images
            generated_range_views: (N2, C, H, W) generated range images
            batch_size: Batch size for feature extraction
            
        Returns:
            Dictionary with metric scores
        """
        print("Computing metrics...")
        
        # Extract features
        print("Extracting features from real scans...")
        real_features = self.extract_features(real_range_views, batch_size)
        
        print("Extracting features from generated scans...")
        gen_features = self.extract_features(generated_range_views, batch_size)
        
        # Compute metrics
        metrics = {}
        
        print("Computing FRID...")
        metrics['frid'] = compute_frid(real_features, gen_features)
        
        print("Computing MMD...")
        metrics['mmd_rbf'] = compute_mmd(real_features, gen_features, kernel='rbf', bandwidth=1.0)
        metrics['mmd_linear'] = compute_mmd(real_features, gen_features, kernel='linear')
        
        # Additional simple metrics on raw data
        print("Computing simple statistics...")
        
        # Mean absolute difference in range
        real_range = real_range_views[:, 0].numpy() if isinstance(real_range_views, torch.Tensor) else real_range_views[:, 0]
        gen_range = generated_range_views[:, 0].numpy() if isinstance(generated_range_views, torch.Tensor) else generated_range_views[:, 0]
        
        metrics['range_mae'] = float(np.abs(real_range.mean() - gen_range.mean()))
        metrics['range_std_diff'] = float(np.abs(real_range.std() - gen_range.std()))
        
        # Mean absolute difference in intensity
        if real_range_views.shape[1] > 1:
            real_intensity = real_range_views[:, 1].numpy() if isinstance(real_range_views, torch.Tensor) else real_range_views[:, 1]
            gen_intensity = generated_range_views[:, 1].numpy() if isinstance(generated_range_views, torch.Tensor) else generated_range_views[:, 1]
            
            metrics['intensity_mae'] = float(np.abs(real_intensity.mean() - gen_intensity.mean()))
            metrics['intensity_std_diff'] = float(np.abs(real_intensity.std() - gen_intensity.std()))
        
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics


def chamfer_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    batch_size: int = 1000,
) -> float:
    """
    Compute Chamfer distance between two point clouds.
    
    Args:
        points1: (N, 3) first point cloud
        points2: (M, 3) second point cloud
        batch_size: Batch size for distance computation
        
    Returns:
        Chamfer distance
    """
    n1 = points1.shape[0]
    n2 = points2.shape[0]
    
    # Compute distances in batches
    dist1_sum = 0.0
    for i in range(0, n1, batch_size):
        batch1 = points1[i:i + batch_size]
        dists = cdist(batch1, points2, 'euclidean')
        dist1_sum += np.min(dists, axis=1).sum()
    
    dist2_sum = 0.0
    for i in range(0, n2, batch_size):
        batch2 = points2[i:i + batch_size]
        dists = cdist(batch2, points1, 'euclidean')
        dist2_sum += np.min(dists, axis=1).sum()
    
    chamfer = (dist1_sum / n1 + dist2_sum / n2) / 2
    
    return float(chamfer)

