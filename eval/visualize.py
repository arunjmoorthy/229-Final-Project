import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple
import torch


def visualize_range_view(
    range_img: np.ndarray,
    intensity_img: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    title: str = "Range View",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Visualize range view with multiple channels.
    
    Args:
        range_img: (H, W) range image
        intensity_img: (H, W) intensity image (optional)
        mask: (H, W) boolean mask (optional)
        labels: (H, W) semantic labels (optional)
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    """
    num_plots = 1 + (intensity_img is not None) + (mask is not None) + (labels is not None)
    
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Range image
    range_masked = range_img.copy()
    if mask is not None:
        range_masked[~mask] = 0
    
    im = axes[plot_idx].imshow(range_masked, cmap='viridis', aspect='auto')
    axes[plot_idx].set_title('Range')
    axes[plot_idx].set_xlabel('Azimuth')
    axes[plot_idx].set_ylabel('Ring')
    plt.colorbar(im, ax=axes[plot_idx])
    plot_idx += 1
    
    # Intensity image
    if intensity_img is not None:
        intensity_masked = intensity_img.copy()
        if mask is not None:
            intensity_masked[~mask] = 0
        
        im = axes[plot_idx].imshow(intensity_masked, cmap='gray', aspect='auto')
        axes[plot_idx].set_title('Intensity')
        axes[plot_idx].set_xlabel('Azimuth')
        axes[plot_idx].set_ylabel('Ring')
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1
    
    # Mask
    if mask is not None:
        axes[plot_idx].imshow(mask, cmap='gray', aspect='auto')
        axes[plot_idx].set_title('Valid Mask')
        axes[plot_idx].set_xlabel('Azimuth')
        axes[plot_idx].set_ylabel('Ring')
        plot_idx += 1
    
    # Labels
    if labels is not None:
        labels_masked = labels.copy()
        if mask is not None:
            labels_masked[~mask] = 0
        
        im = axes[plot_idx].imshow(labels_masked, cmap='tab20', aspect='auto', vmin=0, vmax=19)
        axes[plot_idx].set_title('Semantic Labels')
        axes[plot_idx].set_xlabel('Azimuth')
        axes[plot_idx].set_ylabel('Ring')
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_range_views(
    synthetic: dict,
    real: dict,
    translated: Optional[dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 10),
):
    """
    Compare synthetic, real, and translated range views side by side.
    
    Args:
        synthetic: Dict with 'range', 'intensity', 'mask'
        real: Dict with 'range', 'intensity', 'mask'
        translated: Dict with 'range', 'intensity', 'mask' (optional)
        save_path: Path to save figure
        figsize: Figure size
    """
    num_cols = 3 if translated is not None else 2
    fig, axes = plt.subplots(2, num_cols, figsize=figsize)
    
    def plot_sample(ax_range, ax_intensity, data, title):
        """Helper to plot one sample."""
        range_img = data['range']
        intensity_img = data['intensity']
        mask = data.get('mask', np.ones_like(range_img, dtype=bool))
        
        # Apply mask
        range_masked = range_img.copy()
        intensity_masked = intensity_img.copy()
        range_masked[~mask] = 0
        intensity_masked[~mask] = 0
        
        # Plot range
        im1 = ax_range.imshow(range_masked, cmap='viridis', aspect='auto')
        ax_range.set_title(f'{title} - Range')
        ax_range.set_ylabel('Ring')
        plt.colorbar(im1, ax=ax_range)
        
        # Plot intensity
        im2 = ax_intensity.imshow(intensity_masked, cmap='gray', aspect='auto')
        ax_intensity.set_title(f'{title} - Intensity')
        ax_intensity.set_xlabel('Azimuth')
        ax_intensity.set_ylabel('Ring')
        plt.colorbar(im2, ax=ax_intensity)
    
    # Plot samples
    plot_sample(axes[0, 0], axes[1, 0], synthetic, 'Synthetic')
    plot_sample(axes[0, 1], axes[1, 1], real, 'Real')
    
    if translated is not None:
        plot_sample(axes[0, 2], axes[1, 2], translated, 'Translated')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None,
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(
    metrics_dict: dict,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of metrics across different methods.
    
    Args:
        metrics_dict: Dict mapping method names to metric dicts
        save_path: Path to save figure
    """
    methods = list(metrics_dict.keys())
    metric_names = list(metrics_dict[methods[0]].keys())
    
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(metric_names):
        values = [metrics_dict[method][metric_name] for method in methods]
        
        axes[i].bar(methods, values, color='steelblue', alpha=0.7)
        axes[i].set_title(metric_name.upper(), fontsize=12)
        axes[i].set_ylabel('Value', fontsize=11)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, val in enumerate(values):
            axes[i].text(j, val, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_3d_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    save_path: Optional[str] = None,
):
    """
    Visualize 3D point cloud.
    
    Args:
        points: (N, 3) array of [x, y, z]
        colors: (N, 3) array of RGB colors (optional)
        title: Plot title
        save_path: Path to save figure
    """
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd], window_name=title)
        
    except ImportError:
        print("Open3D not installed. Using matplotlib 3D scatter instead.")
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=0.1, alpha=0.5)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap='viridis', s=0.1, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()

