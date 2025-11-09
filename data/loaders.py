"""
Dataset loaders for SemanticKITTI, SynLiDAR, and CARLA.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import yaml

from .range_projection import RangeProjection, load_kitti_point_cloud, load_kitti_labels


class SemanticKITTIDataset(Dataset):
    """SemanticKITTI dataset loader with range-view projection."""
    
    def __init__(
        self,
        root: str,
        sequences: List[str],
        projection: RangeProjection,
        load_labels: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            root: Path to SemanticKITTI root directory
            sequences: List of sequence IDs (e.g., ["00", "01"])
            projection: RangeProjection instance
            load_labels: Whether to load semantic labels
            cache_dir: Directory to cache preprocessed range views
        """
        self.root = Path(root)
        self.sequences = sequences
        self.projection = projection
        self.load_labels = load_labels
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Build file list
        self.scan_files = []
        self.label_files = []
        
        for seq in sequences:
            seq_path = self.root / "sequences" / seq
            scan_path = seq_path / "velodyne"
            label_path = seq_path / "labels"
            
            if not scan_path.exists():
                print(f"Warning: {scan_path} does not exist, skipping sequence {seq}")
                continue
                
            scans = sorted(scan_path.glob("*.bin"))
            self.scan_files.extend(scans)
            
            if load_labels and label_path.exists():
                labels = sorted(label_path.glob("*.label"))
                self.label_files.extend(labels)
                
        print(f"Loaded {len(self.scan_files)} scans from sequences {sequences}")
        
    def __len__(self) -> int:
        return len(self.scan_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scan_file = self.scan_files[idx]
        
        # Check cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{scan_file.stem}.npz"
            if cache_file.exists():
                data = np.load(cache_file)
                return self._to_torch(data)
        
        # Load point cloud
        points, intensity = load_kitti_point_cloud(str(scan_file))
        
        # Load labels if available
        labels = None
        if self.load_labels and len(self.label_files) > idx:
            labels = load_kitti_labels(str(self.label_files[idx]))
        
        # Project to range view
        range_data = self.projection.project(points, intensity, labels)
        
        # Cache if enabled
        if self.cache_dir:
            np.savez_compressed(cache_file, **range_data)
        
        return self._to_torch(range_data)
    
    def _to_torch(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors."""
        result = {}
        for key, val in data.items():
            if isinstance(val, dict):
                val = val['arr_0'] if 'arr_0' in val else list(val.values())[0]
            result[key] = torch.from_numpy(np.array(val))
        return result


class SynLiDARDataset(Dataset):
    """SynLiDAR synthetic dataset loader."""
    
    def __init__(
        self,
        root: str,
        split: str,
        projection: RangeProjection,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            root: Path to SynLiDAR root directory
            split: 'train' or 'val'
            projection: RangeProjection instance
            cache_dir: Directory to cache preprocessed range views
        """
        self.root = Path(root)
        self.split = split
        self.projection = projection
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SynLiDAR structure: typically .bin files similar to KITTI
        split_path = self.root / split / "velodyne"
        if not split_path.exists():
            split_path = self.root / "velodyne"  # Alternative structure
            
        self.scan_files = sorted(split_path.glob("*.bin")) if split_path.exists() else []
        
        # Labels
        label_path = self.root / split / "labels"
        if not label_path.exists():
            label_path = self.root / "labels"
        self.label_files = sorted(label_path.glob("*.label")) if label_path.exists() else []
        
        print(f"Loaded {len(self.scan_files)} SynLiDAR scans from {split}")
        
    def __len__(self) -> int:
        return len(self.scan_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scan_file = self.scan_files[idx]
        
        # Check cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{scan_file.stem}.npz"
            if cache_file.exists():
                data = np.load(cache_file)
                return self._to_torch(data)
        
        # Load point cloud
        points, intensity = load_kitti_point_cloud(str(scan_file))
        
        # Load labels if available
        labels = None
        if len(self.label_files) > idx:
            labels = load_kitti_labels(str(self.label_files[idx]))
        
        # Project to range view
        range_data = self.projection.project(points, intensity, labels)
        
        # Cache if enabled
        if self.cache_dir:
            np.savez_compressed(cache_file, **range_data)
        
        return self._to_torch(range_data)
    
    def _to_torch(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors."""
        result = {}
        for key, val in data.items():
            if isinstance(val, dict):
                val = val['arr_0'] if 'arr_0' in val else list(val.values())[0]
            result[key] = torch.from_numpy(np.array(val))
        return result


class PairedDataset(Dataset):
    """Dataset for paired synthetic-to-real translation."""
    
    def __init__(
        self,
        synthetic_dataset: Dataset,
        real_dataset: Dataset,
        transform=None,
    ):
        """
        Args:
            synthetic_dataset: Synthetic data source
            real_dataset: Real data target
            transform: Optional transform to apply to both
        """
        self.synthetic_dataset = synthetic_dataset
        self.real_dataset = real_dataset
        self.transform = transform
        
        # Use minimum length for pairing
        self.length = min(len(synthetic_dataset), len(real_dataset))
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        synthetic = self.synthetic_dataset[idx % len(self.synthetic_dataset)]
        real = self.real_dataset[idx % len(self.real_dataset)]
        
        if self.transform:
            synthetic = self.transform(synthetic)
            real = self.transform(real)
            
        return synthetic, real


class RangeViewNPZDataset(Dataset):
    """Generic loader for preprocessed range-view NPZ files.

    Expected NPZ keys: 'range', 'intensity', 'mask', 'beam_angle', optional 'labels'.
    Directory structure: root_dir contains NPZ files (any naming).
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"NPZ root not found: {self.root}")
        self.files = sorted(self.root.glob("*.npz"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No NPZ files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self.files[idx]
        data = np.load(f)
        sample = {}
        for key in ["range", "intensity", "mask", "beam_angle"]:
            if key in data:
                sample[key] = torch.from_numpy(data[key])
        if "labels" in data:
            sample["labels"] = torch.from_numpy(data["labels"]).long()
        return sample


def get_semantickitti_splits() -> Dict[str, List[str]]:
    """Get standard SemanticKITTI train/val/test splits."""
    return {
        'train': [f"{i:02d}" for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]],
        'val': ['08'],
        'test': [f"{i:02d}" for i in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]],
    }


def create_dataloaders(
    config: dict,
    synthetic_root: Optional[str] = None,
    real_root: Optional[str] = None,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        synthetic_root: Path to synthetic data root
        real_root: Path to real data root (SemanticKITTI)
        num_workers: Number of dataloader workers
        
    Returns:
        Dictionary with 'train_syn', 'train_real', 'val_syn', 'val_real' loaders
    """
    # Create projection
    sensor_cfg = config['sensor']
    projection = RangeProjection(
        fov_up=sensor_cfg['fov_up'],
        fov_down=sensor_cfg['fov_down'],
        proj_h=sensor_cfg['n_rings'],
        proj_w=sensor_cfg['n_azimuth'],
        max_range=sensor_cfg['max_range'],
        min_range=sensor_cfg['min_range'],
    )
    
    loaders = {}
    
    data_cfg = config.get('data', {})
    real_dataset_type = data_cfg.get('real_dataset', 'semantickitti')

    # Resolve dataset roots from config if not provided explicitly
    if synthetic_root is None:
        synthetic_root = data_cfg.get('synlidar_root')
    if real_root is None:
        if real_dataset_type == 'semantickitti':
            real_root = data_cfg.get('semantickitti_root')
        elif real_dataset_type == 'nuscenes_npz':
            real_root = data_cfg.get('nuscenes_npz_root')

    # Real data
    if real_dataset_type == 'semantickitti' and real_root:
        splits = get_semantickitti_splits()

        train_real = SemanticKITTIDataset(
            root=real_root,
            sequences=splits['train'],
            projection=projection,
            load_labels=True,
            cache_dir=os.path.join(data_cfg['output_root'], 'cache', 'semantickitti_train')
            if data_cfg.get('output_root')
            else None,
        )

        val_real = SemanticKITTIDataset(
            root=real_root,
            sequences=splits['val'],
            projection=projection,
            load_labels=True,
            cache_dir=os.path.join(data_cfg['output_root'], 'cache', 'semantickitti_val')
            if data_cfg.get('output_root')
            else None,
        )

        loaders['train_real'] = DataLoader(
            train_real,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        loaders['val_real'] = DataLoader(
            val_real,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    elif real_dataset_type == 'nuscenes_npz' and real_root:
        train_split = data_cfg.get('nuscenes_train_split', 'mini_train')
        val_split = data_cfg.get('nuscenes_val_split', 'mini_val')

        train_dir = Path(real_root) / train_split
        val_dir = Path(real_root) / val_split

        train_real = RangeViewNPZDataset(root_dir=str(train_dir))
        val_real = RangeViewNPZDataset(root_dir=str(val_dir))

        loaders['train_real'] = DataLoader(
            train_real,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        loaders['val_real'] = DataLoader(
            val_real,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    elif real_root is None:
        raise FileNotFoundError(
            f"Real dataset root not provided for type '{real_dataset_type}'."
        )
    else:
        raise ValueError(f"Unsupported real dataset type: {real_dataset_type}")
    
    # Synthetic data
    if synthetic_root:
        train_syn = SynLiDARDataset(
            root=synthetic_root,
            split='train',
            projection=projection,
            cache_dir=os.path.join(data_cfg['output_root'], 'cache', 'synlidar_train')
            if data_cfg.get('output_root')
            else None,
        )

        val_syn = SynLiDARDataset(
            root=synthetic_root,
            split='val',
            projection=projection,
            cache_dir=os.path.join(data_cfg['output_root'], 'cache', 'synlidar_val')
            if data_cfg.get('output_root')
            else None,
        )

        loaders['train_syn'] = DataLoader(
            train_syn,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        loaders['val_syn'] = DataLoader(
            val_syn,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders

