"""
Modal deployment for GPU training and inference.
Run training on A100 GPUs with persistent storage.
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("lidar-sim2real")

# Create Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=9.5.0",
        "pyyaml>=6.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "open3d>=0.17.0",
        "scikit-learn>=1.2.0",
        "einops>=0.6.1",
        "timm>=0.9.0",
        "h5py>=3.8.0",
        "pandas>=2.0.0",
    )
)

# Create volumes for persistent storage
vol_data = modal.Volume.from_name("lidar-data", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("lidar-checkpoints", create_if_missing=True)
vol_eval = modal.Volume.from_name("lidar-eval", create_if_missing=True)

# Mount paths
DATA_DIR = "/data"
CKPT_DIR = "/checkpoints"
EVAL_DIR = "/eval"


@app.function(
    image=image,
    gpu="A100",
    timeout=86400,  # 24 hours
    volumes={
        DATA_DIR: vol_data,
        CKPT_DIR: vol_ckpt,
        EVAL_DIR: vol_eval,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],  # Optional: for W&B logging
)
def train_translator(
    config_path: str = "config.yaml",
    stage: str = "direct",  # "direct" or "diffusion"
    checkpoint_path: str = None,
):
    """
    Train the Sim2Real translator.
    
    Args:
        config_path: Path to config file
        stage: Training stage ("direct" for UNet, "diffusion" for diffusion upgrade)
        checkpoint_path: Optional checkpoint to resume from
    """
    import yaml
    import torch
    from data.loaders import create_dataloaders
    from models.unet import RangeViewUNet
    from models.diffusion import DiffusionModel
    from train.trainer import Trainer
    
    print(f"Starting training - Stage: {stage}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths for Modal volumes
    config['data']['semantickitti_root'] = f"{DATA_DIR}/SemanticKITTI"
    config['data']['synlidar_root'] = f"{DATA_DIR}/SynLiDAR"
    config['data']['output_root'] = DATA_DIR
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        config,
        synthetic_root=config['data']['synlidar_root'],
        real_root=config['data']['semantickitti_root'],
        num_workers=4,
    )
    
    # Create model
    if stage == "direct":
        model = RangeViewUNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            channel_multipliers=config['model']['channel_multipliers'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=config['model']['dropout'],
            use_circular_padding=config['model']['use_circular_padding'],
        )
        is_diffusion = False
        output_dir = f"{CKPT_DIR}/direct"
        
    elif stage == "diffusion":
        # Load UNet
        unet = RangeViewUNet(
            in_channels=config['model']['in_channels'] + config['model']['out_channels'] + 1,  # +1 for time
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            channel_multipliers=config['model']['channel_multipliers'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=config['model']['dropout'],
            use_circular_padding=config['model']['use_circular_padding'],
        )
        
        # Wrap in diffusion
        model = DiffusionModel(
            denoise_model=unet,
            timesteps=config['diffusion']['timesteps'],
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            cfg_dropout=0.1,
        )
        is_diffusion = True
        output_dir = f"{CKPT_DIR}/diffusion"
        
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader_syn=dataloaders['train_syn'],
        train_loader_real=dataloaders['train_real'],
        val_loader_syn=dataloaders['val_syn'],
        val_loader_real=dataloaders['val_real'],
        config=config,
        output_dir=output_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        is_diffusion=is_diffusion,
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
    
    # Train
    trainer.train()
    
    # Commit volumes
    vol_ckpt.commit()
    
    print("Training completed!")
    return str(Path(output_dir) / "checkpoints" / "best.pt")


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={
        DATA_DIR: vol_data,
        CKPT_DIR: vol_ckpt,
        EVAL_DIR: vol_eval,
    },
)
def translate_batch(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 16,
    num_steps: int = 50,
    cfg_scale: float = 3.0,
):
    """
    Batch translate synthetic scans to realistic scans.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        input_dir: Directory with synthetic range views
        output_dir: Directory to save translated range views
        batch_size: Batch size
        num_steps: Number of diffusion steps (if using diffusion)
        cfg_scale: Classifier-free guidance scale
    """
    import torch
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    
    print(f"Loading model from {checkpoint_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model (simplified - should match training config)
    # This is a placeholder - you'd load the model based on checkpoint config
    from models.unet import RangeViewUNet
    model = RangeViewUNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get input files
    input_path = Path(f"{DATA_DIR}/{input_dir}")
    output_path = Path(f"{EVAL_DIR}/{output_dir}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(input_path.glob("*.npz"))
    print(f"Found {len(input_files)} files to translate")
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(input_files), batch_size)):
            batch_files = input_files[i:i + batch_size]
            
            # Load batch
            batch_data = []
            for file in batch_files:
                data = np.load(file)
                batch_data.append(data)
            
            # Stack into tensors
            # (simplified - should properly handle data format)
            
            # Translate
            # output = model(...)
            
            # Save results
            for j, file in enumerate(batch_files):
                output_file = output_path / file.name
                # np.savez_compressed(output_file, ...)
    
    vol_eval.commit()
    print(f"Translation completed! Saved to {output_path}")


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={
        DATA_DIR: vol_data,
        EVAL_DIR: vol_eval,
    },
)
def evaluate_metrics(
    real_dir: str,
    generated_dir: str,
):
    """
    Evaluate metrics (FRID, FPD, MMD) between real and generated scans.
    
    Args:
        real_dir: Directory with real range views
        generated_dir: Directory with generated range views
    """
    import torch
    import numpy as np
    from pathlib import Path
    from eval.metrics import MetricsEvaluator
    
    print("Loading scans...")
    
    # Load real scans
    real_path = Path(f"{DATA_DIR}/{real_dir}")
    real_files = sorted(real_path.glob("*.npz"))
    
    real_scans = []
    for file in real_files[:1000]:  # Limit for speed
        data = np.load(file)
        # Stack channels
        scan = np.stack([data['range'], data['intensity'], data['mask']], axis=0)
        real_scans.append(scan)
    
    real_scans = torch.from_numpy(np.stack(real_scans))
    
    # Load generated scans
    gen_path = Path(f"{EVAL_DIR}/{generated_dir}")
    gen_files = sorted(gen_path.glob("*.npz"))
    
    gen_scans = []
    for file in gen_files[:1000]:
        data = np.load(file)
        scan = np.stack([data['range'], data['intensity'], data['mask']], axis=0)
        gen_scans.append(scan)
    
    gen_scans = torch.from_numpy(np.stack(gen_scans))
    
    print(f"Loaded {len(real_scans)} real and {len(gen_scans)} generated scans")
    
    # Compute metrics
    evaluator = MetricsEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    metrics = evaluator.compute_metrics(real_scans, gen_scans)
    
    # Save results
    import json
    results_file = Path(f"{EVAL_DIR}/metrics_{generated_dir.replace('/', '_')}.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    vol_eval.commit()
    
    print(f"Metrics saved to {results_file}")
    return metrics


@app.local_entrypoint()
def main(
    command: str = "train",
    stage: str = "direct",
    checkpoint: str = None,
):
    """
    Local entrypoint for Modal app.
    
    Usage:
        modal run modal_app.py --command train --stage direct
        modal run modal_app.py --command translate --checkpoint /checkpoints/best.pt
        modal run modal_app.py --command eval
    """
    if command == "train":
        checkpoint_path = train_translator.remote(stage=stage, checkpoint_path=checkpoint)
        print(f"Training completed! Checkpoint: {checkpoint_path}")
        
    elif command == "translate":
        if not checkpoint:
            raise ValueError("Must provide --checkpoint for translation")
        translate_batch.remote(
            checkpoint_path=checkpoint,
            input_dir="synlidar_val",
            output_dir="translated",
        )
        
    elif command == "eval":
        metrics = evaluate_metrics.remote(
            real_dir="semantickitti_val",
            generated_dir="translated",
        )
        print("Metrics:", metrics)
        
    else:
        raise ValueError(f"Unknown command: {command}")

