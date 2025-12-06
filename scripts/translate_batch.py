import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet import RangeViewUNet
from models.diffusion import DiffusionModel


def translate_batch(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 16,
    device: str = 'cuda',
    is_diffusion: bool = False,
    num_steps: int = 50,
    cfg_scale: float = 3.0,
):
    """
    Translate synthetic scans to realistic scans.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        input_dir: Directory with synthetic range views (.npz files)
        output_dir: Directory to save translated range views
        batch_size: Batch size for inference
        device: Device to use
        is_diffusion: Whether model is diffusion
        num_steps: Diffusion sampling steps
        cfg_scale: Classifier-free guidance scale
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model
    if is_diffusion:
        unet = RangeViewUNet(
            in_channels=config['model']['in_channels'] + config['model']['out_channels'] + 1,
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            channel_multipliers=config['model']['channel_multipliers'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=0.0,  # No dropout at inference
            use_circular_padding=config['model']['use_circular_padding'],
        )
        model = DiffusionModel(
            denoise_model=unet,
            timesteps=config['diffusion']['timesteps'],
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
        )
    else:
        model = RangeViewUNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            channel_multipliers=config['model']['channel_multipliers'],
            num_res_blocks=config['model']['num_res_blocks'],
            attention_resolutions=config['model']['attention_resolutions'],
            dropout=0.0,
            use_circular_padding=config['model']['use_circular_padding'],
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get input files
    input_files = sorted(input_dir.glob("*.npz"))
    print(f"Found {len(input_files)} files to translate")
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(input_files), batch_size), desc="Translating"):
            batch_files = input_files[i:i + batch_size]
            
            # Load batch
            batch_data = []
            for file in batch_files:
                data = np.load(file)
                batch_data.append({
                    'range': data['range'],
                    'intensity': data['intensity'],
                    'mask': data['mask'],
                    'beam_angle': data['beam_angle'],
                })
            
            # Stack into tensors
            ranges = torch.stack([torch.from_numpy(d['range']) for d in batch_data]).to(device)
            intensities = torch.stack([torch.from_numpy(d['intensity']) for d in batch_data]).to(device)
            masks = torch.stack([torch.from_numpy(d['mask']) for d in batch_data]).to(device)
            beam_angles = torch.stack([torch.from_numpy(d['beam_angle']) for d in batch_data]).to(device)
            
            # Prepare input
            syn_input = torch.stack([ranges, intensities, masks.float(), beam_angles], dim=1)
            mask_input = masks.unsqueeze(1)
            
            # Translate
            if is_diffusion:
                translated = model.sample(
                    condition=syn_input,
                    mask=mask_input,
                    num_steps=num_steps,
                    cfg_scale=cfg_scale,
                )
            else:
                translated = model(syn_input, mask_input)
            
            # Save results
            translated = translated.cpu().numpy()
            
            for j, file in enumerate(batch_files):
                output_file = output_dir / file.name
                
                # Extract channels
                trans_range = translated[j, 0]
                trans_intensity = translated[j, 1] if translated.shape[1] > 1 else intensities[j].cpu().numpy()
                trans_mask = translated[j, 2] > 0.5 if translated.shape[1] > 2 else masks[j].cpu().numpy()
                
                # Save
                np.savez_compressed(
                    output_file,
                    range=trans_range,
                    intensity=trans_intensity,
                    mask=trans_mask,
                    beam_angle=beam_angles[j].cpu().numpy(),
                )
    
    print(f"\nTranslation completed!")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch translate synthetic scans")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with synthetic range views"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save translated range views"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use"
    )
    parser.add_argument(
        "--diffusion",
        action='store_true',
        help="Use diffusion model"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale"
    )
    
    args = parser.parse_args()
    
    translate_batch(
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        is_diffusion=args.diffusion,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
    )


if __name__ == "__main__":
    main()

