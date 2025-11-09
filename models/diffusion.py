"""
Diffusion model wrapper for range-view translation.
Implements DDPM with classifier-free guidance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from tqdm import tqdm


def get_beta_schedule(schedule: str, timesteps: int, start: float = 0.0001, end: float = 0.02) -> torch.Tensor:
    """Get noise schedule."""
    if schedule == "linear":
        return torch.linspace(start, end, timesteps)
    elif schedule == "cosine":
        # Cosine schedule from Improved DDPM
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class DiffusionModel(nn.Module):
    """Diffusion model for conditional image-to-image translation."""
    
    def __init__(
        self,
        denoise_model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cfg_dropout: float = 0.1,  # Classifier-free guidance dropout
    ):
        """
        Args:
            denoise_model: UNet or similar denoising network
            timesteps: Number of diffusion timesteps
            beta_schedule: Noise schedule type
            beta_start: Starting beta value
            beta_end: Ending beta value
            cfg_dropout: Probability of dropping condition for CFG
        """
        super().__init__()
        
        self.denoise_model = denoise_model
        self.timesteps = timesteps
        self.cfg_dropout = cfg_dropout
        
        # Noise schedule
        betas = get_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Precompute useful quantities
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        
        # Posterior variance for sampling
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: add noise to x_start.
        
        Args:
            x_start: (B, C, H, W) clean samples
            t: (B,) timesteps
            noise: (B, C, H, W) noise (optional, sampled if not provided)
            
        Returns:
            Noisy samples at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise from noisy sample.
        
        Args:
            x_t: (B, C, H, W) noisy sample at time t
            t: (B,) timesteps
            condition: (B, C_cond, H, W) conditioning input (synthetic scan)
            mask: (B, 1, H, W) optional mask
            
        Returns:
            Predicted noise
        """
        # Concatenate noisy target with condition
        # Embed timestep (simple approach: add as channel)
        t_embed = t.float().view(-1, 1, 1, 1).expand(-1, 1, x_t.shape[2], x_t.shape[3]) / self.timesteps
        
        model_input = torch.cat([x_t, condition, t_embed], dim=1)
        
        return self.denoise_model(model_input, mask)
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step.
        
        Args:
            x_t: (B, C, H, W) noisy sample at time t
            t: (B,) timesteps
            condition: Conditioning input
            mask: Optional mask
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Denoised sample at time t-1
        """
        # Predict noise
        if cfg_scale > 1.0:
            # Conditional prediction
            noise_pred_cond = self.predict_noise(x_t, t, condition, mask)
            
            # Unconditional prediction (zero condition)
            noise_pred_uncond = self.predict_noise(x_t, t, torch.zeros_like(condition), mask)
            
            # Classifier-free guidance
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = self.predict_noise(x_t, t, condition, mask)
        
        # Compute x_{t-1}
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_alphas_t * (x_t - self.betas[t].view(-1, 1, 1, 1) * noise_pred / 
                                       self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1))
        
        if t[0] > 0:
            # Add noise (not at final step)
            posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return mean + torch.exp(0.5 * posterior_log_variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        cfg_scale: float = 3.0,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.
        
        Args:
            condition: (B, C_cond, H, W) conditioning input
            mask: Optional mask
            num_steps: Number of sampling steps (default: self.timesteps)
            cfg_scale: Classifier-free guidance scale
            return_all: Return all intermediate samples
            
        Returns:
            Generated samples (B, C, H, W)
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        if num_steps is None:
            num_steps = self.timesteps
        
        # Start from pure noise
        x = torch.randn(batch_size, self.denoise_model.out_channels, 
                       condition.shape[2], condition.shape[3], device=device)
        
        if mask is not None:
            x = x * mask
        
        # Reverse process
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        all_samples = [x] if return_all else None
        
        for t in tqdm(timesteps, desc="Sampling", leave=False):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, condition, mask, cfg_scale)
            
            if mask is not None:
                x = x * mask
            
            if return_all:
                all_samples.append(x)
        
        return all_samples if return_all else x
    
    def forward(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training forward pass: add noise and predict it.
        
        Args:
            x_start: (B, C, H, W) clean target samples
            condition: (B, C_cond, H, W) conditioning input
            mask: Optional mask
            
        Returns:
            Predicted noise
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        if mask is not None:
            noise = noise * mask
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)
        
        # Classifier-free guidance: randomly drop condition
        if self.training and self.cfg_dropout > 0:
            dropout_mask = torch.rand(batch_size, device=device) > self.cfg_dropout
            dropout_mask = dropout_mask.view(-1, 1, 1, 1)
            condition = condition * dropout_mask
        
        # Predict noise
        noise_pred = self.predict_noise(x_t, t, condition, mask)
        
        return noise_pred, noise


def test_diffusion():
    """Test diffusion model."""
    from .unet import RangeViewUNet
    
    # Create UNet
    unet = RangeViewUNet(
        in_channels=4 + 4 + 1,  # x_t + condition + t_embed
        out_channels=3,
        base_channels=32,
        channel_multipliers=[1, 2, 4],
        use_circular_padding=True,
    )
    
    # Create diffusion model
    model = DiffusionModel(unet, timesteps=100)
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 256)
    condition = torch.randn(2, 4, 64, 256)
    mask = torch.ones(2, 1, 64, 256)
    
    with torch.no_grad():
        noise_pred, noise_gt = model(x, condition, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Noise pred shape: {noise_pred.shape}")
    print(f"Noise GT shape: {noise_gt.shape}")
    
    # Test sampling
    with torch.no_grad():
        samples = model.sample(condition, mask, num_steps=10, cfg_scale=1.0)
    
    print(f"Generated samples shape: {samples.shape}")
    print("✓ Diffusion model test passed!")


if __name__ == "__main__":
    test_diffusion()

