"""
Main training loop for Sim2Real translation.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, Optional

from .losses import CombinedLoss, DiffusionLoss


class Trainer:
    """Trainer for range-view translation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader_syn: DataLoader,
        train_loader_real: DataLoader,
        val_loader_syn: DataLoader,
        val_loader_real: DataLoader,
        config: dict,
        output_dir: str,
        device: str = 'cuda',
        is_diffusion: bool = False,
    ):
        """
        Args:
            model: Translation model
            train_loader_syn: Training loader for synthetic data
            train_loader_real: Training loader for real data
            val_loader_syn: Validation loader for synthetic data
            val_loader_real: Validation loader for real data
            config: Configuration dictionary
            output_dir: Directory to save checkpoints and logs
            device: Device to use
            is_diffusion: Whether model is a diffusion model
        """
        self.model = model.to(device)
        self.train_loader_syn = train_loader_syn
        self.train_loader_real = train_loader_real
        self.val_loader_syn = val_loader_syn
        self.val_loader_real = val_loader_real
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        self.is_diffusion = is_diffusion
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        
        # Training config
        train_cfg = config['training']
        self.num_epochs = train_cfg['num_epochs']
        self.batch_size = train_cfg['batch_size']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg['weight_decay']
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)
        self.save_interval = train_cfg.get('save_interval', 5)
        self.eval_interval = train_cfg.get('eval_interval', 1)
        self.log_interval = train_cfg.get('log_interval', 100)
        self.use_amp = train_cfg.get('mixed_precision', True)
        
        # Loss
        if is_diffusion:
            self.criterion = DiffusionLoss()
        else:
            loss_cfg = config['loss']
            self.criterion = CombinedLoss(
                l1_weight=loss_cfg.get('l1_weight', 1.0),
                l2_weight=loss_cfg.get('l2_weight', 0.0),
                perceptual_weight=loss_cfg.get('perceptual_weight', 0.1),
                gradient_weight=loss_cfg.get('gradient_weight', 0.1),
            )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        
        # Iterate over both synthetic and real loaders
        syn_iter = iter(self.train_loader_syn)
        real_iter = iter(self.train_loader_real)
        
        num_batches = min(len(self.train_loader_syn), len(self.train_loader_real))
        print(f"  Training on {num_batches} batches...")
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {self.epoch + 1}/{self.num_epochs}")
        
        for batch_idx in pbar:
            # Get synthetic and real batches
            try:
                syn_batch = next(syn_iter)
            except StopIteration:
                syn_iter = iter(self.train_loader_syn)
                syn_batch = next(syn_iter)
            
            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(self.train_loader_real)
                real_batch = next(real_iter)
            
            # Move to device
            syn_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in syn_batch.items()}
            real_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in real_batch.items()}
            
            # Prepare input and target
            # Input: synthetic range view
            syn_input = torch.stack([
                syn_batch['range'],
                syn_batch['intensity'],
                syn_batch['mask'].float(),
                syn_batch['beam_angle'],
            ], dim=1)
            
            # Target: real range view
            real_target = torch.stack([
                real_batch['range'],
                real_batch['intensity'],
                real_batch['mask'].float(),
            ], dim=1)
            
            real_mask = real_batch['mask'].unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    if self.is_diffusion:
                        noise_pred, noise_target = self.model(real_target, syn_input, real_mask)
                        loss = self.criterion(noise_pred, noise_target, real_mask)
                    else:
                        pred = self.model(syn_input, real_mask)
                        loss = self.criterion(pred, real_target, real_mask)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.is_diffusion:
                    noise_pred, noise_target = self.model(real_target, syn_input, real_mask)
                    loss = self.criterion(noise_pred, noise_target, real_mask)
                else:
                    pred = self.model(syn_input, real_mask)
                    loss = self.criterion(pred, real_target, real_mask)
                
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            # Track loss
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            
            # Logging
            if self.global_step % self.log_interval == 0:
                self.writer.add_scalar('train/loss', loss_val, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        return {'loss': sum(epoch_losses) / len(epoch_losses)}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = []
        
        # Sample validation
        num_val_batches = min(50, min(len(self.val_loader_syn), len(self.val_loader_real)))
        
        syn_iter = iter(self.val_loader_syn)
        real_iter = iter(self.val_loader_real)
        
        for _ in tqdm(range(num_val_batches), desc="Validating", leave=False):
            try:
                syn_batch = next(syn_iter)
                real_batch = next(real_iter)
            except StopIteration:
                break
            
            # Move to device
            syn_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in syn_batch.items()}
            real_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in real_batch.items()}
            
            # Prepare input and target
            syn_input = torch.stack([
                syn_batch['range'],
                syn_batch['intensity'],
                syn_batch['mask'].float(),
                syn_batch['beam_angle'],
            ], dim=1)
            
            real_target = torch.stack([
                real_batch['range'],
                real_batch['intensity'],
                real_batch['mask'].float(),
            ], dim=1)
            
            real_mask = real_batch['mask'].unsqueeze(1)
            
            # Forward pass
            if self.is_diffusion:
                noise_pred, noise_target = self.model(real_target, syn_input, real_mask)
                loss = self.criterion(noise_pred, noise_target, real_mask)
            else:
                pred = self.model(syn_input, real_mask)
                loss = self.criterion(pred, real_target, real_mask)
            
            val_losses.append(loss.item())
        
        return {'loss': sum(val_losses) / len(val_losses) if val_losses else float('inf')}
    
    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.output_dir / 'checkpoints' / name
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Train loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self.validate()
                print(f"Epoch {epoch + 1}/{self.num_epochs} - Val loss: {val_metrics['loss']:.4f}")
                
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best.pt')
                    print(f"New best model! Val loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')
            
            # Update learning rate
            self.scheduler.step()
        
        # Save final model
        self.save_checkpoint('final.pt')
        print("Training completed!")
        
        self.writer.close()

