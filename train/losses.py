import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedL1Loss(nn.Module):
    """L1 loss with masking for invalid pixels."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) predictions
            target: (B, C, H, W) targets
            mask: (B, 1, H, W) or (B, C, H, W) boolean mask
            
        Returns:
            Scalar loss
        """
        loss = F.l1_loss(pred, target, reduction='none')
        
        if mask is not None:
            mask = mask.float()
            loss = loss * mask
            
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-6)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss


class MaskedMSELoss(nn.Module):
    """MSE loss with masking for invalid pixels."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction='none')
        
        if mask is not None:
            mask = mask.float()
            loss = loss * mask
            
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-6)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained features.
    For range-view, we use a simple multi-scale L1 loss.
    """
    
    def __init__(self, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Multi-scale L1 loss."""
        total_loss = 0.0
        
        for scale in self.scales:
            if scale > 1:
                pred_scaled = F.avg_pool2d(pred, scale)
                target_scaled = F.avg_pool2d(target, scale)
                mask_scaled = F.avg_pool2d(mask.float(), scale) > 0.5 if mask is not None else None
            else:
                pred_scaled = pred
                target_scaled = target
                mask_scaled = mask
            
            loss = F.l1_loss(pred_scaled, target_scaled, reduction='none')
            
            if mask_scaled is not None:
                mask_scaled = mask_scaled.float()
                loss = (loss * mask_scaled).sum() / (mask_scaled.sum() + 1e-6)
            else:
                loss = loss.mean()
            
            total_loss += loss / len(self.scales)
        
        return total_loss


class GradientLoss(nn.Module):
    """Gradient loss to preserve edges and structure."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient loss in both horizontal and vertical directions.
        """
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Compute losses
        loss_dx = F.l1_loss(pred_dx, target_dx, reduction='none')
        loss_dy = F.l1_loss(pred_dy, target_dy, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            
            loss_dx = (loss_dx * mask_dx.float()).sum() / (mask_dx.sum() + 1e-6)
            loss_dy = (loss_dy * mask_dy.float()).sum() / (mask_dy.sum() + 1e-6)
        else:
            loss_dx = loss_dx.mean()
            loss_dy = loss_dy.mean()
        
        return loss_dx + loss_dy


class CombinedLoss(nn.Module):
    """Combined loss with multiple components."""
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.0,
        perceptual_weight: float = 0.1,
        gradient_weight: float = 0.1,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.gradient_weight = gradient_weight
        
        self.l1_loss = MaskedL1Loss()
        self.l2_loss = MaskedMSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.gradient_loss = GradientLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: (B, C, H, W) predictions
            target: (B, C, H, W) targets
            mask: (B, 1, H, W) boolean mask
            return_components: If True, return dict with individual losses
            
        Returns:
            Scalar loss or dict of losses
        """
        losses = {}
        total_loss = 0.0
        
        # L1 loss
        if self.l1_weight > 0:
            l1 = self.l1_loss(pred, target, mask)
            losses['l1'] = l1
            total_loss += self.l1_weight * l1
        
        # L2 loss
        if self.l2_weight > 0:
            l2 = self.l2_loss(pred, target, mask)
            losses['l2'] = l2
            total_loss += self.l2_weight * l2
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            perceptual = self.perceptual_loss(pred, target, mask)
            losses['perceptual'] = perceptual
            total_loss += self.perceptual_weight * perceptual
        
        # Gradient loss
        if self.gradient_weight > 0:
            gradient = self.gradient_loss(pred, target, mask)
            losses['gradient'] = gradient
            total_loss += self.gradient_weight * gradient
        
        losses['total'] = total_loss
        
        if return_components:
            return losses
        return total_loss


class DiffusionLoss(nn.Module):
    """Loss for diffusion models (simple MSE on predicted noise)."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = MaskedMSELoss()
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noise_pred: Predicted noise
            noise_target: Ground truth noise
            mask: Optional mask
            
        Returns:
            Scalar loss
        """
        return self.mse_loss(noise_pred, noise_target, mask)

