import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RangeNetSegmentation(nn.Module):
    """
    Range-view semantic segmentation model.
    Similar to RangeNet++ but simplified.
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # range, intensity, mask
        num_classes: int = 19,  # SemanticKITTI classes
        base_channels: int = 32,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_encoder_block(base_channels * 4, base_channels * 8)
        self.enc5 = self._make_encoder_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.dec5 = self._make_decoder_block(base_channels * 16, base_channels * 8)
        self.dec4 = self._make_decoder_block(base_channels * 16, base_channels * 4)  # 16 = 8 + 8 (skip)
        self.dec3 = self._make_decoder_block(base_channels * 8, base_channels * 2)
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels)
        self.dec1 = self._make_decoder_block(base_channels * 2, base_channels)
        
        # Output
        self.head = nn.Conv2d(base_channels, num_classes, 1)
        
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) range view input
            mask: (B, 1, H, W) optional mask
            
        Returns:
            (B, num_classes, H, W) logits
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))
        
        # Decoder with skip connections
        dec5 = self.dec5(F.interpolate(enc5, scale_factor=2, mode='bilinear', align_corners=False))
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        dec4 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False)
        
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec3 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False)
        
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec2 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False)
        
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        
        # Output
        logits = self.head(dec1)
        
        # Apply mask if provided
        if mask is not None:
            logits = logits * mask
        
        return logits


class SegmentationTrainer:
    """Trainer for segmentation model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_classes: int = 19,
        learning_rate: float = 0.001,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        
        # Loss (weighted cross-entropy to handle class imbalance)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move to device
            range_view = batch['range'].to(self.device)
            intensity = batch['intensity'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Prepare input
            x = torch.stack([range_view, intensity, mask.float()], dim=1)
            
            # Forward
            logits = self.model(x, mask.unsqueeze(1))
            
            # Compute loss (only on valid pixels)
            labels_masked = labels.clone()
            labels_masked[~mask] = -1  # Ignore invalid pixels
            
            loss = self.criterion(logits, labels_masked)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate and compute mIoU."""
        self.model.eval()
        
        # Per-class intersection and union
        intersection = torch.zeros(self.num_classes, device=self.device)
        union = torch.zeros(self.num_classes, device=self.device)
        
        for batch in self.val_loader:
            range_view = batch['range'].to(self.device)
            intensity = batch['intensity'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Prepare input
            x = torch.stack([range_view, intensity, mask.float()], dim=1)
            
            # Forward
            logits = self.model(x, mask.unsqueeze(1))
            preds = logits.argmax(dim=1)
            
            # Compute IoU per class
            for cls in range(self.num_classes):
                pred_mask = (preds == cls) & mask
                gt_mask = (labels == cls) & mask
                
                intersection[cls] += (pred_mask & gt_mask).sum()
                union[cls] += (pred_mask | gt_mask).sum()
        
        # Compute mIoU
        iou_per_class = intersection / (union + 1e-6)
        miou = iou_per_class.mean()
        
        return miou.item(), iou_per_class.cpu().numpy()
    
    def train(self, num_epochs: int = 50):
        """Train the model."""
        best_miou = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            miou, iou_per_class = self.validate()
            
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, mIoU: {miou:.4f}")
            
            if miou > best_miou:
                best_miou = miou
                print(f"New best mIoU: {best_miou:.4f}")
            
            self.scheduler.step()
        
        return best_miou

