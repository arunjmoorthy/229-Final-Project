"""
UNet architecture with circular padding for range-view translation.
Handles 360° azimuth wrapping properly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CircularPad2d(nn.Module):
    """Circular padding on width (azimuth) dimension."""
    
    def __init__(self, pad_width: int, pad_height: int = 0):
        super().__init__()
        self.pad_width = pad_width
        self.pad_height = pad_height
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            Padded tensor with circular padding on W
        """
        # Circular padding on width (azimuth)
        if self.pad_width > 0:
            x = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode='circular')
        
        # Regular padding on height if needed
        if self.pad_height > 0:
            x = F.pad(x, (0, 0, self.pad_height, self.pad_height), mode='replicate')
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with circular padding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_circular: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_circular = use_circular
        
        if use_circular:
            self.pad1 = CircularPad2d(pad_width=1, pad_height=1)
            self.pad2 = CircularPad2d(pad_width=1, pad_height=1)
            padding = 0
        else:
            padding = 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=padding)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=padding)
        self.bn2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        if self.use_circular:
            out = self.pad1(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        
        out = self.bn1(out)
        out = F.silu(out)
        
        if self.dropout:
            out = self.dropout(out)
        
        if self.use_circular:
            out = self.pad2(out)
            out = self.conv2(out)
        else:
            out = self.conv2(out)
        
        out = self.bn2(out)
        out = out + residual
        out = F.silu(out)
        
        return out


class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)  # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        
        # Scaled dot-product attention
        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        out = self.proj(out)
        return out + residual


class DownBlock(nn.Module):
    """Downsampling block with residual blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_circular: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                use_circular=use_circular,
                dropout=dropout,
            )
            for i in range(num_res_blocks)
        ])
        
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        for block in self.res_blocks:
            x = block(x)
        
        if self.attention:
            x = self.attention(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_circular: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                out_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                use_circular=use_circular,
                dropout=dropout,
            )
            for i in range(num_res_blocks)
        ])
        
        self.attention = AttentionBlock(out_channels) if use_attention else None
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        for block in self.res_blocks:
            x = block(x)
        
        if self.attention:
            x = self.attention(x)
        
        return x


class RangeViewUNet(nn.Module):
    """
    UNet for range-view translation with circular padding.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.1,
        use_circular_padding: bool = True,
    ):
        """
        Args:
            in_channels: Input channels (e.g., range, intensity, mask, beam_angle)
            out_channels: Output channels (e.g., range, intensity, mask)
            base_channels: Base number of channels
            channel_multipliers: Channel multiplier for each resolution
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions to apply self-attention
            dropout: Dropout probability
            use_circular_padding: Use circular padding for azimuth dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_circular_padding = use_circular_padding
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        current_res = 64  # Assuming input is 64xW
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            
            self.down_blocks.append(DownBlock(
                ch, out_ch,
                num_res_blocks=num_res_blocks,
                use_attention=use_attn,
                use_circular=use_circular_padding,
                dropout=dropout,
            ))
            
            ch = out_ch
            current_res = current_res // 2
        
        # Middle
        self.mid_block1 = ResidualBlock(ch, ch, use_circular=use_circular_padding, dropout=dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, use_circular=use_circular_padding, dropout=dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            skip_ch = out_ch
            use_attn = current_res in attention_resolutions
            
            self.up_blocks.append(UpBlock(
                ch, out_ch, skip_ch,
                num_res_blocks=num_res_blocks,
                use_attention=use_attn,
                use_circular=use_circular_padding,
                dropout=dropout,
            ))
            
            ch = out_ch
            current_res = current_res * 2
        
        # Output
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            mask: (B, 1, H, W) optional mask
            
        Returns:
            (B, out_channels, H, W) output tensor
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask
        
        # Initial conv
        x = self.conv_in(x)
        
        # Encoder
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)
        
        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        # Decoder
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip)
        
        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        # Apply mask to output if provided
        if mask is not None:
            x = x * mask
        
        return x


def test_unet():
    """Test the UNet architecture."""
    model = RangeViewUNet(
        in_channels=4,
        out_channels=3,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8],
        use_circular_padding=True,
    )
    
    # Test forward pass
    x = torch.randn(2, 4, 64, 1024)
    mask = torch.ones(2, 1, 64, 1024)
    
    with torch.no_grad():
        y = model(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    assert y.shape == (2, 3, 64, 1024), f"Unexpected output shape: {y.shape}"
    print("✓ UNet test passed!")


if __name__ == "__main__":
    test_unet()

