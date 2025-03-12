"""
EFNet_modified
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
      }
"""

import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import (
    EventImage_ChannelAttentionTransformerBlock,
    ChannelAttentionBlock,
)
from torch.nn import functional as F


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class FrameAttentionFusion(nn.Module):
    """
    Applies self-attention on non-overlapping windows of a 4D feature map (B, C, H, W).
    The feature map is partitioned into windows of size window_size x window_size,
    and self-attention is applied within each window, then the outputs are merged back.
    """
    def __init__(self, in_channels, window_size=8, num_heads=4, dropout=0.0):
        super(FrameAttentionFusion, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.num_heads = num_heads
        
        # Multi-head attention expects input shape (B, L, C) with batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        self.norm2 = nn.LayerNorm(in_channels)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size
        
        # For very small feature maps, use global attention instead of windowed attention
        if H <= ws or W <= ws:
            # Fall back to global attention for small feature maps
            # Flatten spatial dimensions: (B, L, C) where L = H * W
            x_flat = x.view(B, C, H * W).transpose(1, 2)
            
            # Apply self-attention globally
            residual = x_flat
            attn_out, _ = self.attention(x_flat, x_flat, x_flat)
            x_attn = self.norm1(residual + attn_out)
            
            # FFN
            ffn_out = self.ffn(x_attn)
            x_ffn = self.norm2(x_attn + ffn_out)
            
            # Reshape back
            return x_ffn.transpose(1, 2).view(B, C, H, W)

        # Pad if needed to make H and W multiples of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W

        # Partition into windows
        num_windows_h = H_padded // ws
        num_windows_w = W_padded // ws
        
        # Reshape to (B, C, num_windows_h, ws, num_windows_w, ws)
        x_windows = x.view(B, C, num_windows_h, ws, num_windows_w, ws)
        # Permute to (B, num_windows_h, num_windows_w, ws, ws, C)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        # Flatten each window: shape (B * num_windows_h * num_windows_w, ws*ws, C)
        x_windows = x_windows.view(B * num_windows_h * num_windows_w, ws * ws, C)

        # Save residual for skip connection
        residual = x_windows

        # Apply multi-head self-attention within each window
        attn_out, _ = self.attention(x_windows, x_windows, x_windows)
        x_attn = self.norm1(residual + attn_out)

        # Apply feed-forward network with skip connection
        ffn_out = self.ffn(x_attn)
        x_ffn = self.norm2(x_attn + ffn_out)

        # Reshape back to original spatial layout:
        # (B, num_windows_h, num_windows_w, ws, ws, C)
        x_windows = x_ffn.view(B, num_windows_h, num_windows_w, ws, ws, C)
        # Permute to (B, C, num_windows_h, ws, num_windows_w, ws)
        x_windows = x_windows.permute(0, 5, 1, 3, 2, 4).contiguous()
        # Merge windows: (B, C, H, W)
        x_out = x_windows.view(B, C, H_padded, W_padded)

        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :, :H, :W]
        return x_out


class EFNet_frame_att_fusion(nn.Module):
    def __init__(
        self,
        in_chn=3,
        ev_chn=6,
        wf=64,
        depth=3,
        fuse_before_downsample=True,
        relu_slope=0.2,
        num_heads=[1, 2, 4],
        memory_efficient=False,
    ):
        super(EFNet_frame_att_fusion, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.memory_efficient = memory_efficient
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i + 1) < depth else False
            
            # In memory efficient mode, only apply attention at the lowest resolution
            use_heads = None
            if not memory_efficient or i == depth - 1:
                use_heads = self.num_heads[i]

            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    num_heads=use_heads,
                )
            )
            self.down_path_2.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    use_emgc=downsample,
                )
            )
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(
                    UNetEVConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope)
                )

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            prev_channels = (2**i) * wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)

        self.fine_fusion = BidirectionalFrameFusionBlock(channels=wf)

        self.coarse_map = nn.Conv2d(3, wf, kernel_size=1, padding=0)
        self.coarse_unmap = nn.Conv2d(wf, 3, kernel_size=3, padding=1)

        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, mask=None):
        image = x

        # EVencoder
        ev = []
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth - 1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)

        # stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:

                x1, x1_up = down(
                    x1,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample,
                )
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor=0.5**i))

            else:
                x1 = down(
                    x1,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample,
                )

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)
        
        sam_feature, out_1 = self.sam12(x1, image)

        # stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i - 1], mask=masks[i])
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))
        
        out_2 = self.last(x2)
        out_2 = out_2 + image

        out_2_map = self.coarse_map(out_2)

        fused_coarse, fused_feat = self.fine_fusion(out_2_map, x2)
        out_3 = self.last(fused_feat)
        out_3 = out_3 + image

        return [out_1, out_3]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain("leaky_relu", 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(
        self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None
    ):  # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(
                out_size,
                num_heads=self.num_heads,
                ffn_expansion_factor=4,
                bias=False,
                LayerNorm_type="WithBias",
            )
            
        self.use_frame_attention = self.num_heads is not None
        if self.use_frame_attention:
            # Adjust window size based on expected feature map size
            # For deeper layers (with downsampling), use smaller windows
            # For shallower layers, use larger windows
            window_size = 4 if downsample else 8
            
            self.frame_attention = FrameAttentionFusion(
                in_channels=out_size,
                window_size=window_size,
                num_heads=min(4, out_size // 16),  # Ensure num_heads doesn't exceed feature dimension
                dropout=0.0
            )

    def forward(
        self,
        x,
        enc=None,
        dec=None,
        mask=None,
        event_filter=None,
        merge_before_downsample=True,
    ):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
            out = out + out_enc + out_dec

        if event_filter is not None and merge_before_downsample:
            # Apply event-image fusion
            out = self.image_event_transformer(out, event_filter)
            # Apply spatial attention after event-image fusion
            if self.use_frame_attention:
                out = self.frame_attention(out)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.image_event_transformer(out_down, event_filter)
                # Apply spatial attention after event-image fusion for non-merge_before_downsample case
                if self.use_frame_attention:
                    out_down = self.frame_attention(out_down)

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)
                # Apply spatial attention after event-image fusion for non-downsample case
                if self.use_frame_attention:
                    out = self.frame_attention(out)
                return out


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if self.downsample:

            out_down = self.downsample(out)

            if not merge_before_downsample:

                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2, bias=True
        )
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        c = x.shape[1] // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        return x1 * x2


class FusionSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionSubBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sg = SimpleGate()  # optional
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        return x


class BidirectionalFrameFusionBlock(nn.Module):
    def __init__(self, channels):
        super(BidirectionalFrameFusionBlock, self).__init__()

        half = channels // 2

        self.forward_block = FusionSubBlock(
            in_channels=half + channels, out_channels=channels
        )

        self.backward_block = FusionSubBlock(
            in_channels=half + channels, out_channels=channels
        )

    def forward(self, f_i, f_ip1):
        B, C, H, W = f_i.shape
        half = C // 2

        fa_i = f_i[:, :half, :, :]
        fb_i = f_i[:, half:, :, :]

        forward_in = torch.cat([fa_i, f_ip1], dim=1)
        f_ip1_new = self.forward_block(forward_in)
        backward_in = torch.cat([f_ip1_new, fb_i], dim=1)
        f_i_new = self.backward_block(backward_in)

        return f_i_new, f_ip1_new


if __name__ == "__main__":
    # Quick test
    B, C, H, W = 1, 3, 256, 256
    image = torch.randn(B, C, H, W)
    event = torch.randn(B, 6, H, W)
    
    # Test regular mode
    print("Testing regular mode...")
    model = EFNet_frame_att_fusion()
    outs = model(image, event)
    for i, o in enumerate(outs, start=1):
        print(f"Output {i} shape: {o.shape}")  # [B,3,H,W] for out_1/out_3
    
    # Test memory-efficient mode
    print("\nTesting memory-efficient mode...")
    model_efficient = EFNet_frame_att_fusion(memory_efficient=True)
    outs_efficient = model_efficient(image, event)
    for i, o in enumerate(outs_efficient, start=1):
        print(f"Output {i} shape: {o.shape}")  # [B,3,H,W] for out_1/out_3
    
    print("\nTest forward pass done!")
