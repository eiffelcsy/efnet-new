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


class InlineTrackerBlock(nn.Module):
    """
    Inline feature tracking block that computes optical flow between image and event features,
    then warps the image features using the computed flow.
    
    Args:
        channels (int): Number of channels in input feature maps
    """
    def __init__(self, channels):
        super(InlineTrackerBlock, self).__init__()
        # Flow estimation network: concatenated features -> 2-channel flow field
        self.flow_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True)
        self.flow_relu = nn.LeakyReLU(0.2, inplace=False)
        self.flow_conv2 = nn.Conv2d(channels, 2, kernel_size=3, padding=1, bias=True)
        
    def forward(self, img_feat, event_feat):
        """
        Args:
            img_feat (Tensor): Image branch features [B, C, H, W]
            event_feat (Tensor): Event branch features [B, C, H, W]
            
        Returns:
            Tensor: Warped image features
        """
        # Concatenate features along channel dimension
        concat_feat = torch.cat([img_feat, event_feat], dim=1)
        
        # Estimate flow field (x, y displacements)
        flow = self.flow_conv1(concat_feat)
        flow = self.flow_relu(flow)
        flow = self.flow_conv2(flow)
        
        # Create sampling grid for warping
        B, _, H, W = img_feat.size()
        device = img_feat.device
        
        # Create normalized 2D grid [-1, 1]
        xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1).float() / (W-1) * 2 - 1
        yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W).float() / (H-1) * 2 - 1
        
        # Stack and reshape to [B, H, W, 2]
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        grid = grid.permute(0, 2, 3, 1)
        
        # Scale flow to normalized coordinates [-1, 1]
        flow_x = flow[:, 0, :, :] / ((W-1) / 2)
        flow_y = flow[:, 1, :, :] / ((H-1) / 2)
        flow_scaled = torch.stack([flow_x, flow_y], dim=-1)
        
        # Add flow to grid and perform warping
        grid_flow = grid + flow_scaled.permute(0, 2, 3, 1)
        warped_feat = F.grid_sample(img_feat, grid_flow, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped_feat


class EFNet_track_fusion(nn.Module):
    def __init__(
        self,
        in_chn=3,
        ev_chn=6,
        wf=64,
        depth=3,
        fuse_before_downsample=True,
        relu_slope=0.2,
        num_heads=[1, 2, 4],
        use_tracking=False,
    ):
        super(EFNet_track_fusion, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.use_tracking = use_tracking
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

            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    num_heads=self.num_heads[i],
                    use_tracking=self.use_tracking,
                )
            )
            self.down_path_2.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    use_emgc=downsample,
                    use_tracking=self.use_tracking,
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
        ev_features = []
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth - 1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                    ev_features.append(e1_up)
                else:
                    ev.append(e1)
                    ev_features.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)
                ev_features.append(e1)

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
                    event_feat=ev_features[i] if self.use_tracking else None,
                )
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor=0.5**i))

            else:
                x1 = down(
                    x1,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample,
                    event_feat=ev_features[i] if self.use_tracking else None,
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
                    x2, x2_up = down(x2, encs[i], decs[-i - 1], mask=masks[i], event_feat=ev_features[i] if self.use_tracking else None)
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i - 1], event_feat=ev_features[i] if self.use_tracking else None)
                blocks.append(x2_up)
            else:
                x2 = down(x2, event_feat=ev_features[i] if self.use_tracking else None)

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
        self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None, use_tracking=False
    ):  # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.use_tracking = use_tracking

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
            
        # Initialize feature tracker if needed
        if self.use_tracking:
            self.feature_tracker = InlineTrackerBlock(out_size)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(
                out_size,
                num_heads=self.num_heads,
                ffn_expansion_factor=4,
                bias=False,
                LayerNorm_type="WithBias",
            )

    def forward(
        self,
        x,
        enc=None,
        dec=None,
        mask=None,
        event_filter=None,
        merge_before_downsample=True,
        event_feat=None,  # Added parameter for event features
    ):
        out = self.conv_1(x)

        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.relu_2(out)
        out = out + self.identity(x)

        if enc is not None and dec is not None and self.use_emgc:
            # Skip existing EMGC code since we're just adding feature tracking
            if mask is not None:
                out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
                out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
                out = out + out_enc + out_dec

        if self.num_heads is not None and event_filter is not None:
            # Skip existing transformer code
            if merge_before_downsample:
                out = self.image_event_transformer(out, event_filter)

        # Apply feature tracking if enabled and event features are provided
        if self.use_tracking and event_feat is not None:
            out = self.feature_tracker(out, event_feat)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample and self.num_heads is not None and event_filter is not None:
                out_down = self.image_event_transformer(out_down, event_filter)
            return out_down, out
        else:
            if not merge_before_downsample and self.num_heads is not None and event_filter is not None:
                out = self.image_event_transformer(out, event_filter)
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
    model = EFNet_track_fusion()
    outs = model(image, event)
    for i, o in enumerate(outs, start=1):
        print(f"Output {i} shape: {o.shape}")
    print("Test forward pass done!")
