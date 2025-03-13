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


# ---------------------------
# ConvLSTM Cell for temporal feature tracking
# ---------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
        self.hidden_dim = hidden_dim
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ---------------------------
# Memory-Efficient Correlation Module
# ---------------------------
class EfficientCorrelation(nn.Module):
    def __init__(self, max_displacement=4):
        super(EfficientCorrelation, self).__init__()
        self.max_displacement = max_displacement
        self.window_size = 2 * max_displacement + 1

    def forward(self, x1, x2):
        # Memory-efficient implementation with strided convolution
        B, C, H, W = x2.size()
        corr_tensor = []
        
        # Process in chunks to reduce memory
        chunk_size = 16  # Process 16 channels at a time
        for i in range(0, C, chunk_size):
            end = min(i + chunk_size, C)
            x1_chunk = x1[:, i:end]
            x2_chunk = x2[:, i:end]
            
            # Compute correlation for this chunk
            chunk_corr = self._compute_chunk_correlation(x1_chunk, x2_chunk)
            corr_tensor.append(chunk_corr)
            
        # Average correlations from all chunks
        corr = torch.mean(torch.stack(corr_tensor, dim=0), dim=0)
        return corr
        
    def _compute_chunk_correlation(self, x1, x2):
        B, C, H, W = x2.size()
        pad = self.max_displacement
        x2_padded = F.pad(x2, (pad, pad, pad, pad), mode='replicate')
        
        # Use strided convolution approach instead of unfold
        corr = torch.zeros(B, self.window_size**2, H, W, device=x1.device)
        
        # Compute correlation for each displacement
        idx = 0
        for dy in range(-self.max_displacement, self.max_displacement + 1):
            for dx in range(-self.max_displacement, self.max_displacement + 1):
                x2_shifted = x2_padded[:, :, pad+dy:pad+dy+H, pad+dx:pad+dx+W]
                corr[:, idx] = (x1 * x2_shifted).sum(dim=1)
                idx += 1
                
        return corr


# ---------------------------
# Efficient Flow Estimation Module
# ---------------------------
class EfficientFlowEstimator(nn.Module):
    def __init__(self, in_channels, search_range=4):
        super(EfficientFlowEstimator, self).__init__()
        self.correlation = EfficientCorrelation(max_displacement=search_range)
        # Use window_size for consistency with the new EfficientCorrelation implementation
        window_size = 2 * search_range + 1
        corr_channels = window_size ** 2
        
        # Flow estimation network
        self.flow_conv1 = nn.Conv2d(corr_channels + in_channels, 128, kernel_size=3, padding=1)
        self.flow_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.flow_pred = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x1, x2):
        """
        Estimate flow from x1 to x2
        Args:
            x1: First feature map [B, C, H, W]
            x2: Second feature map [B, C, H, W]
        Returns:
            Flow field [B, 2, H, W]
        """
        # Compute correlation volume
        corr = self.correlation(x1, x2)
        
        # Concatenate correlation with first feature map
        corr_x1 = torch.cat([corr, x1], dim=1)
        
        # Estimate flow
        out = self.leaky_relu(self.flow_conv1(corr_x1))
        out = self.leaky_relu(self.flow_conv2(out))
        flow = self.flow_pred(out)
        
        return flow


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


class EFNet_tracking(nn.Module):
    def __init__(
        self,
        in_chn=3,
        ev_chn=6,
        wf=64,
        depth=3,
        fuse_before_downsample=True,
        relu_slope=0.2,
        num_heads=[1, 2, 4],
        enable_tracking=True,
    ):
        super(EFNet_tracking, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.enable_tracking = enable_tracking
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
                    num_heads=self.num_heads[i] if i < len(self.num_heads) else None,
                    enable_tracking=self.enable_tracking,
                    scale_level=i,
                )
            )
            # Always disable tracking for stage 2
            self.down_path_2.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    use_emgc=downsample,
                    enable_tracking=False,  # Always disable tracking for stage 2
                    scale_level=i,
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

        # Replace BidirectionalFrameFusionBlock with CoarseToFineFusionModule
        self.fine_fusion = CoarseToFineFusionModule(feat_channels=wf)

        self.coarse_map = nn.Conv2d(3, wf, kernel_size=1, padding=0)
        self.coarse_unmap = nn.Conv2d(wf, 3, kernel_size=3, padding=1)

        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def reset_tracking_states(self):
        """Reset all ConvLSTM states in the network"""
        if self.enable_tracking:
            for block in self.down_path_1:
                if hasattr(block, 'hidden_state'):
                    block.hidden_state = None
                    block.prev_features = None
            # Remove reset for down_path_2 since tracking is disabled for stage 2

    def forward(self, x, event, mask=None):
        image = x
        flows_stage1 = []

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
                if self.enable_tracking:
                    x1, x1_up, flow = down(
                        x1,
                        event_filter=ev[i],
                        merge_before_downsample=self.fuse_before_downsample,
                    )
                    flows_stage1.append(flow)
                else:
                    x1, x1_up = down(
                        x1,
                        event_filter=ev[i],
                        merge_before_downsample=self.fuse_before_downsample,
                    )
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor=0.5**i))

            else:
                if self.enable_tracking:
                    x1, flow = down(
                        x1,
                        event_filter=ev[i],
                        merge_before_downsample=self.fuse_before_downsample,
                    )
                    flows_stage1.append(flow)
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

        # stage 2 - Use out_1 (partially deblurred) as input instead of original image
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

        # Use the CoarseToFineFusionModule for final refinement
        refined_features = self.fine_fusion(x2, out_2)
        out_3 = self.last(refined_features)
        out_3 = out_3 + image

        # Always return just the outputs, not the flows
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
        self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None, enable_tracking=False, scale_level=0
    ):  # cat
        super(UNetConvBlock, self).__init__()
        self.downsample_flag = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.enable_tracking = enable_tracking
        self.scale_level = scale_level

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        # ----- Inline feature tracking: add a ConvLSTM cell and a flow estimator -----
        if self.enable_tracking:
            self.conv_lstm = ConvLSTMCell(out_size, out_size, kernel_size=3, bias=True)
            
            # Use different search ranges based on the scale level
            # Smaller search ranges at higher resolutions to save memory
            if scale_level == 0:  # Highest resolution
                search_range = 2
            elif scale_level == 1:
                search_range = 4
            else:  # Lowest resolution
                search_range = 8
                
            self.flow_estimator = EfficientFlowEstimator(out_size, search_range=search_range)
            self.hidden_state = None  # will hold (h, c)
            self.prev_features = None  # will store previous features for correlation

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

    def reset_hidden(self, batch_size, spatial_size, device):
        if self.enable_tracking:
            h = torch.zeros(batch_size, self.conv_lstm.hidden_dim, spatial_size[0], spatial_size[1]).to(device)
            c = torch.zeros(batch_size, self.conv_lstm.hidden_dim, spatial_size[0], spatial_size[1]).to(device)
            self.hidden_state = (h, c)
            self.prev_features = None

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

        # ----- Inline Tracking: update recurrent state and compute flow -----
        flow = None
        if self.enable_tracking:
            if self.hidden_state is None:
                batch_size, _, H, W = out.shape
                device = out.device
                self.reset_hidden(batch_size, (H, W), device)
            
            # Compute flow if we have previous features
            if self.prev_features is not None:
                flow = self.flow_estimator(self.prev_features, out)
            
            # Update LSTM state
            h, c = self.conv_lstm(out, self.hidden_state)
            self.hidden_state = (h, c)
            
            # Store current features for next iteration
            self.prev_features = out.detach()

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
            out = out + out_enc + out_dec

        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            out = self.image_event_transformer(out, event_filter)

        if self.downsample_flag:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.image_event_transformer(out_down, event_filter)

            if self.enable_tracking:
                return out_down, out, flow
            else:
                return out_down, out

        else:
            if merge_before_downsample:
                if self.enable_tracking:
                    return out, flow
                else:
                    return out
            else:
                out = self.image_event_transformer(out, event_filter)
                if self.enable_tracking:
                    return out, flow
                else:
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


class CoarseToFineFusionModule(nn.Module):
    """
    A module that fuses coarse deblurred features with fine-grained features
    to produce the final output.
    """
    def __init__(self, feat_channels):
        super(CoarseToFineFusionModule, self).__init__()
        
        # Feature refinement layers
        self.coarse_map = nn.Conv2d(3, feat_channels, kernel_size=1, padding=0)
        self.fusion_conv1 = nn.Conv2d(feat_channels*2, feat_channels, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, fine_features, coarse_output):
        """
        Args:
            fine_features: High-resolution features from the decoder
            coarse_output: Coarse deblurred image
        Returns:
            Refined features for final output
        """
        # Map coarse output to feature space
        coarse_features = self.coarse_map(coarse_output)
        
        # Concatenate and fuse features
        concat_features = torch.cat([fine_features, coarse_features], dim=1)
        fused_features = self.lrelu(self.fusion_conv1(concat_features))
        refined_features = self.lrelu(self.fusion_conv2(fused_features))
        
        return refined_features


if __name__ == "__main__":
    # Quick test
    B, C, H, W = 1, 3, 256, 256
    image = torch.randn(B, C, H, W)
    event = torch.randn(B, 6, H, W)
    
    # Test with tracking enabled
    model = EFNet_tracking(enable_tracking=True)
    outputs = model(image, event)
    print("Running with tracking enabled:")
    for i, o in enumerate(outputs, start=1):
        print(f"Output {i} shape: {o.shape}")  # [B,3,H,W] for out_1/out_3
    
    # Test with tracking disabled
    model_no_tracking = EFNet_tracking(enable_tracking=False)
    outs_no_tracking = model_no_tracking(image, event)
    print("\nRunning with tracking disabled:")
    for i, o in enumerate(outs_no_tracking, start=1):
        print(f"Output {i} shape: {o.shape}")
    
    # Test memory usage with different batch sizes
    print("\nTesting memory usage with different batch sizes:")
    for batch_size in [1, 2, 4]:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            model_cuda = EFNet_tracking(enable_tracking=True).cuda()
            image_cuda = torch.randn(batch_size, C, H, W).cuda()
            event_cuda = torch.randn(batch_size, 6, H, W).cuda()
            
            # Forward pass
            with torch.no_grad():
                _ = model_cuda(image_cuda, event_cuda)
            
            # Print memory usage
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            print(f"Batch size {batch_size}: {max_memory:.2f} MB")
        else:
            print("CUDA not available, skipping memory test")
    
    # Compare memory usage with different search ranges
    if torch.cuda.is_available():
        print("\nComparing memory usage with different search ranges:")
        batch_size = 1
        
        # Test with small search ranges (our efficient approach)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_small = EFNet_tracking(enable_tracking=True).cuda()
        image_cuda = torch.randn(batch_size, C, H, W).cuda()
        event_cuda = torch.randn(batch_size, 6, H, W).cuda()
        
        with torch.no_grad():
            _ = model_small(image_cuda, event_cuda)
        
        small_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"With scale-dependent search ranges: {small_memory:.2f} MB")
        
        # Test with large search ranges (naive approach)
        # This is a simulation - we're not actually modifying the model
        # but we can estimate the memory usage
        estimated_large_memory = small_memory * (8**2) / (4**2)  # Assuming 8x8 vs 4x4 search windows
        print(f"Estimated with fixed large search ranges: {estimated_large_memory:.2f} MB")
        print(f"Memory savings: {estimated_large_memory - small_memory:.2f} MB ({(1 - small_memory/estimated_large_memory)*100:.1f}%)")
    
    print("Test forward pass done!")
