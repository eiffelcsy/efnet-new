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
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock
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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        conv_output = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class InlineTrackerBlock(nn.Module):

    def __init__(self, channels, lstm_hidden_dim=64):
        super(InlineTrackerBlock, self).__init__()
        self.flow_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True)
        self.flow_relu = nn.LeakyReLU(0.2, inplace=False)
        self.flow_conv2 = nn.Conv2d(channels, 2, kernel_size=3, padding=1, bias=True)
        
        self.lstm_hidden_dim = lstm_hidden_dim
        
        self.feature_encoder = nn.Conv2d(channels * 2 + 2, lstm_hidden_dim, kernel_size=3, padding=1, bias=True)
        self.feature_encoder_act = nn.LeakyReLU(0.2, inplace=False)
        
        self.conv_lstm = ConvLSTMCell(
            input_dim=lstm_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            kernel_size=3,
            bias=True
        )
        
        self.flow_decoder = nn.Conv2d(lstm_hidden_dim, 2, kernel_size=3, padding=1, bias=True)
        
        self.lstm_h = None
        self.lstm_c = None

    def forward(self, img_feat, event_feat):

        B, C, H, W = img_feat.size()
        device = img_feat.device
        
        concat_feat = torch.cat([img_feat, event_feat], dim=1)
        
        flow_features = self.flow_conv1(concat_feat)
        flow_features = self.flow_relu(flow_features)
        initial_flow = self.flow_conv2(flow_features)
        
        if self.lstm_h is None or self.lstm_c is None or self.lstm_h.size(0) != B:
            self.lstm_h = torch.zeros(B, self.lstm_hidden_dim, H, W, device=device)
            self.lstm_c = torch.zeros(B, self.lstm_hidden_dim, H, W, device=device)
            
            lstm_input_features = torch.cat([concat_feat, initial_flow], dim=1)
            lstm_input_features = self.feature_encoder(lstm_input_features)
            lstm_input_features = self.feature_encoder_act(lstm_input_features)
            
            self.lstm_h, self.lstm_c = self.conv_lstm(lstm_input_features, (self.lstm_h, self.lstm_c))
            
            flow_refinement = self.flow_decoder(self.lstm_h)
            
            flow = initial_flow + flow_refinement
        else:
            flow = initial_flow
        
        xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1).float() / (W-1) * 2 - 1
        yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W).float() / (H-1) * 2 - 1
        
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        grid = grid.permute(0, 2, 3, 1)
        
        flow_x = flow[:, 0, :, :] / ((W-1) / 2)
        flow_y = flow[:, 1, :, :] / ((H-1) / 2)
        flow_scaled = torch.stack([flow_x, flow_y], dim=-1)
        
        grid_flow = grid + flow_scaled
        warped_feat = F.grid_sample(img_feat, grid_flow, mode='bilinear', padding_mode='border', align_corners=True)
        
        output = warped_feat + img_feat

        return output

    def reset_states(self):

        self.lstm_h = None
        self.lstm_c = None


class EFNet_att_track_freq_fusion(nn.Module):
    def __init__(
        self,
        in_chn=3,
        ev_chn=6,
        wf=64,
        depth=3,
        fuse_before_downsample=True,
        relu_slope=0.2,
        num_heads=[1, 2, 4],
        use_tracking=True,
    ):
        super(EFNet_att_track_freq_fusion, self).__init__()
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

            # Only enable tracking for the first two layers
            layer_use_tracking = use_tracking and i in [0, 1] # added here
            
            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    num_heads=self.num_heads[i],
                    use_tracking=layer_use_tracking,
                )
            )
            self.down_path_2.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    use_emgc=downsample,
                    use_tracking=False,
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

        self.cat12 = nn.Conv2d(prev_channels * 3, prev_channels, 1, 1, 0)

        self.fcfe = FCFE(wf) # added here

        self.fine_fusion = BidirectionalFrameFusionBlock(channels=wf)

        self.coarse_map = nn.Conv2d(3, wf, kernel_size=1, padding=0)
        self.coarse_unmap = nn.Conv2d(wf, 3, kernel_size=3, padding=1)

        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, mask=None):
        image = x

        # EVencoder
        ev = []
        e1 = self.conv_ev1(event)
        e1_clone = e1.clone() # added here

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
        x1_clone = x1.clone() # added here

        output_fcfe = self.fcfe(e1_clone, x1_clone) # added here
        
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
        x2 = self.cat12(torch.cat([x2, sam_feature, output_fcfe], dim=1)) # added here
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

    def reset_lstm_states(self):
        for i in range(min(2, self.depth)):  # Only reset first two layers where tracking is used
            if hasattr(self.down_path_1[i], 'feature_tracker') and self.down_path_1[i].use_tracking:
                self.down_path_1[i].feature_tracker.reset_states()
            
            if hasattr(self.down_path_2[i], 'feature_tracker') and self.down_path_2[i].use_tracking:
                self.down_path_2[i].feature_tracker.reset_states()


class UNetConvBlock(nn.Module):
    def __init__(
        self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None, 
        use_tracking=False
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
            self.feature_tracker = InlineTrackerBlock(
                out_size, 
                lstm_hidden_dim=out_size,
            )

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
            if mask is not None:
                out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
                out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
                out = out + out_enc + out_dec

        if self.num_heads is not None and event_filter is not None:
            if merge_before_downsample:
                out = self.image_event_transformer(out, event_filter)

        if self.use_tracking and event_feat is not None: # added here
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

class FCFE(nn.Module):
    def __init__(self, channels):
        super(FCFE, self).__init__()
        #self.norm_event = nn.LayerNorm([channels, 256, 256])
        #self.norm_image = nn.LayerNorm([channels, 256, 256])
        self.conv1x1_1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.dw_conv_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.dw_conv_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.dw_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv1x1_real = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_imag = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1_2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_3_real = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_3_imag = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_4_real = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_4_imag = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.conv1x1_6 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_7 = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_attention = nn.Conv2d(channels, channels, kernel_size=1)
        self.filter_generation = nn.Conv2d(channels, channels, kernel_size=1)
        self.cross_attention_real = nn.MultiheadAttention(embed_dim=channels, num_heads=1)
        self.cross_attention_imag = nn.MultiheadAttention(embed_dim=channels, num_heads=1)
        self.geglu = nn.GELU()

    def forward(self, event_features, image_features):
        #print("event_features, image_features", event_features.size(), image_features.size())
        
        device = event_features.device
        #print(device)

        batch_size, channels, height, width = event_features.shape
        self.norm_event = nn.LayerNorm([channels, height, width]).to(device)
        self.norm_image = nn.LayerNorm([channels, height, width]).to(device)

        x1 = self.norm_event(event_features)
        x2 = self.norm_image(image_features)
        #print("x1, x2", x1.size(), x2.size())
        x3 = self.conv1x1_1(torch.cat([x1, x2], dim=1))
        #print("x3", x3.size())
        x4 = torch.fft.fft2(self.dw_conv_1(x3))
        #print("x4", x4.size())
        real = F.relu(self.conv1x1_real(x4.real))
        imag = F.relu(self.conv1x1_imag(x4.imag))
        #print("x4 - real and imag", real.size(), imag.size())
        x5 = self.sigmoid(self.conv1x1_5(torch.cat([real, imag], dim=1)))
        #print("x5", x5.size())
        x6 = torch.fft.fft2(self.dw_conv_2(self.conv1x1_2(x2)))
        #print("x6", x6.size())
        product = x5 * x6
        product_real = self.conv1x1_3_real(F.relu(self.conv1x1_4_real(product.real)))
        product_imag = self.conv1x1_3_imag(F.relu(self.conv1x1_4_imag(product.imag)))
        #x7 = torch.fft.ifft2(self.conv1x1_3(F.relu(self.conv1x1_4((x5 * x6).real)))).real
        x7 = torch.fft.ifft2(torch.complex(product_real, product_imag))
        #print("x7", x7.size())
        #x_check = torch.reshape(x7, (x7.shape[0] * x7.shape[1], x7.shape[2], x7.shape[3]))
        #x7 = torch.cat([x7.real, x7.imag], dim = 1)
        #print("x7_concat", x7.size())
        x8 = torch.fft.fft(torch.reshape(x7, (x7.shape[0] * x7.shape[1], x7.shape[2], x7.shape[3])))
        #print("x8", x8.size())
        x9 = self.filter_generation(self.channel_attention(x7.real + x3 + x7.imag))
        x9 = torch.reshape(x9, (x9.shape[0]*x9.shape[1], x9.shape[2], x9.shape[3]))
        #print("x9", x9.size())
        x10_real = torch.reshape(torch.fft.ifft(x8 * x9), x7.shape).real
        x10_imag = torch.reshape(torch.fft.ifft(x8 * x9), x7.shape).imag
        x10 = torch.complex(x10_real, x10_imag)
        #print("x10", x10.size())

        x10_batch_size, x10_channels, x10_height, x10_width = x10.shape
        x10_reshaped = x10.permute(0,2,3,1).reshape(x10_batch_size,x10_height*x10_width, x10_channels)
        x1_reshaped = self.conv1x1_6(x1).permute(0, 2, 3, 1).reshape(x10_batch_size, x10_height*x10_width, x10_channels)

        x_cross_real, _  = self.cross_attention_real(x10_reshaped.real, x1_reshaped, x1_reshaped)
        #print("x_cross_imag", x_cross_real.size())
        x_cross_real = x_cross_real.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        #print("x_cross", x_cross.size())

        x_cross_imag, _  = self.cross_attention_imag(x10_reshaped.imag, x1_reshaped, x1_reshaped)
        #print("x_cross_imag", x_cross_imag.size())
        x_cross_imag = x_cross_imag.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        #print("x_cross", x_cross.size())

        x11 = self.conv1x1_7(self.geglu(self.dw_conv_3(torch.complex(x_cross_real, x_cross_imag).abs())))
        #print("x11", x11.size())
        output = x11 + x2

        return output

if __name__ == "__main__":
    # Quick test
    B, C, H, W = 1, 3, 256, 256
    image = torch.randn(B, C, H, W)
    event = torch.randn(B, 6, H, W)
    model = EFNet_att_track_freq_fusion()
    outs = model(image, event)
    for i, o in enumerate(outs, start=1):
        print(f"Output {i} shape: {o.shape}")
    print("Test forward pass done!")
