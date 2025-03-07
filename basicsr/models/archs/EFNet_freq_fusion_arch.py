'''
EFNet_modified
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
      }
'''

import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock, ChannelAttentionBlock
from torch.nn import functional as F

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

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
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class EFNet_freq_fusion(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(EFNet_freq_fusion, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 

            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i]))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_emgc=downsample))
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope))

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels*3, prev_channels, 1, 1, 0)

        self.fine_fusion = CoarseToFineFusionModule(feat_channels=wf)

        self.last = conv3x3(prev_channels, in_chn, bias=True)

        self.fcfe = FCFE(wf)

    def forward(self, x, event, mask=None):
        image = x

        ev = []
        #EVencoder
        e1 = self.conv_ev1(event)
        e1_clone = e1.clone() # added here


        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)

        #stage 1
        x1 = self.conv_01(image)
        x1_clone = x1.clone() # added here

        output_fcfe = self.fcfe(e1_clone, x1_clone) # added here


        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:

                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)


        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        #stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature, output_fcfe], dim=1)) # added here
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i-1], mask=masks[i])
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]))

        out_2 = self.last(x2)
        out_2 = out_2 + image

        out_3 = self.fine_fusion(x2, out_1)
        out_3 = out_3 + image

        return [out_1, out_2, out_3]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None): # cat
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
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1-mask)*enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask*dec)
            out = out + out_enc + out_dec        
            
        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            out = self.image_event_transformer(out, event_filter) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)


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

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
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
            else : 
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class CoarseToFineFusionModule(nn.Module):
    def __init__(self, feat_channels=64, reduction=16):
        super(CoarseToFineFusionModule, self).__init__()
        
        self.coarse_map = nn.Conv2d(3, feat_channels, kernel_size=1, padding=0)

        self.fusion_block_1 = ChannelAttentionBlock(n_feat=feat_channels*2, reduction=reduction, act=nn.ReLU(True))

        self.fusion_block_2 = ChannelAttentionBlock(n_feat=feat_channels*2, reduction=reduction, act=nn.ReLU(True))

        self.final_conv = nn.Conv2d(feat_channels*2, 3, kernel_size=3, padding=1, bias=True)


    def forward(self, x2, out_1):
        out_1_map = self.coarse_map(out_1)

        fused = torch.cat([x2, out_1_map], dim=1)

        fused = self.fusion_block_1(fused)
        fused = self.fusion_block_2(fused)

        out_3 = self.final_conv(fused)

        return out_3


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
        self.conv1x1_3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.conv1x1_6 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_7 = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_attention = nn.Conv2d(channels, channels, kernel_size=1)
        self.filter_generation = nn.Conv2d(channels, channels, kernel_size=1)
        self.cross_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=1)
        self.geglu = nn.GELU()

    def forward(self, event_features, image_features):
        #print("event_features, image_features", event_features.size(), image_features.size())
        
        device = event_features.device

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
        x7 = torch.fft.ifft2(self.conv1x1_3(F.relu(self.conv1x1_4((x5 * x6).real)))).real
        #print("need to add imag as well to x7", x7.size())
        #x_check = torch.reshape(x7, (x7.shape[0] * x7.shape[1], x7.shape[2], x7.shape[3]))
        #print("check", x_check.size())
        x8 = torch.fft.fft(torch.reshape(x7, (x7.shape[0] * x7.shape[1], x7.shape[2], x7.shape[3])))
        #print("x8", x8.size())
        x9 = self.filter_generation(self.channel_attention(x7 + x3))
        x9 = torch.reshape(x9, (x9.shape[0]*x9.shape[1], x9.shape[2], x9.shape[3]))
        #print("x9", x9.size())
        x10 = torch.reshape(torch.fft.ifft(x8 * x9), x7.shape).real
        #print("x10", x10.size())

        x10_batch_size, x10_channels, x10_height, x10_width = x10.shape
        x10_reshaped = x10.permute(0,2,3,1).reshape(x10_batch_size,x10_height*x10_width, x10_channels)
        x1_reshaped = self.conv1x1_6(x1).permute(0, 2, 3, 1).reshape(x10_batch_size, x10_height*x10_width, x10_channels)

        x_cross, _  = self.cross_attention(x10_reshaped, x1_reshaped, x1_reshaped)
        #print("x_cross", x_cross.size())
        x_cross = x_cross.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        #print("x_cross", x_cross.size())

        x11 = self.conv1x1_7(self.geglu(self.dw_conv_3(x_cross)))
        #print("x11", x11.size())
        output = x11 + x2

        return output


if __name__ == "__main__":
    pass
