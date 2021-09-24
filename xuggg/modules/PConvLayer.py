from modules.partialconv import PartialConv2d
import torch.nn as nn
import torch.nn.functional as F
PartialConv = PartialConv2d


class PConvLayer(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 bn=True,
                 sample='none-3',
                 activ='relu',
                 conv_bias=False,
                 deconv=False):
        super().__init__()
        if sample == 'down-0':
            self.conv = PartialConv(in_ch,
                                    out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=conv_bias,
                                    multi_channel=True)
        elif sample == 'down-1':
            self.conv = PartialConv(in_ch,
                                    out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=2,
                                    dilation=2,
                                    bias=conv_bias,
                                    multi_channel=True)
        elif sample == 'down-2':
            self.conv = PartialConv(in_ch,
                                    out_ch,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=conv_bias,
                                    multi_channel=True)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch,
                                    out_ch,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    multi_channel=True,
                                    bias=False)
        elif sample == 'down-4':
            self.conv = PartialConv(in_ch,
                                    out_ch,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3,
                                    multi_channel=True,
                                    bias=False)
        elif sample == 'down-5':
            self.conv = PartialConv(in_ch,
                                    out_ch,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2,
                                    multi_channel=True,
                                    bias=False)
        else:
            self.conv = PartialConv(in_ch,
                                    out_ch,
                                    3,
                                    1,
                                    1,
                                    bias=conv_bias,
                                    multi_channel=True)
        if deconv:
            self.deconv = nn.ConvTranspose2d(out_ch,
                                             out_ch,
                                             4,
                                             2,
                                             1,
                                             bias=conv_bias)
        else:
            self.deconv = None
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.GELU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if self.deconv is not None:
            h = self.deconv(h)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        h_mask = F.interpolate(h_mask, size=h.size()[2:])
        return h, h_mask
