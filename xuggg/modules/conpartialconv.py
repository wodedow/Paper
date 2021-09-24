####################################################################
# Based on PartialConv2d
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
####################################################################

import torch
import torch.nn.functional as F
from torch import nn


class ContentPartialConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride']
        self.padding = kwargs['padding']
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)
        self.multi_channel = kwargs.get('multi_channel', True)
        self.return_mask = kwargs.get('return_mask', True)
        self.bias = False
        self.mask_bias = None
        self.ratio = None

        super(ContentPartialConv2d, self).__init__()
        self.partialconv = nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)

    def update_mask(self, mask):
        if self.multi_channel:
            weight_maskUpdater = torch.ones([
                self.out_channels, self.in_channels, self.kernel_size,
                self.kernel_size
            ],
                                            device=mask.device)
        else:
            weight_maskUpdater = torch.ones(
                [1, 1, self.kernel_size, self.kernel_size], device=mask.device)

        update_mask = F.conv2d(mask,
                               weight_maskUpdater,
                               bias=self.mask_bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups)
        update_mask = torch.clamp(update_mask, 0, 1)

        return update_mask

    def pretreat_input(self, input, mask):
        output1 = F.adaptive_avg_pool2d(input, 16)
        output2 = F.adaptive_avg_pool2d(input, 8)
        mask1 = F.adaptive_avg_pool2d(mask, 16)
        mask2 = F.adaptive_avg_pool2d(mask, 8)
        output1 = output1 / (mask1 + 1e-7)
        output2 = output2 / (mask2 + 1e-7)

        output1 = F.interpolate(output1,
                                size=[input.size(2)] * 2,
                                mode='nearest')
        output2 = F.interpolate(output2,
                                size=[input.size(2)] * 2,
                                mode='nearest')
        mask1 = F.interpolate(mask1, size=[input.size(2)] * 2, mode='nearest')
        mask2 = F.interpolate(mask2, size=[input.size(2)] * 2, mode='nearest')
        output1 = input + output1 * mask
        output2 = input + output2 * mask

        output1 = F.adaptive_avg_pool2d(output1, 16)
        output2 = F.adaptive_avg_pool2d(output2, 8)
        mask1 = F.adaptive_avg_pool2d(mask1, 16)
        mask2 = F.adaptive_avg_pool2d(mask1, 8)
        output1 = output1 / (mask1 + 1e-7)
        output2 = output2 / (mask2 + 1e-7)

        output1 = F.interpolate(output1,
                                size=[input.size(2)] * 2,
                                mode='bilinear',
                                align_corners=True)
        output2 = F.interpolate(output2,
                                size=[input.size(2)] * 2,
                                mode='bilinear',
                                align_corners=True)
        # self.ratio = nn.Parameter(torch.ones(1)).sigmoid()
        result = input + (0.7 * output1 + (0.3 * output2) * mask)
        return result

    def forward(self, input, mask):
        output = self.pretreat_input(input, mask)
        mask = self.update_mask(mask)
        output = self.partialconv(output)
        output = output * mask
        if self.return_mask:
            return output, mask
        else:
            return output
