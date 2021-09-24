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
from math import log
from modules.PConvLayer import PConvLayer
from modules.partialconv import PartialConv2d
from modules.involution import Involution


class ContentTrans(nn.Module):
    def __init__(self, **kwargs):
        super(ContentTrans, self).__init__()
        self.size = kwargs['size']
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.first = kwargs.get('first', False)
        self.times = int(log(self.size, 2)) - 4
        self.conv_list = []
        if self.size in [128, 256]:
            self.conv_list.append(
                PConvLayer(self.in_channels,
                           self.out_channels // 2,
                           sample='down-5',
                           activ='relu'))
        else:
            self.conv_list.append(
                PConvLayer(self.in_channels,
                           self.out_channels // 2,
                           activ='relu'))

        self.conv_list.append(
            PConvLayer(self.out_channels // 2,
                       self.out_channels // 2,
                       activ='relu'))

        if self.first:
            self.conv_list.append(
                PConvLayer(self.out_channels // 2,
                           self.out_channels,
                           activ='relu',
                           sample='down-2'))
        else:
            self.conv_list.append(
                PConvLayer(self.out_channels // 2,
                           self.out_channels,
                           activ='relu',
                           sample='down-0'))

        if self.size in [128, 256]:
            self.conv_list.append(
                PConvLayer(self.out_channels,
                           self.out_channels * 2,
                           activ='relu',
                           sample='down-2'))
        else:
            self.conv_list.append(
                PConvLayer(self.out_channels,
                           self.out_channels * 2,
                           activ='relu',
                           sample='down-0'))

        self.conv_list.append(
            PConvLayer(self.out_channels * 2,
                       self.out_channels * 2,
                       activ='relu',
                       sample='down-1'))
        self.conv_list.append(
            PConvLayer(self.out_channels * 2,
                       self.out_channels * 2,
                       activ='relu',
                       sample='down-1'))

        for i in range(len(self.conv_list)):
            layer_name = f'enc_{i + 1}'
            setattr(self, layer_name, self.conv_list[i])

        self.dec_1 = PConvLayer(self.out_channels * 4,
                                self.out_channels * 2,
                                activ='leaky',
                                deconv='True')

        self.in_channels = self.out_channels * 2
        for i in range(1, self.times):
            layer_name = f'dec_{i+1}'
            block = PConvLayer(self.in_channels,
                               self.out_channels,
                               activ='leaky',
                               deconv=True)
            self.in_channels = self.out_channels
            setattr(self, layer_name, block)

        self.pconv = PartialConv2d(in_channels=self.out_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   multi_channel=True,
                                   bias=False)
        self.invo = Involution(self.out_channels, kernel_size=3, stride=1)

    def forward(self, feature, mask):
        for i in range(len(self.conv_list)):
            feature, mask = getattr(self, f'enc_{i + 1}')(feature, mask)
            if i == len(self.conv_list) - 2:
                feature_ = feature
                mask_ = mask

        feature = torch.cat([feature_, feature], dim=1)
        mask = torch.cat([mask_, mask], dim=1)
        for i in range(self.times):
            feature, mask = getattr(self, f'dec_{i+1}')(feature, mask)

        feature, mask = self.pconv(feature, mask)
        feature = self.invo(feature)

        return feature, mask


class ContentTransposed(nn.Module):
    def __init__(self, **kwargs):
        super(ContentTransposed, self).__init__()
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.size = kwargs['size']
        self.first = True
        self.tconv1 = ContentTrans(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   size=self.size,
                                   first=self.first)
        self.tconv2 = ContentTrans(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   size=self.size)
        self.conv1 = PConvLayer(self.out_channels * 2,
                                self.out_channels,
                                activ='relu',
                                sample='down-0')
        self.invo = Involution(self.out_channels, kernel_size=3, stride=1)

    def forward(self, input1, input2, mask1, mask2):
        output1, mask1 = self.tconv1(input1, mask1)
        output2, mask2 = self.tconv2(input2, mask2)

        x1 = torch.cat([output1, output2], dim=1)
        m1 = torch.cat([mask1, mask2], dim=1)
        output, mask = self.conv1(x1, m1)
        output = self.invo(output)

        return output, mask


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
        self.return_mask = True
        self.bias = False
        self.mask_bias = None
        self.size = kwargs['size']
        self.return_att = kwargs.get('return_att', True)
        if self.size in [128, 256]:
            self.size1 = 64
            self.size2 = 32
        else:
            self.size1 = 32
            self.size2 = 16

        super(ContentPartialConv2d, self).__init__()
        self.pconv = PartialConv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   dilation=self.dilation,
                                   groups=self.groups,
                                   multi_channel=True)
        self.merge = ContentTransposed(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       size=self.size)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def pretreat_input(self, input, mask):
        output1 = F.adaptive_avg_pool2d(input, self.size1)
        output2 = F.adaptive_avg_pool2d(input, self.size2)
        mask1 = F.adaptive_avg_pool2d(mask, self.size1)
        mask2 = F.adaptive_avg_pool2d(mask, self.size2)
        output1 = output1 / (mask1 + 1e-7)
        output2 = output2 / (mask2 + 1e-7)
        mask1 = F.interpolate(mask, size=[self.size1, self.size1])
        mask2 = F.interpolate(mask, size=[self.size2, self.size2])

        return output1, output2, mask1, mask2

    def forward(self, input, mask):
        output1, output2, mask1, mask2 = self.pretreat_input(input, mask)
        output, mask_ = self.pconv(input)
        output_, mask_o = self.merge(output1, output2, mask1, mask2)

        mask = mask[:, 0:1]
        masks = mask_
        mask_ = mask_[:, 0:1]
        mask = F.interpolate(
            mask,
            mask_.size()[2:]) if mask.shape[2:] != mask_.shape[2:] else mask
        m = torch.abs(mask_ - mask)

        output = output * mask + output_ * m
        output = F.relu(self.bn1(output), inplace=True)
        output_ = F.relu(self.bn2(output), inplace=True)

        return output, masks, output_, mask_o
