import torch
import torch.nn as nn


class Involution(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels // reduction_ratio,
                      kernel_size=1,
                      bias=False), nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(in_channels=channels // reduction_ratio,
                      out_channels=kernel_size**2 * self.groups,
                      kernel_size=1,
                      stride=1,
                      bias=False)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(
            self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h,
                             w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels,
                                  self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class Involution_M(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(Involution_M, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels // reduction_ratio,
                      kernel_size=1,
                      bias=False), nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(in_channels=channels // reduction_ratio,
                      out_channels=kernel_size**2 * self.groups,
                      kernel_size=1,
                      stride=1,
                      bias=False)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x, m):
        weight = self.conv2(
            self.conv1(x if self.stride == 1 else self.avgpool(x) / (self.avgpool(m) + 1e-7)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h,
                             w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels,
                                  self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out
