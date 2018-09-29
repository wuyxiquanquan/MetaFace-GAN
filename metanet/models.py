# -*- coding:utf-8 -*-
from torch import nn
import torch
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):
    """
        VGG: To get different size features
             High-level features: Overall and Shape features
             Low-level features: Detailed features
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping.keys():
                outs.append(x)
        return outs


class MyConv2D(nn.Module):
    """
        Using in Transform Net
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size)).to(device)
        self.bias = torch.zeros(out_channels).to(device)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        return s.format(**self.__dict__)


def convLayer(in_channels, out_channels, kernel_size=3, stride=1,
              unsample=None, instance_norm=True, relu=True, trainable=False):
    layers = []
    if unsample:
        layers.append(nn.Upsample(mode="nearest", scale_factor=unsample))
    layers.append(nn.ReflectionPad2d(kernel_size // 2))
    if trainable:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    else:
        layers.append(MyConv2D(in_channels, out_channels, kernel_size, stride))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        # different
        layers.append(nn.LeakyReLU(0.2))
    return layers


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            *convLayer(channels, channels, kernel_size=3, stride=1),
            *convLayer(channels, channels, kernel_size=, stride=1, relu=False)
        )
        
    def forward(self, x):
        return self.conv(x) + x

class TransformNet(nn.Module):

    def __init__(self, base=8):
        super().__init__()
        self.base = base
        self.weights = []
        self.downsampling = nn.Sequential(
            *convLayer(3, base, kernel_size=9, trainable=True),
            *convLayer(base, base * 2, kernel_size=3, stride=2),
            *convLayer(base * 2, base * 4, kernel_size=3, stride=2),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])
        self.upsampling = nn.Sequential(
            *convLayer(base * 4, base * 2, kernel_size=3, upsample=2),
            *convLayer(base * 2, base, kernel_size=3, upsample=2),
            *convLayer(base, 3, kernel_size=9, instance_norm=False, relu=False, trainable=True),
        )
        self.get_param_dict()

    def forward(self, X):
        y = self.downsampling(X)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y

    