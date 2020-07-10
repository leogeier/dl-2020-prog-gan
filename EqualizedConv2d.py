import torch
from torch.nn.modules.utils import _pair
from torch.nn.functional import conv2d
from numpy import prod, sqrt

class EqualizedConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(EqualizedConv2d, self).__init__()

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *_pair(kernel_size)))

        self.stride = stride
        self.padding = padding

        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

        fan_in = prod(_pair(kernel_size)) * in_channels
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        return conv2d(input=x,
                weight=self.weight * self.scale,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding)
