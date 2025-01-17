import torch
from torch.nn.modules.utils import _pair
from torch.nn.functional import conv2d, conv_transpose2d
from numpy import prod, sqrt


class EqualizedConv2d(torch.nn.Module):
    """
    Convolutional layer with equalized learning rate as suggested in section 4.1 of
    https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf.
    Scales weights according to the initialization method known as He's initializer (https://arxiv.org/abs/1502.01852),
    but dynamically at runtime for each pass. This aims at counteracting the normalization of gradient updates
    performed by optimization algorithms such as e.g. Adam (https://arxiv.org/pdf/1412.6980.pdf) for which the effective
    step size is invariant of the weight scale. This serves the purpose of avoiding slower convergence for weights with
    larger scale.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        :param in_channels: number of input channels before the convolution
        :param out_channels: number of output channels after the convolution
        :param kernel_size: kernel size for the convolution (single value if square, tuple otherwise)
        :param stride: stride for the convolution
        :param padding: padding for the convolution
        """
        super(EqualizedConv2d, self).__init__()

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *_pair(kernel_size)))

        self.stride = stride
        self.padding = padding

        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        fan_in = prod(_pair(kernel_size)) * in_channels
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        return conv2d(input=x,
                      weight=self.weight * self.scale,
                      bias=self.bias,
                      stride=self.stride,
                      padding=self.padding)


class EqualizedDeconv2d(torch.nn.Module):
    """
    Transpose of the convolutional layer with equalized learning rate. See EqualizedConv2d.
    Used here for upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        :param in_channels: number of input channels before the convolution
        :param out_channels: number of output channels after the convolution
        :param kernel_size: kernel size for the convolution (single value if square, tuple otherwise)
        :param stride: stride for the convolution
        :param padding: padding for the convolution
        """
        super(EqualizedDeconv2d, self).__init__()

        self.weight = torch.nn.Parameter(torch.randn(in_channels, out_channels, *_pair(kernel_size)))

        self.stride = stride
        self.padding = padding

        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        fan_in = in_channels
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,
                                bias=self.bias,
                                stride=self.stride,
                                padding=self.padding)
