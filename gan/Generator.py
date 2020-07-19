import torch
from torch.nn import LeakyReLU
from torch.nn.functional import interpolate

from gan.EqualizedLayers import EqualizedConv2d, EqualizedDeconv2d


class PixelwiseNormalization(torch.nn.Module):
    """
    Normalize feature vectors per pixel as suggested in section 4.2 of
    https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf.
    For each pixel location (i,j) in the input image, takes the vector across all channels and normalizes it to
    unit length.
    """
    def __init__(self):
        super(PixelwiseNormalization, self).__init__()

    def forward(self, x, eps=1e-8):
        """
        :param x: input with shape (batch_size x num_channels x img_width x img_height)
        :param eps: small constant to avoid division by zero
        :return:
        """
        return x / x.pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()


class GenInitialBlock(torch.nn.Module):
    """
    Initial block of generator. Consisting of the following layers:

    input: latent noise vector (latent_size x 1 x 1)

    layer               activation       output shape
    Convolution 4 x 4   LeakyReLU        latent_size x 4 x 4
    Convolution 3 x 3   LeakyReLU        latent_size x 4 x 4

    output: image with latent_size channels (latent_size x 4 x 4)
    """

    def __init__(self, latent_size):
        """
        :param latent_size: size of noise input for generator
        """
        super(GenInitialBlock, self).__init__()

        self.layer1 = EqualizedDeconv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=(4, 4))
        self.layer2 = EqualizedConv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=(3, 3), padding=1)

        self.pixel_normalization = PixelwiseNormalization()
        self.activation = LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        """
        :param x: input noise (batch_size x latent_size)
        :return:
        """
        # add image width and height dimensions:
        # (batch_size x latent_size) --> (batch_size x latent_size x 1 x 1)
        y = torch.unsqueeze(torch.unsqueeze(x, -1), -1)

        y = self.activation(self.layer1(y))
        y = self.activation(self.layer2(y))

        return self.pixel_normalization(y)


class GenConvolutionalBlock(torch.nn.Module):
    """
    Regular block of generator. Consisting of following layers:

    input: image (in_channels x img_width x img_height)

    layer               activation       output shape
    Upsampling          -                in_channels x 2*img_width x 2*img_height
    Convolution 3 x 3   LeakyReLU        out_channels x 2*img_width x 2*img_height
    Convolution 3 x 3   LeakyReLU        out_channels x 2*img_width x 2*img_height

    output: image with latent_size channels and doubled size (out_channels x 2*img_width x 2*img_height)
    """

    def __init__(self, in_channels, out_channels):
        super(GenConvolutionalBlock, self).__init__()

        self.upsample = lambda x: interpolate(x, scale_factor=2)
        self.layer1 = EqualizedConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.layer2 = EqualizedConv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)

        self.pixel_normalization = PixelwiseNormalization()
        self.activation = LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        y = self.upsample(x)
        y = self.pixel_normalization(self.activation(self.layer1(y)))
        y = self.pixel_normalization(self.activation(self.layer2(y)))
        return y


class Generator(torch.nn.Module):

    @staticmethod
    def __to_rgb(in_channels):
        return EqualizedConv2d(in_channels, 3, (1, 1))

    def __init__(self, depth, latent_size):
        """
        :param depth: depth of the generator, i.e. number of blocks (initial + convolutional)
        :param latent_size: size of input noise for the generator
        """
        super(Generator, self).__init__()

        self.depth = depth
        self.latent_size = latent_size

        self.initial_block = GenInitialBlock(self.latent_size)
        self.blocks = torch.nn.ModuleList([])

        # hold an rgb converter for every intermediate resolution to visualize intermediate results
        self.rgb_converters = torch.nn.ModuleList([self.__to_rgb(self.latent_size)])

        for i in range(self.depth - 1):
            if i < 3:
                # first three blocks do not reduce the number of channels
                in_channels = self.latent_size
                out_channels = self.latent_size
            else:
                # half number of channels in each block
                in_channels = self.latent_size // pow(2, i - 3)
                out_channels = self.latent_size // pow(2, i - 2)

            block = GenConvolutionalBlock(in_channels, out_channels)
            rgb = self.__to_rgb(out_channels)

            self.blocks.append(block)
            self.rgb_converters.append(rgb)

    def forward(self, x, current_depth, alpha):
        """
        :param x: input noise (batch_size x latent_size)
        :param current_depth: depth at which to evaluate (maximum depth of the forward pass)
        :param alpha: interpolation between current depth output (alpha) and previous depth output (1 - alpha)
        :return:
        """
        y = self.initial_block(x)

        if current_depth == 0:
            return self.rgb_converters[0](y)

        for block in self.blocks[:current_depth - 1]:
            y = block(y)

        residual = self.rgb_converters[current_depth - 1](interpolate(y, scale_factor=2))
        straight = self.rgb_converters[current_depth](self.blocks[current_depth - 1](y))

        # fade in new layer
        return alpha * straight + (1 - alpha) * residual
