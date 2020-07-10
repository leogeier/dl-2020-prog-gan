import torch
from torch.nn.functional import interpolate
from EqualizedConv2d import EqualizedConv2d

class GenInitialBlock(torch.nn.Module):

    def __init__(self, latent_size):
        self.bias = torch.nn.Parameter(torch.FloatTensor(latent_size).fill_(0))

    def forward(self, x):
        return x * self.bias

class GenConvolutionalBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

    def forward(self, x):
        return x * self.bias

class Generator(torch.nn.Module):

    @staticmethod
    def __toRGB(in_channels):
        return EqualizedConv2d(in_channels, 3, (1, 1))

    def __init__(self, depth, latent_size):
        """

        :param depth: depth of the GAN, i.e. number of layers
        :param latent_size: size of input noise for the generator
        """
        super(Generator, self).__init__()

        self.depth = depth
        self.latent_size = latent_size

        self.initial_block = GenInitialBlock(self.latent_size)
        self.layers = torch.nn.ModuleList([])

        self.rgb_converters = torch.nn.ModuleList([self.__toRGB(self.latent_size)])

        for i in range(self.depth - 1):
            if i < 3:
                layer = GenConvolutionalBlock(self.latent_size, self.latent_size)
                rgb = self.__toRGB(self.latent_size)
            else:
                layer = GenConvolutionalBlock(
                        int(self.latent_size // (2 ** i - 3)),
                        int(self.latent_size // (2 ** i - 2)),
                    )
                rgb = self.__toRGB(int(self.latent_size // (2 ** i - 2)))

            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x, current_depth, alpha):
        y = self.initial_block(x)

        if current_depth == 0:
            return self.rgb_converters[0](y)

        for block in self.layers[:current_depth - 1]:
            y = block(y)

        residual = self.rgb_converters[current_depth - 1](interpolate(y, scale_factor=2))
        straight = self.rgb_converters[current_depth](self.layers[current_depth - 1](y))

        # fade in new layer
        return (alpha * straight) + ((1 - alpha) * residual)
