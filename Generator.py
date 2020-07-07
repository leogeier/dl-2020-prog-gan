import torch


class Generator(torch.nn.Module):

    def __init__(self, depth, latent_size):
        """

        :param depth: depth of the GAN, i.e. number of layers
        :param latent_size: size of input noise for the generator
        """
        super(Generator, self).__init__()

        self.depth = depth
        self.latent_size = latent_size

        # TODO
        self.initial_block = None
        self.layers = torch.nn.ModuleList([])

        for i in range(self.depth - 1):
            layer = ConvolutionalBlock()
            # TODO
            # rgb_converter = None
            self.layers.append(layer)
            # self.rgb_converters.append(rgb_converter)

    def forward(self, x, depth, alpha):
        pass
