import torch


class Discriminator(torch.nn.Module):

        def __init__(self, depth, feature_size):
            """
            :param depth: total height of the discriminator (Must be equal to the Generator depth)
            :param feature_size: size of the deepest features extracted
                                 (Must be equal to Generator latent_size)
            """
            super(Discriminator, self).__init__()

            # create state of the object
            self.depth = depth
            self.feature_size = feature_size

            # TODO
            self.final_block = None

            self.layers = torch.nn.ModuleList([])

            # create the remaining layers
            for i in range(self.depth - 1):
                layer = ConvolutionalBlock()
                # TODO rgb
                self.layers.append(layer)
                # self.rgb_to_features.append(rgb)

        def forward(self, x, height, alpha):
            """
            :param x: input to the network
            :param height: current height of operation (Progressive GAN)
            :param alpha: current value of alpha for fade-in
            :return: out => raw prediction values (WGAN-GP)
            """
            pass
