import torch
from torch.nn import LeakyReLU, AvgPool2d

from gan.EqualizedLayers import EqualizedConv2d


class MinibatchStdDev(torch.nn.Module):
    """
    Concatenate a constant statistic calculated across the minibatch to each pixel location (i, j) as a new channel.
    Here the standard deviation averaged over channels and locations. This is to increase variation of images produced
    by the generator. (see section 3)
    https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf
    """
    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, eps=1e-8):
        batch_size, _, img_width, img_height = x.shape

        # subtract batch mean
        y = x - x.mean(dim=0, keepdim=True)

        # calculate stddev across batch
        y = y.pow(2).mean(dim=0, keepdim=False).add(eps).sqrt()

        # average over all channels and locations
        y = y.mean().view(1, 1, 1, 1)

        # replicate for every channel and location
        y = y.repeat(batch_size, 1, img_width, img_height)

        # append as new channel
        return torch.cat([x, y], 1)


class DisFinalBlock(torch.nn.Module):

    def __init__(self, feature_size):
        super(DisFinalBlock, self).__init__()

        self.minibatch_std_dev = MinibatchStdDev()
        self.layer1 = EqualizedConv2d(in_channels=feature_size + 1, out_channels=feature_size, kernel_size=(3, 3),
                                      padding=1)
        self.layer2 = EqualizedConv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=(4, 4))
        self.fully_connected = EqualizedConv2d(in_channels=feature_size, out_channels=1, kernel_size=(1, 1))
        self.activation = LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        y = self.minibatch_std_dev(x)

        y = self.activation(self.layer1(y))
        y = self.activation(self.layer2(y))

        y = self.fully_connected(y)

        return y.view(-1)


class DisConditionalFinalBlock(torch.nn.Module):
    """
    uses projection-based attribute passing from https://arxiv.org/pdf/1802.05637.pdf
    """

    def __init__(self, feature_size, num_attributes):
        super(DisConditionalFinalBlock, self).__init__()

        self.minibatch_std_dev = MinibatchStdDev()
        self.layer1 = EqualizedConv2d(in_channels=feature_size + 1, out_channels=feature_size, kernel_size=(3, 3),
                                      padding=1)
        self.layer2 = EqualizedConv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=(4, 4))
        self.fully_connected = EqualizedConv2d(in_channels=feature_size, out_channels=1, kernel_size=(1, 1))
        self.activation = LeakyReLU(negative_slope=0.2)

        # learn one embedding vector with unit norm of length feature_size per attribute (sum over all attr in img)
        self.attribute_embedder = torch.nn.EmbeddingBag(num_attributes, feature_size, max_norm=1, mode="sum")

    def forward(self, x, attributes):
        y = self.minibatch_std_dev(x)

        y = self.activation(self.layer1(y))
        y = self.activation(self.layer2(y))  # batch_size x num_channels x 1 x 1

        # convert one-hot attributes to embedding indices
        attribute_indices = attributes.nonzero(as_tuple=True)[1]
        attribute_offsets = torch.cat((torch.zeros(1, dtype=torch.long), attributes.sum(dim=1)))
        attribute_offsets = attribute_offsets.cumsum(dim=-1).narrow(0, 0, attribute_offsets.shape[0] - 1)

        embedded = self.attribute_embedder(attribute_indices, attribute_offsets)  # batch_size x num_channels
        y_squeezed = torch.squeeze(torch.squeeze(y, dim=-1), dim=-1)  # batch_size x num_channels

        print(attribute_indices, attribute_offsets)

        # for calculation below, see equation 3 in https://arxiv.org/pdf/1802.05637.pdf
        projection = (embedded * y_squeezed).sum(dim=1)  # batch_size
        print(projection.shape)

        return self.fully_connected(y).view(-1) + projection


class DisConvolutionalBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DisConvolutionalBlock, self).__init__()

        self.layer1 = EqualizedConv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1)
        self.layer2 = EqualizedConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

        self.downsample = AvgPool2d(2)
        self.activation = LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        y = self.activation(self.layer1(x))
        y = self.activation(self.layer2(y))
        return self.downsample(y)


# TODO: Rename to critic
class Discriminator(torch.nn.Module):

    @staticmethod
    def __from_rgb(out_channels):
        return EqualizedConv2d(in_channels=3, out_channels=out_channels, kernel_size=(1, 1))

    def __init__(self, depth, feature_size, conditional=False, num_attributes=None):
        """
        :param depth: depth of the discriminator, i.e. number of blocks (must be equal to the generator depth)
        :param feature_size: size of the deepest features extracted (must be equal to generator latent_size)
        :param conditional: whether to use attribute labels for the images
        :param num_attributes: number of different attributes
        """
        super(Discriminator, self).__init__()

        self.depth = depth
        self.feature_size = feature_size
        self.conditional = conditional

        self.final_block = DisConditionalFinalBlock(self.feature_size, num_attributes) if self.conditional \
            else DisFinalBlock(self.feature_size)
        self.blocks = torch.nn.ModuleList([])

        self.rgb_to_features = torch.nn.ModuleList([self.__from_rgb(self.feature_size)])
        self.downsample = torch.nn.AvgPool2d(2)

        for i in range(self.depth - 1):
            if i > 2:
                in_channels = self.feature_size // pow(2, i - 2)
                out_channels = self.feature_size // pow(2, i - 3)
            else:
                in_channels = self.feature_size
                out_channels = self.feature_size

            block = DisConvolutionalBlock(in_channels, out_channels)
            rgb = self.__from_rgb(in_channels)

            self.blocks.append(block)
            self.rgb_to_features.append(rgb)

    def forward(self, x, current_depth, alpha, attributes=None):
        """
            :param x: input to the network
            :param current_depth: current height of operation
            :param alpha: current interpolation value for fade-in
            :param attributes: attribute labels for x if conditional
            """
        if self.conditional:
            assert attributes is not None, "Conditional discriminator needs attributes"

        if current_depth == 0:
            y = self.rgb_to_features[0](x)
            return self.final_block(y, attributes) if self.conditional else self.final_block(y)

        residual = self.rgb_to_features[current_depth - 1](self.downsample(x))
        straight = self.blocks[current_depth - 1](self.rgb_to_features[current_depth](x))

        y = alpha * straight + (1 - alpha) * residual

        for block in reversed(self.blocks[:current_depth - 1]):
            y = block(y)

        out = self.final_block(y, attributes) if self.conditional else self.final_block(y)

        return out
