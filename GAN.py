from torch.optim import Adam

from Discriminator import Discriminator
from Generator import Generator
from Loss import Loss


class GAN:
    def __init__(self, depth, latent_size, lr):
        """
        :param depth: depth of the GAN, i.e. number of layers
        :param latent_size: size of input noise for the generator
        :param lr: learning rate for the optimizer
        """
        self.depth = depth
        self.latent_size = latent_size
        self.lr = lr
        self.betas = (0, 0.99)
        self.loss = Loss()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.g_optimizer = Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

    def optimize_discriminator(self, noise_batch, real_batch, depth, alpha):
        pass

    def optimize_generator(self, noise_batch, real_batch, depth, alpha):
        pass

    def train(self, dataset, epochs, batch_sizes, fade_in_percentage):