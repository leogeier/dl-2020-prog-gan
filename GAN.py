from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision

from Discriminator import Discriminator
from Generator import Generator
from Loss import Loss
from CelebA import CelebA

import sys


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

        self.generator = Generator(self.depth, self.latent_size)
        self.discriminator = Discriminator()

        self.g_optimizer = Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

    def optimize_discriminator(self, noise_batch, real_batch, depth, alpha):
        pass

    def optimize_generator(self, noise_batch, real_batch, depth, alpha):
        pass

    def train(self, dataset, epochs_per_depth, batch_size_per_depth, fade_in_epoch_ratios):
        self.generator.train()
        self.discriminator.train()

        start_time = time.time()
        print("Start training")
        for current_depth in range(depth):
            resolution = 2 ** (current_depth + 2)
            print("Currently on depth {} ({} x {})".format(current_depth, resolution, resolution))

            dataloader = DataLoader(dataset, batch_size_per_depth[current_depth], shuffle=True, pin_memory=True)
            
            total_steps = 1
            batches_count = len(dataloader)
            fader_point = int(fade_in_percentage[current_depth] * epochs_per_depth[current_depth] * batches_count)

            for bath lolol

            total_steps += 1




if __name__ == "__main__":
    if (len(sys.argv) < 6):
        print("Usage: {} [dataset root] [depth] [latent size] [learning rate]".format(sys.argv[0]))
        sys.exit(1)

    dataset = CelebA(sys.argv[1], transform=torchvision.transforms.ToTensor())
    depth = int(sys.argv[2])
    latent_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    epochs_per_depth = None
    batch_size_per_depth = None
    fade_in_epoch_ratios = None

    gan = GAN(depth, latent_size, learning_rate)
    gan.train(dataset, epochs, batch_sizes, fade_in_percentage)
