from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision

from Discriminator import Discriminator
from Generator import Generator
from Loss import Loss
from CelebA import CelebA

import sys


class GAN:
    def __init__(self, depth, latent_size, lr, device):
        """
        :param depth: depth of the GAN, i.e. number of layers
        :param latent_size: size of input noise for the generator
        :param lr: learning rate for the optimizer
        """
        self.depth = depth
        self.latent_size = latent_size
        self.lr = lr
        self.device = device
        self.betas = (0, 0.99)
        self.loss = Loss()
        self.discriminator_updates = None

        self.generator = Generator(self.depth, self.latent_size)
        self.discriminator = Discriminator(self.depth, self.latent_size)

        self.generator_optimizer = Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

    def __progressive_downsampling(self, real_batch, current_depth, alpha):
        down_sample_factor = 2 ** (self.depth - current_depth - 1)
        prior_downsample_factor = max(2 ** (self.depth - current_depth), 0)

        ds_real_samples = nn.AvgPool2d(down_sample_factor)(real_batch)

        if current_depth > 0:
            prior_ds_real_samples = nn.functional.interpolate(AvgPool2d(prior_downsample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        return (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)
        
    def optimize_discriminator(self, noise_batch, real_batch, current_depth, alpha):
        real_samples = self.__progressive_downsampling(real_batch, current_depth, alpha)

        for _ in range(self.discriminator_updates):
            fake_samples = self.gen(noise_batch, current_depth, alpha).detach()

            loss = self.loss.discriminator_loss(real_samples, fake_samples, current_depth, alpha)

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()

    def optimize_generator(self, noise_batch, real_batch, current_depth, alpha):
        real_samples = self.__progressive_downsampling(real_batch, current_depth, alpha)

        fake_samples = self.gen(noise, current_depth, alpha)
        loss = self.loss.generator_loss(real_samples, fake_samples, current_depth, alpha)

        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()

    def train(self, dataset, epochs_per_depth, batch_size_per_depth, fade_in_epoch_ratios):
        self.generator.train()
        self.discriminator.train()

        # start_time = time.time()
        print("Start training")
        for current_depth in range(depth):
            resolution = 2 ** (current_depth + 2)
            print("Currently on depth {} ({} x {})".format(current_depth, resolution, resolution))

            dataloader = DataLoader(dataset, batch_size_per_depth[current_depth], shuffle=True, pin_memory=True)
            
            total_steps = 1
            batches_count = len(dataloader)
            fader_point = int(fade_in_epoch_ratios[current_depth] * epochs_per_depth[current_depth] * batches_count)

            for epoch in range(1, epochs_per_depth[current_depth] + 1):
                print("Epoch", epoch)

                for batch in dataloader:
                    alpha = total_steps / fader_point if total_steps <= fader_point else 1

                    images = batch.to(self.device)
                    noise = torch.randn(images.shape[0], self.latent_size).to(self.device)
                    
                    self.optimize_discriminator(noise, images, current_depth, alpha)
                    self.optimize_generator(noise, images, current_depth, alpha)

                    total_steps += 1
        print("Training ended")

if __name__ == "__main__":
    if (len(sys.argv) < 5):
        print("Usage: {} [dataset root] [depth] [latent size] [learning rate]".format(sys.argv[0]))
        sys.exit(1)

    dataset = CelebA(sys.argv[1], transform=torchvision.transforms.ToTensor())
    depth = int(sys.argv[2])
    latent_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    epochs_per_depth = None
    batch_size_per_depth = None
    fade_in_epoch_ratios = None



    gan = GAN(depth, latent_size, learning_rate, "cuda")
    gan.train(dataset, epochs, batch_sizes, fade_in_percentage)
