import datetime
import os
import sys
import time
from math import sqrt, log

import torch
import torchvision
from torch.nn import AvgPool2d
from torch.nn.functional import interpolate
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan.CelebA import CelebA
from gan.Discriminator import Discriminator
from gan.Generator import Generator
from gan.Loss import WassersteinLoss


class GAN:

    @staticmethod
    def __log(log_dir, current_depth, current_batch, dis_loss, gen_loss):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
        with open(log_file, "a") as gan_log:
            gan_log.write(str(current_batch) + "\tDis: " + str(dis_loss) + "\tGen: " + str(gen_loss) + "\n")

    def __init__(self, depth, latent_size, lr, device):
        """
        :param depth: depth of the GAN, i.e. number of layers
        :param latent_size: size of input noise for the generator
        :param lr: learning rate for the optimizer
        :param device: device to run on (cpu or gpu)
        """
        self.depth = depth
        self.latent_size = latent_size
        self.lr = lr
        self.device = device

        self.generator = Generator(self.depth, self.latent_size).to(self.device)
        self.discriminator = Discriminator(self.depth, self.latent_size).to(self.device)

        self.loss = WassersteinLoss(self.discriminator, use_gradient_penalty=True)

        self.discriminator_updates = 1  # updates of discriminator per generator update
        betas = (0, 0.99)  # adam hyper param
        self.generator_optimizer = Adam(self.generator.parameters(), lr=self.lr, betas=betas)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.lr, betas=betas)

        # TODO: use DataParallel here?
        # TODO: use exponential moving average

    def __progressive_downsampling(self, real_batch, current_depth, alpha):
        down_sample_factor = 2 ** (self.depth - current_depth - 1)
        assert down_sample_factor <= (real_batch.shape[-1] / 4), \
            "Image size is too small for downsampling at this depth"
        prior_downsample_factor = max(2 ** (self.depth - current_depth), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if current_depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),
                                                              scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        return (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

    def optimize_discriminator(self, noise_batch, real_batch, current_depth, alpha):
        total_loss = 0
        real_samples = self.__progressive_downsampling(real_batch, current_depth, alpha)

        for _ in range(self.discriminator_updates):
            fake_samples = self.generator(noise_batch, current_depth, alpha).detach()

            loss = self.loss.discriminator_loss(real_samples, fake_samples, current_depth, alpha)

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.discriminator_updates

    def optimize_generator(self, noise_batch, real_batch, current_depth, alpha):
        real_samples = self.__progressive_downsampling(real_batch, current_depth, alpha)

        fake_samples = self.generator(noise_batch, current_depth, alpha)
        loss = self.loss.generator_loss(real_samples, fake_samples, current_depth, alpha)

        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()

        return loss.item()

    def __save_samples(self, sample_dir, fixed_input, current_depth, current_epoch, current_batch, alpha):
        os.makedirs(sample_dir, exist_ok=True)
        img_file = os.path.join(sample_dir,
                                "gen_" + str(current_depth) + "_" + str(current_epoch) + "_"
                                + str(current_batch) + ".png")
        samples=self.generator(fixed_input, current_depth, alpha).detach()
        scale=int(pow(2, self.depth - current_depth - 1))
        if scale > 1:
            samples = interpolate(samples, scale_factor=scale)
        save_image(samples, img_file, nrow=int(sqrt(len(samples))), normalize=True, scale_each=True)

    def __save_model(self, model_dir, current_depth):
        os.makedirs(model_dir, exist_ok=True)
        gen_file = os.path.join(model_dir, "GAN_GEN_" + str(current_depth) + ".pth")
        dis_file = os.path.join(model_dir, "GAN_DIS_" + str(current_depth) + ".pth")
        gen_optim_file = os.path.join(model_dir, "GAN_GEN_OPTIM_" + str(current_depth) + ".pth")
        dis_optim_file = os.path.join(model_dir, "GAN_DIS_OPTIM_" + str(current_depth) + ".pth")
        torch.save(self.generator.state_dict(), gen_file)
        torch.save(self.discriminator.state_dict(), dis_file)
        torch.save(self.generator_optimizer.state_dict(), gen_optim_file)
        torch.save(self.discriminator_optimizer.state_dict(), dis_optim_file)

    def train(self, dataset, epochs_per_depth, batch_size_per_depth, fade_in_ratios, start_depth=0,
              log_frequency=100, num_samples=16, log_dir="./logs", sample_dir="./samples", model_dir="./models"):
        """
        :param dataset: dataset used for training (no dataloader)
        :param epochs_per_depth: list of epochs to train for each resolution
        :param batch_size_per_depth: list of batch sizes for each resolution
        :param fade_in_ratios: fraction of epochs per resolution used for fade in of new resolution block
        :param start_depth: start training at this depth / resolution
        :param log_frequency: how many log entries per epoch
        :param num_samples: number of samples for output during training
        :param log_dir: directory for storing the logs
        :param sample_dir: directory for storing sample images
        :param model_dir: directory for storing the trained models
        """
        assert dataset[0].shape[-1] == dataset[0].shape[-2], "Image size is not square"
        assert log(dataset[0].shape[-1], 2).is_integer(), "Image size is not a power of two"
        assert 4 * pow(2, self.depth - 1) == dataset[0].shape[-1], \
            "Depth and image size are not compatible"

        self.generator.train()
        self.discriminator.train()

        global_time = time.time()

        # input for consistent samples during training
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

        print("Start training.")

        for current_depth in range(self.depth):
            resolution = 2 ** (current_depth + 2)
            print("Currently on depth {} with resolution ({} x {})".format(current_depth, resolution, resolution))

            dataloader = DataLoader(dataset, batch_size_per_depth[current_depth], shuffle=True, pin_memory=True)

            total_steps = 1
            batches_count = len(dataloader)
            log_interval = max(int(batches_count / log_frequency), 1)
            # calculate number of batches for fade-in
            fader_point = int(fade_in_ratios[current_depth] * epochs_per_depth[current_depth] * batches_count)

            for epoch in range(1, epochs_per_depth[current_depth] + 1):
                print("Epoch ", epoch)

                for i, batch in enumerate(dataloader, 1):
                    alpha = total_steps / fader_point if total_steps <= fader_point else 1

                    images = batch.to(self.device)
                    noise = torch.randn(batch.shape[0], self.latent_size).to(self.device)

                    dis_loss = self.optimize_discriminator(noise, images, current_depth, alpha)
                    gen_loss = self.optimize_generator(noise, images, current_depth, alpha)

                    if i % log_interval == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s] Batch: %d Dis. Loss: %f Gen. Loss: %f" % (elapsed, i, dis_loss, gen_loss))
                        self.__log(log_dir, current_depth, i, dis_loss, gen_loss)
                        with torch.no_grad():
                            self.__save_samples(sample_dir, fixed_input, current_depth, epoch, i, alpha)

                    total_steps += 1

                self.__save_model(model_dir, current_depth)

        self.generator.eval()
        self.discriminator.eval()
        print("Training finished.")


if __name__ == "__main__":
    if len(sys.argv) < 5:
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
