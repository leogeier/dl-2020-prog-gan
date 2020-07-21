import datetime
import os
import time
from math import sqrt

import torch
from torch.nn import AvgPool2d
from torch.nn.functional import interpolate
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan.Discriminator import Discriminator
from gan.Generator import Generator
from gan.Loss import ConditionalWLoss


class ConditionalGAN:

    @staticmethod
    def __log(log_dir, current_depth, current_batch, dis_loss, gen_loss):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
        with open(log_file, "a") as gan_log:
            gan_log.write(str(current_batch) + "\t" + str(dis_loss) + "\t" + str(gen_loss) + "\n")

    def __init__(self, num_attributes, depth, latent_size, lr, device, attributes_dict):
        """
        :param num_attributes: number of different attribute labels of images
        :param depth: depth of the GAN, i.e. number of layers
        :param latent_size: size of input noise for the generator
        :param lr: learning rate for the optimizer
        :param device: device to run on (cpu or gpu)
        """
        self.attributes_dict = attributes_dict
        self.num_attributes = num_attributes
        self.depth = depth
        self.latent_size = latent_size
        self.lr = lr
        self.device = device

        self.generator = Generator(self.depth, self.latent_size).to(self.device)
        self.discriminator = Discriminator(self.depth, self.latent_size,
                                           conditional=True, num_attributes=num_attributes).to(self.device)

        self.loss = ConditionalWLoss(self.discriminator)

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

    def optimize_discriminator(self, noise_batch, real_batch, current_depth, alpha, attributes):
        total_loss = 0
        real_samples = self.__progressive_downsampling(real_batch, current_depth, alpha)

        for _ in range(self.discriminator_updates):
            fake_samples = self.generator(noise_batch, current_depth, alpha).detach()

            loss = self.loss.discriminator_loss(attributes, real_samples, fake_samples, current_depth, alpha)

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.discriminator_updates

    def optimize_generator(self, noise_batch, current_depth, alpha, attributes):
        fake_samples = self.generator(noise_batch, current_depth, alpha)
        loss = self.loss.generator_loss(attributes, fake_samples, current_depth, alpha)

        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()

        return loss.item()

    def __save_attribute_info(self, sample_dir, attributes):
        os.makedirs(sample_dir, exist_ok=True)
        attr_file = os.path.join(sample_dir, "attributes.txt")
        with open(attr_file, "w") as file:
            file.write(", ".join([a[1] for a in sorted(self.attributes_dict.items())]) + "\n")
            for i, attr in enumerate(attributes):
                file.write(str(i) + ": " + ", ".join([str(a.item()) for a in attr]) + "\n")

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

    def __select_attributes(self, input_attributes):
        sel_attr_indices = torch.LongTensor([list(self.attributes_dict.keys())])
        return input_attributes.gather(dim=1, index=sel_attr_indices.expand(input_attributes.shape[0], -1))

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
        self.generator.train()
        self.discriminator.train()

        global_time = time.time()

        # input for consistent samples during training
        temp_dataloader = DataLoader(dataset, num_samples, shuffle=True)
        temp_iterator = iter(temp_dataloader)
        _, some_attributes = next(temp_iterator)
        some_attributes = self.__select_attributes(some_attributes)
        fixed_attributes = some_attributes.view(num_samples, -1).to(self.device)
        fixed_images = torch.randn(num_samples, self.latent_size - self.num_attributes).to(self.device)
        # concatenated for generator. using projection for discriminator. see below
        fixed_input = torch.cat((fixed_attributes.float(), fixed_images), dim=-1)
        self.__save_attribute_info(sample_dir, some_attributes)
        del temp_iterator
        del temp_dataloader

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

                for i, (images, attributes) in enumerate(dataloader, 1):
                    alpha = total_steps / fader_point if total_steps <= fader_point else 1

                    images = images.to(self.device)
                    attributes = self.__select_attributes(attributes.view(images.shape[0], -1)).to(self.device)

                    noise = torch.randn(images.shape[0], self.latent_size - self.num_attributes).to(self.device)
                    # Noise and attributes are concatenated for the generator.
                    # The discriminator receives only the generator output initially. Attributes are multiplied in
                    # in the final block. (following https://arxiv.org/pdf/1802.05637.pdf architecture in figure 14)
                    gen_input = torch.cat((attributes.float(), noise), dim=-1)

                    dis_loss = self.optimize_discriminator(gen_input, images, current_depth, alpha, attributes)
                    gen_loss = self.optimize_generator(gen_input, current_depth, alpha, attributes)

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
