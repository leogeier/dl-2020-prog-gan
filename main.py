import sys

import torch
import torchvision

from gan.CelebA import CelebA
from gan.ConditionalGAN import ConditionalGAN
from gan.GAN import GAN

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: {} [dataset root] [depth] [latent size] [learning rate] [epochs] [fade_in] [use_conditional]".format(sys.argv[0]))
        sys.exit(1)

    depth = int(sys.argv[2])

    appropriate_size = 4 * pow(2, depth - 1)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(appropriate_size),
        torchvision.transforms.ToTensor()
    ])
    dataset = CelebA(root=sys.argv[1], split="all", transform=transform)

    print("Using data with image format {}".format(dataset[0][0].shape))

    latent_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    epochs_per_depth = [int(sys.argv[5])] * depth
    batch_size_per_depth = [16, 16, 16, 16, 16, 16, 14, 6, 3]  # according to nvidia paper
    fade_in_epoch_ratios = [float(sys.argv[6])] * depth
    use_conditional = bool(sys.argv[7])

    if use_conditional:
        gan = ConditionalGAN(num_attributes=dataset[0][1].shape[0], depth=depth, latent_size=latent_size,
                             lr=learning_rate, device=torch.device('cpu'))
        gan.train(dataset, epochs_per_depth, batch_size_per_depth, fade_in_epoch_ratios)
    else:
        gan = GAN(depth=depth, latent_size=latent_size, lr=learning_rate, device=torch.device('cpu'))
        gan.train(dataset, epochs_per_depth, batch_size_per_depth, fade_in_epoch_ratios)
