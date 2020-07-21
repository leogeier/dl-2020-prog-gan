import sys

import torch
import torchvision

from gan.CelebA import CelebA
from gan.ConditionalGAN import ConditionalGAN
from gan.GAN import GAN

SELECTED_ATTRIBUTE_NAMES = {4:  'Bald',        8: 'Black_Hair',  9: 'Blond_Hair',
                            11: 'Brown_Hair', 15: 'Eyeglasses', 17: 'Gray_Hair',
                            20: 'Male',       22: 'Mustache',   24: 'No_Beard',
                            31: 'Smiling'}
SELECTED_ATTRIBUTES = [4, 8, 9, 11, 15, 17, 20, 22, 24, 31]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} [dataset root] [epochs]".format(sys.argv[0]))
        sys.exit(1)

    depth = 6

    appropriate_size = 4 * pow(2, depth - 1)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(appropriate_size),
        torchvision.transforms.ToTensor()
    ])
    dataset = CelebA(root=sys.argv[1], split="all", transform=transform, size=13948)

    print("Using data with image format {}".format(dataset[0][0].shape))

    latent_size = 128
    learning_rate = 0.001
    epochs_per_depth = [int(sys.argv[2])] * depth
    batch_size_per_depth = [16, 16, 16, 16, 16, 16, 14, 6, 3]  # according to nvidia paper
    fade_in_epoch_ratios = [0.5] * depth

    gan = ConditionalGAN(num_attributes=len(SELECTED_ATTRIBUTES), depth=depth, latent_size=latent_size,
                         lr=learning_rate, device=torch.device('cpu'), attributes_dict=SELECTED_ATTRIBUTE_NAMES)
    gan.train(dataset, epochs_per_depth, batch_size_per_depth, fade_in_epoch_ratios)
