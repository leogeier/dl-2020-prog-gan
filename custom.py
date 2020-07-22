import os
import sys

import torch
import torchvision

from gan.CelebA import CelebA
from gan.ConditionalGAN import ConditionalGAN

SELECTED_ATTRIBUTE_NAMES = {4:  'Bald',        8: 'Black_Hair',  9: 'Blond_Hair',
                            11: 'Brown_Hair', 15: 'Eyeglasses', 17: 'Gray_Hair',
                            20: 'Male',       22: 'Mustache',   24: 'No_Beard',
                            31: 'Smiling'}
SELECTED_ATTRIBUTES = [4, 8, 9, 11, 15, 17, 20, 22, 24, 31]

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: {} dataset_root epochs load_saved [start_depth] [load_same_depth]".format(sys.argv[0]))
        sys.exit(1)
    elif 4 <= len(sys.argv) < 6:
        print("Usage: {} dataset_root epochs load_saved [start_depth] [load_same_depth]".format(sys.argv[0]))
        sys.exit(1)

    depth = 6
    load_saved = sys.argv[3] == "True"

    appropriate_size = 4 * pow(2, depth - 1)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(appropriate_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CelebA(root=sys.argv[1], split="all", transform=transform)

    print("Using data with image format {}".format(dataset[0][0].shape))

    latent_size = 128
    learning_rate = 0.001
    epochs_per_depth = [int(sys.argv[2])] * depth
    batch_size_per_depth = [16, 16, 16, 16, 16, 16, 14, 6, 3]  # according to nvidia paper
    fade_in_epoch_ratios = [0.5] * depth
    log_frequency = 50

    gan = ConditionalGAN(num_attributes=len(SELECTED_ATTRIBUTES), depth=depth, latent_size=latent_size,
                         lr=learning_rate, device=torch.device('cuda'), attributes_dict=SELECTED_ATTRIBUTE_NAMES)
    if load_saved:
        for i, model_part in enumerate([gan.generator, gan.generator_optimizer, gan.discriminator, gan.discriminator_optimizer]):
            start_depth = int(sys.argv[4])
            load_depth = start_depth if sys.argv[5] == "True" else max(start_depth - 1, 0)
            if i == 0:
                filename = os.path.join("./models", "GAN_GEN_" + str(load_depth) + ".pth")
            elif i == 1:
                filename = os.path.join("./models", "GAN_GEN_OPTIM_" + str(load_depth) + ".pth")
            elif i == 2:
                filename = os.path.join("./models", "GAN_DIS_" + str(load_depth) + ".pth")
            elif i == 3:
                filename = os.path.join("./models", "GAN_DIS_OPTIM_" + str(load_depth) + ".pth")
            model_part.load_state_dict(torch.load(filename, map_location=str(gan.device)))
            gan.train(dataset, epochs_per_depth, batch_size_per_depth, fade_in_epoch_ratios, start_depth, log_frequency)
    else:
        gan.train(dataset, epochs_per_depth, batch_size_per_depth, fade_in_epoch_ratios, log_frequency=log_frequency)
