import os
import sys

import torch

from gan.ConditionalGAN import ConditionalGAN

SELECTED_ATTRIBUTE_NAMES = {4:  'Bald',        8: 'Black_Hair',  9: 'Blond_Hair',
                            11: 'Brown_Hair', 15: 'Eyeglasses', 17: 'Gray_Hair',
                            20: 'Male',       22: 'Mustache',   24: 'No_Beard',
                            31: 'Smiling'}
SELECTED_ATTRIBUTES = [4, 8, 9, 11, 15, 17, 20, 22, 24, 31]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} num_samples [depth]".format(sys.argv[0]))
        sys.exit(1)

    depth = -1 if len(sys.argv) < 3 else int(sys.argv[2])

    gan = ConditionalGAN(num_attributes=len(SELECTED_ATTRIBUTES), depth=6, latent_size=128,
                             lr=0.001, device=torch.device('cpu'), attributes_dict=SELECTED_ATTRIBUTE_NAMES)

    for i, model_part in enumerate(
            [gan.generator, gan.generator_optimizer, gan.discriminator, gan.discriminator_optimizer]):
        load_depth = 5
        if i == 0:
            filename = os.path.join("./models", "GAN_GEN_" + str(load_depth) + ".pth")
        elif i == 1:
            filename = os.path.join("./models", "GAN_GEN_OPTIM_" + str(load_depth) + ".pth")
        elif i == 2:
            filename = os.path.join("./models", "GAN_DIS_" + str(load_depth) + ".pth")
        elif i == 3:
            filename = os.path.join("./models", "GAN_DIS_OPTIM_" + str(load_depth) + ".pth")
        model_part.load_state_dict(torch.load(filename, map_location=str(gan.device)))

    inference_attributes = torch.LongTensor([[0,0,0,0,1,1,1,1,0,0]], device=gan.device)
    gan.infer(num_samples=int(sys.argv[1]), attributes=inference_attributes, depth=depth)