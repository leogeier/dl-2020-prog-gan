import torch

from gan.Generator import GenInitialBlock


def test_shape():
    block = GenInitialBlock(latent_size=512)
    x = torch.randn(1, 512)
    y = block(x)
    assert y.shape == (1, 512, 4, 4)
