import torch

from gan.Generator import GenInitialBlock, GenConvolutionalBlock, Generator


def test_initial_shape():
    block = GenInitialBlock(latent_size=512)
    x = torch.randn(1, 512)
    y = block(x)
    assert y.shape == (1, 512, 4, 4)


def test_conv_shape():
    block = GenConvolutionalBlock(in_channels=32, out_channels=16)
    x = torch.randn(1, 32, 4, 4)
    y = block(x)
    assert y.shape == (1, 16, 8, 8)


def test_depth():
    gen = Generator(5, 32)
    assert gen.depth == 5
    assert len(gen.blocks) == 4


def test_overall_shape():
    gen = Generator(5, 32)
    noise = torch.randn(1, 32)
    y = gen(noise, current_depth=gen.depth - 1, alpha=1.0)
    assert y.shape == (1, 3, 64, 64)
    y = gen(noise, current_depth=gen.depth - 1, alpha=0.5)
    assert y.shape == (1, 3, 64, 64)
    y = gen(noise, current_depth=gen.depth - 1, alpha=0.0)
    assert y.shape == (1, 3, 64, 64)
    y = gen(noise, current_depth=gen.depth - 2, alpha=1.0)
    assert y.shape == (1, 3, 32, 32)
