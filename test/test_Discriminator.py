import torch

from gan.Discriminator import Discriminator, DisFinalBlock, DisConvolutionalBlock


def test_depth():
    dis = Discriminator(5, 32)
    assert dis.depth == 5
    assert len(dis.blocks) == 4


def test_final_shape():
    block = DisFinalBlock(feature_size=512)
    x = torch.randn(1, 512, 4, 4)
    y = block(x)
    assert y.shape == (1,)


def test_conv_shape():
    block = DisConvolutionalBlock(in_channels=16, out_channels=32)
    x = torch.randn(1, 16, 8, 8)
    y = block(x)
    assert y.shape == (1, 32, 4, 4)


def test_overall_shape():
    dis = Discriminator(5, 32)
    img = torch.randn(1, 3, 64, 64)
    y = dis(img, current_depth=dis.depth - 1, alpha=1.0)
    assert y.shape == (1,)
    y = dis(img, current_depth=dis.depth - 1, alpha=0.5)
    assert y.shape == (1,)
    y = dis(img, current_depth=dis.depth - 1, alpha=0.0)
    assert y.shape == (1,)
    img = torch.randn(1, 3, 32, 32)
    y = dis(img, current_depth=dis.depth - 2, alpha=1.0)
    assert y.shape == (1,)
