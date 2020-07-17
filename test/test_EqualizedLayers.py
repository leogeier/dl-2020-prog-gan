import torch

from gan.EqualizedLayers import PixelwiseNormalization, EqualizedConv2d, EqualizedDeconv2d, EqualizedLinear


def test_pixnorm():
    norm = PixelwiseNormalization()
    x = torch.arange(54, dtype=torch.float32).reshape(2, 3, 3, 3)
    y = norm(x)
    solution = torch.tensor([[[[0.0000, 0.0806, 0.1512],
                               [0.2132, 0.2679, 0.3162],
                               [0.3592, 0.3976, 0.4320]],

                              [[0.7746, 0.8058, 0.8315],
                               [0.8528, 0.8705, 0.8854],
                               [0.8980, 0.9087, 0.9179]],

                              [[1.5492, 1.5311, 1.5119],
                               [1.4924, 1.4732, 1.4546],
                               [1.4368, 1.4199, 1.4039]]],

                             [[[0.7348, 0.7423, 0.7493],
                               [0.7559, 0.7622, 0.7682],
                               [0.7740, 0.7794, 0.7846]],

                              [[0.9798, 0.9808, 0.9818],
                               [0.9827, 0.9835, 0.9843],
                               [0.9850, 0.9857, 0.9863]],

                              [[1.2247, 1.2194, 1.2143],
                               [1.2095, 1.2048, 1.2004],
                               [1.1961, 1.1920, 1.1881]]]])

    assert torch.all(torch.isclose(y, solution, rtol=1e-3, atol=1e-5)).item()


def test_equalizedconv2d():
    x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    conv = EqualizedConv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    assert conv.weight.shape == (3, 2, 3, 3)
    assert torch.all(conv.bias == torch.zeros(3)).item()
    assert conv(x).shape == (1, 3, 4, 4)


def test_equalizeddeconv2d():
    x = torch.arange(2, dtype=torch.float32).reshape(1, 2, 1, 1)
    conv = EqualizedDeconv2d(in_channels=2, out_channels=3, kernel_size=4)
    assert conv.weight.shape == (2, 3, 4, 4)
    assert torch.all(conv.bias == torch.zeros(3)).item()
    assert conv(x).shape == (1, 3, 4, 4)
