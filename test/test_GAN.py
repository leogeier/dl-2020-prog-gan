import torch

from gan.ConditionalGAN import ConditionalGAN
from gan.GAN import GAN


def test_instanciation():
    gan = GAN(depth=9, latent_size=512, lr=0.001, device=torch.device('cpu'))
    assert gan.depth == 9
    assert gan.latent_size == 512
    assert gan.device == torch.device('cpu')


def test_training(tmp_path):
    dataset = torch.utils.data.TensorDataset(torch.randn((1, 3, 64, 64)), torch.zeros(1,))
    gan = GAN(depth=5, latent_size=512, lr=0.001, device=torch.device('cpu'))
    gan.train(dataset, [1] * gan.depth, [1] * gan.depth, [0.5] * gan.depth,
              log_dir=tmp_path / "logs",
              sample_dir=tmp_path / "samples",
              model_dir=tmp_path / "model")
    assert len(list(tmp_path.iterdir())) == 3
    assert "gen_0_1_1.png" in [f.name for f in (tmp_path / "samples").iterdir()]
    assert "GAN_GEN_4.pth" in [f.name for f in (tmp_path / "model").iterdir()]
    assert "loss_0.log" in [f.name for f in (tmp_path / "logs").iterdir()]


def test_conditional_training(tmp_path):
    dataset = torch.utils.data.TensorDataset(torch.randn((1, 3, 64, 64)), torch.LongTensor([[0, 1, 0]]))
    gan = ConditionalGAN(num_attributes=3, depth=5, latent_size=512, lr=0.001, device=torch.device('cpu'))
    gan.train(dataset, [1] * gan.depth, [1] * gan.depth, [0.5] * gan.depth, num_samples=1,
              log_dir=tmp_path / "logs",
              sample_dir=tmp_path / "samples",
              model_dir=tmp_path / "model")
    assert len(list(tmp_path.iterdir())) == 3
    assert "attributes.txt" in [f.name for f in (tmp_path / "samples").iterdir()]
    assert "gen_0_1_1.png" in [f.name for f in (tmp_path / "samples").iterdir()]
    assert "GAN_GEN_4.pth" in [f.name for f in (tmp_path / "model").iterdir()]
    assert "loss_0.log" in [f.name for f in (tmp_path / "logs").iterdir()]
