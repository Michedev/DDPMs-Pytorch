import torch
from model import UNetTimeStep


def test_unet_same_width_height():
    x = torch.rand(2, 1, 28, 28)
    unet = UNetTimeStep([1, 32, 32, 64, 128], [3, 3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], True, [0.0, 0.0, 0.0, 0.0], 100)
    x_recon, v = unet(x, torch.tensor([0.25] * 2).view(-1))
    assert x.shape == x_recon.shape


def test_unet_different_width_height():
    x = torch.rand(2, 3, 32, 35)
    unet = UNetTimeStep([3, 32, 64, 128], [3, 3, 3], [1, 1, 1], [1, 1, 1], True, [0.0, 0.0, 0.0], 10)
    x_recon, v = unet(x, torch.tensor([0.25, 0.5]).view(-1))
    assert x.shape == x_recon.shape
