from math import sqrt
from random import randint
from typing import Dict, List
import torch
import pytorch_lightning as pl
from torch import nn

from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler


def positional_embedding_vector(t: int, dim: int) -> torch.FloatTensor:
    """

    Args:
        t (int): time step
        dim (int): embedding size

    Returns: the transformer sinusoidal positional embedding vector

    """
    two_i = 2 * torch.arange(0, dim).unsqueeze(1)
    return torch.sin(t / torch.pow(10_000, two_i / dim))


def positional_embedding_matrix(T: int, dim: int) -> torch.FloatTensor:
    pos = torch.arange(0, T)
    two_i = 2 * torch.arange(0, dim).unsqueeze(1)
    return torch.sin(pos / torch.pow(10_000, two_i / dim))


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 time_embed_size: int, p_dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.groupnorm = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU()
        self.l_embedding = nn.Linear(time_embed_size, out_channels)
        self.out_layer = nn.Sequential(
            nn.GroupNorm(1, out_channels),
            nn.Dropout2d(p_dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU())

    def forward(self, x, time_embed):
        x = self.conv(x)
        h = self.groupnorm(x)
        time_embed = self.l_embedding(time_embed).view(time_embed.shape[0], time_embed.shape[1], 1, 1)
        h = h + time_embed
        return self.out_layer(h) + x


class DDPMUNet(pl.LightningModule):

    def __init__(self, channels: List[int], kernel_sizes: List[int], strides: List[int], paddings: List[int],
                 downsample: bool, p_dropouts: List[float], T: int, time_embed_size: int,
                 variance_scheduler: Scheduler, lambda_variational: float, width: int,
                 height: int):
        super().__init__()

        assert len(channels) == (len(kernel_sizes) + 1) == (len(strides) + 1) == (len(paddings) + 1) == \
               (len(p_dropouts) + 1), f'{len(channels)} == {(len(kernel_sizes) + 1)} == ' \
                                      f'{(len(strides) + 1)} == {(len(paddings) + 1)} == \
                                                              {(len(p_dropouts) + 1)}'
        self.channels = channels
        self.T = T
        self.time_embed_size = time_embed_size
        self.downsample_blocks = nn.ModuleList([
            ResBlock(channels[i], channels[i + 1], kernel_sizes[i], strides[i],
                     paddings[i], time_embed_size, p_dropouts[i]) for i in range(len(channels) - 1)
        ])

        self.downsample_blocks = nn.ModuleList([
            ResBlock(channels[i], channels[i + 1], kernel_sizes[i], strides[i],
                     paddings[i], time_embed_size, p_dropouts[i]) for i in range(len(channels) - 1)
        ])
        self.use_downsample = downsample
        self.downsample_op = nn.MaxPool2d(kernel_size=2)
        self.middle_block = ResBlock(channels[-1], channels[-1], kernel_sizes[-1], strides[-1],
                                     paddings[-1], time_embed_size, p_dropouts[-1])
        self.upsample_blocks = nn.ModuleList([
            ResBlock(2 * channels[-i - 1], channels[-i], kernel_sizes[-i - 1], strides[-i - 1],
                     paddings[-i - 1], time_embed_size, p_dropouts[-i - 1]) for i in range(len(channels) - 1)
        ])
        self.upsample_op = nn.UpsamplingNearest2d(size=2)
        self.var_scheduler = variance_scheduler
        self.lambda_variational = lambda_variational
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat()
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alpha_noise()
        self.variance = self.var_scheduler.get_variance()
        self.mse = nn.MSELoss()
        self.width = width
        self.height = height

    def forward(self, x: torch.FloatTensor, t: int) -> Dict:
        time_embedding = positional_embedding_vector(t, self.time_embed_size)
        hs = []
        h = x
        for i, downsample_block in enumerate(self.downsample_blocks):
            h = downsample_block(h, time_embedding)
            if self.use_downsample and i != (len(self.downsample_blocks) - 1):
                h = self.downsample_op(h)
                hs.append(h)
        for i, upsample_block in enumerate(self.upsample_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = upsample_block(h, time_embedding)
            if self.use_downsample and i != 0:
                h = self.upsample_op(h)
        return h

    def training_step(self, batch, batch_idx):
        t: int = randint(0, self.T - 1)
        alpha_hat = self.alphas_hat[t]
        eps = torch.randn_like(batch)
        x_t = sqrt(alpha_hat) * batch + sqrt(1 - alpha_hat) * eps
        pred_eps = self(x_t, t)
        loss = self.mse(eps, pred_eps)
        return dict(loss=loss)

    def sample(self):
        x = torch.randn(1, self.channels[0], self.width, self.height)
        for t in range(self.T, 0, -1):
            z = 0 if t > 1 else torch.randn_like(x)
            x = 1 / sqrt(self.alphas[t-1]) * \
                (x - ((1 - self.alphas[t-1]) / sqrt(1 - self.alphas_hat[t-1])) * self(x, t)) + self.variance[t] * z
        return x
