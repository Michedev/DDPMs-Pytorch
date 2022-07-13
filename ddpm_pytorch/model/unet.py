from math import sqrt, log
from random import randint
from typing import Dict, List, Tuple
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

class ImageSelfAttention(nn.Module):

    def __init__(self, num_channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = num_channels
        self.heads = num_heads

        self.attn_layer = nn.MultiheadAttention(num_channels, num_heads=num_heads)

    def forward(self, x):
        """

        :param x: tensor with shape [batch_size, channels, width, height]
        :return: the attention output applied to the image with the shape [batch_size, channels, width, height]
        """
        b, c, w, h = x.shape
        x = x.reshape(b, w * h, c)
        attn_output = self.attn_layer(x, x, x)
        return attn_output.reshape(b, c, w, h)

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
        channels[0] += 1 # because the output is the image plus the estimated variance coefficients
        self.upsample_blocks = nn.ModuleList([
            ResBlock(2 * channels[-i - 1], channels[-i], kernel_sizes[-i - 1], strides[-i - 1],
                     paddings[-i - 1], time_embed_size, p_dropouts[-i - 1]) for i in range(len(channels) - 1)
        ])
        self.self_attn = ImageSelfAttention(channels[2])
        self.upsample_op = nn.UpsamplingNearest2d(size=2)
        self.var_scheduler = variance_scheduler
        self.lambda_variational = lambda_variational
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat()
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alpha_noise()
        self.betas = self.var_scheduler.get_betas()
        self.betas_hat = self.var_scheduler.get_betas_hat()
        self.mse = nn.MSELoss()
        self.width = width
        self.height = height

    def forward(self, x: torch.FloatTensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        time_embedding = positional_embedding_vector(t, self.time_embed_size)
        hs = []
        h = x
        for i, downsample_block in enumerate(self.downsample_blocks):
            h = downsample_block(h, time_embedding)
            if self.use_downsample and i != (len(self.downsample_blocks) - 1):
                h = self.downsample_op(h)
            if i == 2:
                h = self.self_attn(h)
            hs.append(h)
        for i, upsample_block in enumerate(self.upsample_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = upsample_block(h, time_embedding)
            if self.use_downsample and i != 0:
                h = self.upsample_op(h)
        x_recon, v = h[:, :3], h[:, 3:]
        return x_recon, v

    def training_step(self, batch, batch_idx):
        t: int = randint(0, self.T - 1)
        alpha_hat = self.alphas_hat[t]
        eps = torch.randn_like(batch)
        x_t = sqrt(alpha_hat) * batch + sqrt(1 - alpha_hat) * eps
        pred_eps = self(x_t, t)
        loss = self.mse(eps, pred_eps)  #todo add variational lower bound
        return dict(loss=loss)

    def variational_loss(self, x_t: torch.Tensor, x_0: torch.Tensor, model_noise: torch.Tensor, v: torch.Tensor, t: int):
        """
        Compute variational loss for time step t
        :param x_t: the image at step t obtained with closed form formula from x_0
        :param x_0: the input image
        :param model_noise: the unet predicted noise
        :param v: the unet predicted coefficients for the variance
        :param t: the time step
        :return: the pixel wise variational loss, with shape [batch_size, channels, width, height]
        """
        b, c, w, h = x_t.shape
        if t == 0:
            p = torch.distributions.Normal(self.mu_x_t(x_t, t, model_noise), self.sigma_x_t(v, t))
            return - p.log_prob(x_0)
        elif t == (self.T-1):
            p = torch.distributions.Normal(0, 1)
            q = torch.distributions.Normal(sqrt(self.alphas_hat[t]) * x_0, (1 - self.alphas_hat[t]))
            return torch.kl_div(q, p)
        q = torch.distributions.Normal(self.mu_hat_xt_x0(x_t, x_0, t), self.sigma_hat(w * h, t, x_t.device))
        p = torch.distributions.Normal(self.mu_x_t(x_t, t, model_noise), self.sigma_x_t(v, t))
        return torch.distributions.kl_divergence(q, p)

    def mu_x_t(self, x_t: torch.Tensor, t: int, model_noise: torch.Tensor) -> torch.Tensor:
        return 1 / sqrt(self.alphas[t]) * (x_t - self.betas[t] / sqrt(1 - self.alphas_hat[t]) * model_noise)

    def sigma_x_t(self, v: torch.Tensor, t: int) -> torch.Tensor:
        return torch.exp(v * log(self.betas[t]) + (1 - v) * log(self.betas_hat[t]))

    def mu_hat_xt_x0(self, x_t: torch.Tensor, x_0: torch.Tensor, t: int):
        return sqrt(self.alphas_hat[t-1]) * self.betas[t] / (1 - self.alphas_hat[t]) * x_0 +\
               sqrt(self.alphas[t]) * (1 - self.alphas_hat[t-1]) / (1 - self.alphas_hat[t]) * x_t

    def sigma_hat(self, num_pixels: int, t: int, device: str) -> float:
        return self.betas_hat[t]

    def sample(self):
        x = torch.randn(1, self.channels[0], self.width, self.height)
        for t in range(self.T, 0, -1):
            z = 0 if t > 1 else torch.randn_like(x)
            x = 1 / sqrt(self.alphas[t-1]) * \
                (x - ((1 - self.alphas[t-1]) / sqrt(1 - self.alphas_hat[t-1])) * self(x, t)) + self.variance[t] * z
        return x
