from math import sqrt, log
from random import randint
from typing import Dict, List, Tuple, Optional

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler

# import tensorguard as tg
from distributions import mu_hat_xt_x0, mu_x_t, sigma_x_t, sigma_hat


def positional_embedding_vector(t: int, dim: int) -> torch.FloatTensor:
    """

    Args:
        t (int): time step
        dim (int): embedding size

    Returns: the transformer sinusoidal positional embedding vector

    """
    two_i = 2 * torch.arange(0, dim)
    return torch.sin(t / torch.pow(10_000, two_i / dim)).unsqueeze(0)


def positional_embedding_matrix(T: int, dim: int) -> torch.FloatTensor:
    pos = torch.arange(0, T)
    two_i = 2 * torch.arange(0, dim)
    return torch.sin(pos / torch.pow(10_000, two_i / dim)).unsqueeze(0)


class ResBlockTimeEmbed(nn.Module):

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
        time_embed = self.l_embedding(time_embed)
        time_embed = time_embed.view(time_embed.shape[0], time_embed.shape[1], 1, 1)
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
        attn_output, _ = self.attn_layer(x, x, x)
        return attn_output.reshape(b, c, w, h)


class UNetTimeStep(nn.Module):

    def __init__(self, channels: List[int], kernel_sizes: List[int], strides: List[int], paddings: List[int],
                 downsample: bool, p_dropouts: List[float], time_embed_size: int):
        super().__init__()
        assert len(channels) == (len(kernel_sizes) + 1) == (len(strides) + 1) == (len(paddings) + 1) == \
               (len(p_dropouts) + 1), f'{len(channels)} == {(len(kernel_sizes) + 1)} == ' \
                                      f'{(len(strides) + 1)} == {(len(paddings) + 1)} == \
                                                              {(len(p_dropouts) + 1)}'
        self.channels = channels
        self.T = T
        self.time_embed_size = time_embed_size
        self.downsample_blocks = nn.ModuleList([
            ResBlockTimeEmbed(channels[i], channels[i + 1], kernel_sizes[i], strides[i],
                              paddings[i], time_embed_size, p_dropouts[i]) for i in range(len(channels) - 1)
        ])

        self.downsample_blocks = nn.ModuleList([
            ResBlockTimeEmbed(channels[i], channels[i + 1], kernel_sizes[i], strides[i],
                              paddings[i], time_embed_size, p_dropouts[i]) for i in range(len(channels) - 1)
        ])
        self.use_downsample = downsample
        self.downsample_op = nn.MaxPool2d(kernel_size=2)
        self.middle_block = ResBlockTimeEmbed(channels[-1], channels[-1], kernel_sizes[-1], strides[-1],
                                              paddings[-1], time_embed_size, p_dropouts[-1])
        channels[0] += 1  # because the output is the image plus the estimated variance coefficients
        self.upsample_blocks = nn.ModuleList([
            ResBlockTimeEmbed((2 if i != 0 else 1) * channels[-i - 1], channels[-i - 2], kernel_sizes[-i - 1],
                              strides[-i - 1],
                              paddings[-i - 1], time_embed_size, p_dropouts[-i - 1]) for i in range(len(channels) - 1)
        ])
        self.self_attn = ImageSelfAttention(channels[2])

    def forward(self, x: torch.FloatTensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_channels = x.shape[1]
        # tg.guard(x, "B, C, W, H")
        time_embedding = positional_embedding_vector(t, self.time_embed_size)
        hs = []
        h = x
        for i, downsample_block in enumerate(self.downsample_blocks):
            h = downsample_block(h, time_embedding)
            if i == 2:
                h = self.self_attn(h)
            if i != (len(self.downsample_blocks) - 1): hs.append(h)
            if self.use_downsample and i != (len(self.downsample_blocks) - 1):
                h = self.downsample_op(h)
        h = self.middle_block(h, time_embedding)
        for i, upsample_block in enumerate(self.upsample_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = upsample_block(h, time_embedding)
            if self.use_downsample and (i != (len(self.upsample_blocks) - 1)):
                h = F.interpolate(h, size=hs[-i - 1].shape[-1], mode='nearest')
        x_recon, v = h[:, :x_channels], h[:, x_channels:]
        # tg.guard(x_recon, "B, C, W, H")
        # tg.guard(v, "B, C, W, H")
        return x_recon, v


class DDPM(pl.LightningModule):

    def __init__(self, denoiser_module: nn.Module, T: int,
                 variance_scheduler: Scheduler, lambda_variational: float, width: int,
                 height: int, log_loss: int):
        super().__init__()
        self.denoiser_module = denoiser_module
        self.T = T

        self.var_scheduler = variance_scheduler
        self.lambda_variational = lambda_variational
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat()
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alpha_noise()
        self.betas = self.var_scheduler.get_betas()
        self.betas_hat = self.var_scheduler.get_betas_hat()
        self.mse = nn.MSELoss()
        self.width = width
        self.height = height
        self.log_loss = log_loss
        self.iteration = 0

    def forward(self, x: torch.FloatTensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.denoiser_module(x, t)

    def training_step(self, batch, batch_idx):
        X, y = batch
        t: int = randint(0, self.T - 1)  # todo replace this with importance sampling
        alpha_hat = self.alphas_hat[t]
        eps = torch.randn_like(X)
        x_t = sqrt(alpha_hat) * X + sqrt(1 - alpha_hat) * eps
        pred_eps, v = self(x_t, t)
        loss = self.mse(eps, pred_eps) + self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(
            dim=0).sum()
        if (self.iteration % self.log_loss) == 0:
            self.log('loss/train_loss', loss, on_step=True)
        self.iteration += 1
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        t: int = randint(0, self.T - 1)  # todo replace this with importance sampling
        alpha_hat = self.alphas_hat[t]
        eps = torch.randn_like(X)
        x_t = sqrt(alpha_hat) * X + sqrt(1 - alpha_hat) * eps
        pred_eps, v = self(x_t, t)
        loss = self.mse(eps, pred_eps) + self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(
            dim=0).sum()
        self.log('loss/val_loss', loss, on_step=True)
        return dict(loss=loss)

    def variational_loss(self, x_t: torch.Tensor, x_0: torch.Tensor, model_noise: torch.Tensor, v: torch.Tensor,
                         t: int):
        """
        Compute variational loss for time step t
        :param x_t: the image at step t obtained with closed form formula from x_0
        :param x_0: the input image
        :param model_noise: the unet predicted noise
        :param v: the unet predicted coefficients for the variance
        :param t: the time step
        :return: the pixel-wise variational loss, with shape [batch_size, channels, width, height]
        """
        if t == 0:
            p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas),
                                           sigma_x_t(v, t, self.betas_hat, self.betas))
            return - p.log_prob(x_0)
        elif t == (self.T - 1):
            p = torch.distributions.Normal(0, 1)
            q = torch.distributions.Normal(sqrt(self.alphas_hat[t]) * x_0, (1 - self.alphas_hat[t]))
            return torch.distributions.kl_divergence(q, p)
        q = torch.distributions.Normal(mu_hat_xt_x0(x_t, x_0, t, self.alphas_hat, self.alphas, self.betas),
                                       sigma_hat(t, self.betas_hat))
        p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas),
                                       sigma_x_t(v, t, self.betas_hat, self.betas))
        return torch.distributions.kl_divergence(q, p)

    def sample(self):
        x = torch.randn(1, self.channels[0], self.width, self.height)
        for t in range(self.T, 0, -1):
            z = 0 if t > 1 else torch.randn_like(x)
            x = 1 / sqrt(self.alphas[t - 1]) * \
                (x - ((1 - self.alphas[t - 1]) / sqrt(1 - self.alphas_hat[t - 1])) * self(x, t)) + self.variance[t] * z
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=1e-4)

    def generate(self, batch_size: int = None, T: Optional[int] = None):
        batch_size = batch_size or 1
        T = T or self.T
        X_noise = torch.randn(batch_size, self.channels[0], self.width, self.height)
        for t in range(T - 1, 0, -1):
            eps, v = self.denoiser_module(X_noise, T)
            sigma = self.sigma_x_t(v, t)
            if t == 0:
                sigma.fill_(0)
            alpha_t = self.alphas[t]
            X_noise = 1 / (sqrt(alpha_t)) * (X_noise - ((1 - alpha_t) / sqrt(1 - self.alphas_hat[t])) * eps) + sigma
        return X_noise
