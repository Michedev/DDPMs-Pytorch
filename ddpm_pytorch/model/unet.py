import math
from math import sqrt, log
from random import randint
from typing import Dict, List, Tuple, Optional

import hydra
import torch
import pytorch_lightning as pl
import torchvision.utils
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler

# import tensorguard as tg
from distributions import mu_hat_xt_x0, mu_x_t, sigma_x_t, sigma_hat_xt_x0


def positional_embedding_vector(t: int, dim: int) -> torch.FloatTensor:
    """

    Args:
        t (int): time step
        dim (int): embedding size

    Returns: the transformer sinusoidal positional embedding vector

    """
    two_i = 2 * torch.arange(0, dim)
    return torch.sin(t / torch.pow(10_000, two_i / dim)).unsqueeze(0)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlockTimeEmbed(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 time_embed_size: int, p_dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.groupnorm = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU()
        self.l_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_size, out_channels)
        )
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
        self.dropouts = nn.ModuleList([nn.Dropout2d(p) for p in p_dropouts])
        self.p_dropouts = p_dropouts
        self.self_attn = ImageSelfAttention(channels[2])
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_size, self.time_embed_size),
            nn.SiLU(),
            nn.Linear(self.time_embed_size, self.time_embed_size),
        )

    def forward(self, x: torch.FloatTensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_channels = x.shape[1]
        # tg.guard(x, "B, C, W, H")
        time_embedding = self.time_embed(timestep_embedding(t, self.time_embed_size))
        hs = []
        h = x
        for i, downsample_block in enumerate(self.downsample_blocks):
            h = downsample_block(h, time_embedding)
            if i == 2:
                h = self.self_attn(h)
            h = self.dropouts[i](h)
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


class GaussianDDPM(pl.LightningModule):

    def __init__(self, denoiser_module: nn.Module, T: int,
                 variance_scheduler: Scheduler, lambda_variational: float, width: int,
                 height: int, input_channels: int, log_loss: int):
        """
        :param denoiser_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, t)
        :param T: the amount of noising steps
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        :param lambda_variational: the coefficient in from of variational loss
        :param width: image width
        :param height: image height
        :param input_channels: image input channels
        :param log_loss: frequency of logging loss function during training
        """
        super().__init__()
        self.input_channels = input_channels
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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        X, y = batch
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],),
                                        device=X.device)  # todo replace this with importance sampling
        alpha_hat = self.alphas_hat[t]
        eps = torch.randn_like(X)
        x_t = torch.sqrt(alpha_hat) * X + torch.sqrt(1 - alpha_hat) * eps
        pred_eps, v = self(x_t, t)
        loss = self.mse(eps, pred_eps) + self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(
            dim=0).sum()
        if (self.iteration % self.log_loss) == 0:
            self.log('loss/train_loss', loss, on_step=True)
            norm_params = sum(
                [torch.norm(p.grad) for p in self.parameters() if
                 hasattr(p, 'grad') and p.grad is not None])
            self.log('grad_norm', norm_params)
        self.iteration += 1
        return dict(loss=loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if batch_idx == 0:
            gen_images = self.generate(32)
            gen_images = torchvision.utils.make_grid(gen_images)
            self.logger.experiment.add_image('gen_val_images', gen_images, self.current_epoch)
        X, y = batch
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],),
                                        device=X.device)  # todo replace this with importance sampling
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)
        x_t = torch.sqrt(alpha_hat) * X + torch.sqrt(1 - alpha_hat) * eps
        pred_eps, v = self(x_t, t)
        loss = self.mse(eps, pred_eps) + self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t) \
            .mean(dim=0).sum()
        self.log('loss/val_loss', loss, on_step=True)
        return dict(loss=loss)

    def variational_loss(self, x_t: torch.Tensor, x_0: torch.Tensor,
                         model_noise: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """
        Compute variational loss for time step t
        :param x_t: the image at step t obtained with closed form formula from x_0
        :param x_0: the input image
        :param model_noise: the unet predicted noise
        :param v: the unet predicted coefficients for the variance
        :param t: the time step
        :return: the pixel-wise variational loss, with shape [batch_size, channels, width, height]
        """
        vlb = 0.0
        t_eq_0 = t == 0
        if torch.any(t_eq_0):
            p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas),
                                           sigma_x_t(v, t, self.betas_hat, self.betas))
            vlb += - p.log_prob(x_0) * t_eq_0.float()
        t_eq_last = t == (self.T - 1)
        if torch.any(t_eq_last):
            p = torch.distributions.Normal(0, 1)
            q = torch.distributions.Normal(sqrt(self.alphas_hat[t]) * x_0, (1 - self.alphas_hat[t]))
            vlb = torch.distributions.kl_divergence(q, p) * t_eq_last
        q = torch.distributions.Normal(mu_hat_xt_x0(x_t, x_0, t, self.alphas_hat, self.alphas, self.betas),
                                       sigma_hat_xt_x0(t, self.betas_hat))  # q(x_{t-1} | x_t, x_0)
        p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas).detach(),
                                       sigma_x_t(v, t, self.betas_hat, self.betas)) # p(x_t | x_{t-1})
        vlb += torch.distributions.kl_divergence(q, p) * (~t_eq_last) * (~t_eq_0)
        return vlb

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=1e-4)

    def generate(self, batch_size: int = None, T: Optional[int] = None):
        batch_size = batch_size or 1
        T = T or self.T
        X_noise = torch.randn(batch_size, self.input_channels, self.width, self.height)
        for t in range(T - 1, -1, -1):
            t = torch.LongTensor([t])
            eps, v = self.denoiser_module(X_noise, t)
            sigma = sigma_x_t(v, t, self.betas_hat, self.betas)
            z = torch.randn_like(X_noise)
            if t == 0:
                z.fill_(0)
            alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
            X_noise = 1 / (torch.sqrt(alpha_t)) * (X_noise - ((1 - alpha_t) / torch.sqrt(1 - self.alphas_hat[t].reshape(-1, 1, 1, 1))) * eps) + sigma * z
        return X_noise
