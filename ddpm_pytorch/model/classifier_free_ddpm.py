from random import random
from typing import Literal, List, Union, Optional

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.nn.functional import one_hot

from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler


class GaussianDDPMClassifierFreeGuidance(pl.LightningModule):

    def __init__(self, denoiser_module: nn.Module, T: int,
                 w: float, p_uncond: float, width: int,
                 height: int, input_channels: int, num_classes: int,
                 logging_freq: int, v: float, variance_scheduler: Scheduler):
        """
        :param denoiser_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, c)
        :param T: the amount of noising steps
        :param w: strength of class conditional sampling
        :param p_uncond: probability of train a batch without class conditioning
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        :param width: image width
        :param height: image height
        :param input_channels: image input channels
        :param num_classes: number of classes
        :param logging_freq: frequency of logging loss function during training
        :param v: generation variance parameter
        """
        assert 0.0 <= v <= 1.0, f'0.0 <= {v} <= 1.0'
        assert 0.0 <= w <= 1.0, f'0.0 <= {w} <= 1.0'
        assert 0.0 <= p_uncond <= 1.0, f'0.0 <= {p_uncond} <= 1.0'
        super().__init__()
        self.input_channels = input_channels
        self.denoiser_module = denoiser_module
        self.T = T
        self.w = w
        self.v = v
        self.var_scheduler = variance_scheduler
        self.betas = self.var_scheduler.get_betas()

        self.p_uncond = p_uncond
        self.mse = nn.MSELoss()
        self.width = width
        self.height = height
        self.logging_freq = logging_freq
        self.iteration = 0
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param x: input image [bs, c, w, h]
        :param t: time step [bs]
        :param c:  class [bs, num_classes]
        :return: the predicted noise to transition from t to t-1
        """
        return self.denoiser_module(x, t, c)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            batch_size = 32
            for i_c in range(self.num_classes):
                c = torch.zeros(batch_size, self.num_classes, device=self.device)
                c[:, i_c] = 1
                x_c = self.generate(batch_size, c)
                x_c = torchvision.utils.make_grid(x_c)
                self.logger.experiment.add_image(f'epoch_gen_val_images_class_{i_c}', x_c, self.current_epoch)
        return self._step(batch, batch_idx, 'valid')

    def _step(self, batch, batch_idx, dataset: Literal['train', 'valid']) -> torch.Tensor:
        X, y = batch
        with torch.no_grad():
            X = X * 2 - 1  # normalize to -1, 1
        y = one_hot(y, self.num_classes).float()
        is_class_uncond = random() < self.p_uncond
        if is_class_uncond:
            with torch.no_grad():
                y.fill_(0)  # null class
        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)
        t_expanded = t.reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)
        x_t = X * self._alpha_t(t_expanded) + self._sigma_t(t_expanded) * eps # go from x_0 to x_t with the formula
        pred_eps = self(x_t, t, y)
        loss = self.mse(eps, pred_eps)
        if dataset == 'valid' or (self.iteration % self.logging_freq) == 0:
            self.log(f'loss/{dataset}_loss', loss, on_step=True)
            self.log(f'loss/{dataset}_loss_{"uncond" if is_class_uncond else "cond"}', loss, on_step=True)
        self.iteration += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters())

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)

    def generate(self, batch_size: Optional[int] = None, c: Optional[torch.Tensor] = None, T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = T or self.T
        batch_size = batch_size or 1
        is_c_none = c is None
        if is_c_none:
            c = torch.zeros(batch_size, self.num_classes, device=self.device)
        if get_intermediate_steps:
            steps = []
        z_t = torch.randn(batch_size, self.input_channels,  # start with random noise sampled from N(0, 1)
                          self.width, self.height, device=self.device)
        for t in range(T - 1, -1, -1):
            if get_intermediate_steps:
                steps.append(z_t)
            t = torch.LongTensor([t] * batch_size).to(self.device)
            t_expanded = t.view(-1, 1, 1, 1)
            if is_c_none:
                eps = self(z_t, t, c)  # predict via nn the noise
            else:
                eps = (1 + self.w) * self(z_t, t,  c) - self.w * self(z_t, t, c * 0)
            x_t = (z_t - self._sigma_t(t_expanded) * eps) / self._alpha_t(t_expanded)
            if torch.any(t > 0):
                z_t = self._mu_t1_t_z_x(t_expanded, t_expanded-1, z_t, x_t) + \
                      self._sigma_t1_t_z_x(t_expanded, t_expanded-1) ** (1 - self.v) * \
                      self._sigma_t1_t_z_x(t_expanded, t_expanded-1) ** self.v * torch.randn_like(x_t)
            else:
                z_t = x_t
        z_t = (z_t + 1) / 2  # bring back to [0, 1]
        if get_intermediate_steps:
            steps.append(z_t)
        return z_t if not get_intermediate_steps else steps

    def _alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t].sigmoid()

    def _sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - self._alpha_t(t)

    def _mu_t1_t_z_x(self, t1, t, z, x):
        e_t_t1 = (t - t1).exp()
        alpha_t = self._alpha_t(t)
        return e_t_t1 * self._alpha_t(t1) / alpha_t * z + (1 - e_t_t1) * alpha_t * x

    def _sigma_t1_t_z_x(self, t1, t):
        return (1 - (self.betas[t] - self.betas[t1]).exp()) * self._sigma_t(t1)