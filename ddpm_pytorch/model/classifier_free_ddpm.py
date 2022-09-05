from random import random
from typing import Literal, List, Union, Optional

import pytorch_lightning as pl
import torch
import torchvision
from path import Path
from torch import nn
from torch.nn.functional import one_hot

from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler
from ddpm_pytorch.distributions import x0_to_xt


class GaussianDDPMClassifierFreeGuidance(pl.LightningModule):
    """
    Implementation of "Classifier-Free Diffusion Guidance"
    """

    def __init__(self, denoiser_module: nn.Module, T: int,
                 w: float, p_uncond: float, width: int,
                 height: int, input_channels: int, num_classes: int,
                 logging_freq: int, v: float, variance_scheduler: Scheduler):
        """
        :param denoiser_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, c)
        :param T: the amount of noising steps
        :param w: strength of class guidance
        :param p_uncond: probability of train a batch without class conditioning
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        :param width: image width
        :param height: image height
        :param input_channels: image input channels
        :param num_classes: number of classes
        :param logging_freq: frequency of logging loss function during training
        :param v: generative variance hyper-parameter
        """
        assert 0.0 <= v <= 1.0, f'0.0 <= {v} <= 1.0'
        assert 0.0 <= w, f'0.0 <= {w}'
        assert 0.0 <= p_uncond <= 1.0, f'0.0 <= {p_uncond} <= 1.0'
        super().__init__()
        self.input_channels = input_channels
        self.denoiser_module = denoiser_module
        self.T = T
        self.w = w
        self.v = v
        self.var_scheduler = variance_scheduler
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)

        self.p_uncond = p_uncond
        self.mse = nn.MSELoss()
        self.width = width
        self.height = height
        self.logging_freq = logging_freq
        self.iteration = 0
        self.num_classes = num_classes
        self.gen_images = Path('gen_images')
        self.gen_images.mkdir_p()

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
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            batch_size = 32
            for i_c in range(self.num_classes):
                c = torch.zeros(batch_size, self.num_classes, device=self.device)
                c[:, i_c] = 1
                x_c = self.generate(batch_size, c)
                x_c = torchvision.utils.make_grid(x_c)
                self.logger.experiment.add_image(f'epoch_gen_val_images_class_{i_c}', x_c, self.current_epoch)
                torchvision.utils.save_image(x_c, self.gen_images / f'epoch_{self.current_epoch}_class_{i_c}.png')

        return self._step(batch, batch_idx, 'valid')

    def _step(self, batch, batch_idx, dataset: Literal['train', 'valid']) -> torch.Tensor:
        X, y = batch
        with torch.no_grad():
            X = X * 2 - 1  # normalize to -1, 1
        y = one_hot(y, self.num_classes).float()
        is_class_cond = torch.rand(size=(X.shape[0], 1), device=X.device) >= self.p_uncond
        y = y * is_class_cond.float()
        t = torch.randint(0, self.T - 1, (X.shape[0], 1), device=X.device)
        t_expanded = t.reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)  # [bs, c, w, h]
        alpha_hat_t = self.alphas_hat[t_expanded]
        x_t = x0_to_xt(X, alpha_hat_t, eps)  # go from x_0 to x_t with the formula
        pred_eps = self(x_t, t / self.T, y)
        loss = self.mse(eps, pred_eps)
        if dataset == 'valid' or (self.iteration % self.logging_freq) == 0:
            self.log(f'loss/{dataset}_loss', loss, on_step=True)
            if dataset == 'train':
                norm_params = sum(
                    [torch.norm(p.grad) for p in self.parameters() if
                     hasattr(p, 'grad') and p.grad is not None])
                self.log('grad_norm', norm_params)
            self.logger.experiment.add_image(f'{dataset}_pred_score', eps[0], self.iteration)
            with torch.no_grad():
                self.log(f'noise/{dataset}_mean_eps', eps.mean(), on_step=True)
                self.log(f'noise/{dataset}_std_eps', eps.flatten(1).std(dim=1).mean(), on_step=True)
                self.log(f'noise/{dataset}_max_eps', eps.max(), on_step=True)
                self.log(f'noise/{dataset}_min_eps', eps.min(), on_step=True)

        self.iteration += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=1e-4)

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_hat = self.alphas_hat.to(self.device)

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
        for t in range(T - 1, 0, -1):
            if get_intermediate_steps:
                steps.append(z_t)
            t = torch.LongTensor([t] * batch_size).to(self.device).view(-1, 1)
            t_expanded = t.view(-1, 1, 1, 1)
            if is_c_none:
                eps = self(z_t, t / T, c)  # predict via nn the noise
            else:
                eps1 = (1 + self.w) * self(z_t, t / T, c)
                eps2 = self.w * self(z_t, t / T, c * 0)
                eps = eps1 - eps2
            alpha_t = self.alphas[t_expanded]
            z = torch.randn_like(z_t)
            alpha_hat_t = self.alphas_hat[t_expanded]
            z_t = 1 / (torch.sqrt(alpha_t)) * \
                  (z_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps) + \
                  self.betas[
                      t_expanded] * z  # denoise step from x_t to x_{t-1} following the DDPM paper. Differently from the
        z_t = (z_t + 1) / 2  # bring back to [0, 1]
        if get_intermediate_steps:
            steps.append(z_t)
        return z_t if not get_intermediate_steps else steps

    def _alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t].sigmoid()

    def _sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - self._alpha_t(t)

    def _mu_t1_t_z_x(self, t1, t, z, x):
        e_t_t1 = (self.betas[t] - self.betas[t1]).exp()
        alpha_t = self._alpha_t(t)
        return e_t_t1 * self._alpha_t(t1) / alpha_t * z + (1 - e_t_t1) * alpha_t * x

    def _sigma_t1_t_z_x(self, t1, t2):
        return (1 - (self.betas[t1] - self.betas[t2]).exp()) * self._sigma_t(t1)

    def _sigma_hat_t1_t_z_x(self, t1, t2):
        return (1 - (self.betas[t2] - self.betas[t1]).exp()) * self._sigma_t(t1)
