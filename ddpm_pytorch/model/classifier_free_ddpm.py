from random import random
from typing import Literal, List, Union, Optional

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.nn.functional import one_hot

from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler
from distributions import x0_to_xt, sigma_x_t


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
        return self._step(batch, batch_idx, 'valid')

    def _step(self, batch, batch_idx, dataset: Literal['train', 'valid']) -> torch.Tensor:
        X, y = batch
        with torch.no_grad():
            X = X * 2 - 1  # normalize to -1, 1
        y = one_hot(y, self.num_classes).float()
        is_class_cond = torch.rand(size=(X.shape[0],1), device=X.device) >= self.p_uncond
        y = y * is_class_cond.float()
        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)
        t_expanded = t.reshape(-1, 1, 1, 1)
        alpha_hat = self.alphas_hat[t_expanded]
        eps = torch.randn_like(X)  # [bs, c, w, h]
        x_t = x0_to_xt(X, alpha_hat, eps)  # go from x_0 to x_t with the formula
        pred_eps = self(x_t, t, y)
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
        return torch.optim.Adam(params=self.parameters())

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_hat = self.alphas_hat.to(self.device)

    def generate(self, batch_size: Optional[int] = None, c: Optional[torch.Tensor] = None, T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a batch of images via denoising diffusion probabilistic model
        :param batch_size: batch size of generated images. The default value is 1
        :param T: number of diffusion steps to generated images. The default value is the training diffusion steps
        :param get_intermediate_steps: return all the denoising steps instead of the final step output
        :return: The tensor [bs, c, w, h] of generated images or a list of tensors [bs, c, w, h] if get_intermediate_steps=True
        """
        batch_size = batch_size or 1
        if c is None:
            c = torch.zeros(batch_size, self.num_classes, device=self.device)
        T = T or self.T
        if get_intermediate_steps:
            steps = []
        X_noise = torch.randn(batch_size, self.input_channels,  # start with random noise sampled from N(0, 1)
                              self.width, self.height, device=self.device)
        for t in range(T - 1, -1, -1):
            if get_intermediate_steps:
                steps.append(X_noise)
            t = torch.LongTensor([t] * batch_size).to(self.device)
            eps = ((1 + self.w) * self(X_noise, t, c)) - (self.w * self(X_noise, t, c * 0))  # predict via nn the noise
            # if variational lower bound is present on the loss function hence v (the logit of variance) is trained
            # else the variance is taked fixed as in the original DDPM paper
            sigma = self.betas_hat[t].reshape(-1, 1, 1, 1)
            z = torch.randn_like(X_noise)
            if t == 0:
                z.fill_(0)
            alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
            X_noise = (X_noise - ((1 - alpha_t) / torch.sqrt(1 - self.alphas_hat[t].reshape(-1, 1, 1, 1))) * eps) / (torch.sqrt(alpha_t))\
                      + sigma * z  # denoise step from x_t to x_{t-1} following the DDPM paper. Differently from the
            # original paper, the variance is estimated via nn instead of be fixed, as in Improved DDPM paper
        X_noise = (X_noise + 1) / 2  # rescale from [-1, 1] to [0, 1]
        if get_intermediate_steps:
            steps.append(X_noise)
            return steps
        return X_noise

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
