from math import sqrt
from typing import Tuple, Optional, Union, List

import pytorch_lightning as pl
import torch
import torchvision.utils
from torch import nn

from ddpm_pytorch.distributions import sigma_x_t, mu_x_t, mu_hat_xt_x0, sigma_hat_xt_x0, x0_to_xt
from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler


class GaussianDDPM(pl.LightningModule):
    """
    Gaussian De-noising Diffusion Probabilistic Model
    This class implements
    """

    def __init__(self, denoiser_module: nn.Module, T: int,
                 variance_scheduler: Scheduler, lambda_variational: float, width: int,
                 height: int, input_channels: int, log_loss: int, vlb: bool):
        """
        :param denoiser_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, t)
        :param T: the amount of noising steps
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        :param lambda_variational: the coefficient in from of variational loss
        :param width: image width
        :param height: image height
        :param input_channels: image input channels
        :param log_loss: frequency of logging loss function during training
        :param vlb: true to include the variational lower bound into the loss function
        """
        super().__init__()
        self.input_channels = input_channels
        self.denoiser_module = denoiser_module
        self.T = T

        self.var_scheduler = variance_scheduler
        self.lambda_variational = lambda_variational
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alpha_noise().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)
        self.mse = nn.MSELoss()
        self.width = width
        self.height = height
        self.log_loss = log_loss
        self.vlb = vlb
        self.iteration = 0

    def forward(self, x: torch.FloatTensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.denoiser_module(x, t)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        X, y = batch
        with torch.no_grad():
            X = X * 2 - 1  # map image values from [0, 1] to [-1, 1]
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],),
                                        device=X.device)  # todo add importance sampling
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)
        x_t = x0_to_xt(X, alpha_hat, eps)  # go from x_0 to x_t with the formula
        pred_eps, v = self(x_t, t)
        loss = 0.0
        noise_loss = self.mse(eps, pred_eps)
        loss = loss + noise_loss
        if self.vlb:
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(dim=0).sum()
            loss = loss + loss_vlb
        if (self.iteration % self.log_loss) == 0:
            self.log('loss/train_loss', loss, on_step=True)
            if self.vlb:
                self.log('loss/train_loss_noise', noise_loss, on_step=True)
                self.log('loss/train_loss_vlb', loss_vlb, on_step=True)
            norm_params = sum(
                [torch.norm(p.grad) for p in self.parameters() if
                 hasattr(p, 'grad') and p.grad is not None])
            self.log('grad_norm', norm_params)
        self.iteration += 1
        return dict(loss=loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if batch_idx == 0:
            gen_images = self.generate(batch_size=16)
            gen_images = torchvision.utils.make_grid(gen_images)
            self.logger.experiment.add_image('gen_val_images', gen_images, self.current_epoch)
        X, y = batch
        with torch.no_grad():
            X = X * 2 - 1
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],),
                                        device=X.device)  # todo replace this with importance sampling
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)
        x_t = x0_to_xt(X, alpha_hat, eps)
        pred_eps, v = self(x_t, t)
        loss = self.mse(eps, pred_eps)
        if self.vlb:
            eps_loss = loss
            self.log('loss/val_eps_loss', eps_loss, on_step=True)
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t) \
                .mean(dim=0).sum()
            self.log('loss/val_vlb_loss', loss_vlb, on_step=True)
            loss = loss + loss_vlb
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
        t_eq_0 = (t == 0).reshape(-1, 1, 1, 1)
        if torch.any(t_eq_0):
            p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas),
                                           sigma_x_t(v, t, self.betas_hat, self.betas))
            vlb += - p.log_prob(x_0) * t_eq_0.float()
        t_eq_last = (t == (self.T - 1)).reshape(-1, 1, 1, 1)
        if torch.any(t_eq_last):
            p = torch.distributions.Normal(0, 1)
            q = torch.distributions.Normal(sqrt(self.alphas_hat[t]) * x_0, (1 - self.alphas_hat[t]))
            vlb = torch.distributions.kl_divergence(q, p) * t_eq_last
        q = torch.distributions.Normal(mu_hat_xt_x0(x_t, x_0, t, self.alphas_hat, self.alphas, self.betas),
                                       sigma_hat_xt_x0(t, self.betas_hat))  # q(x_{t-1} | x_t, x_0)
        p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas).detach(),
                                       sigma_x_t(v, t, self.betas_hat, self.betas)) # p(x_t | x_{t-1})
        vlb += torch.distributions.kl_divergence(q, p) * (~t_eq_last).float() * (~t_eq_0).float()
        return vlb

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters())

    def generate(self, batch_size: Optional[int] = None, T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a batch of images via denoising diffusion probabilistic model
        :param batch_size: batch size of generated images. The default value is 1
        :param T: number of diffusion steps to generated images. The default value is the training diffusion steps
        :param get_intermediate_steps: return all the denoising steps instead of the final step output
        :return: The tensor [bs, c, w, h] of generated images or a list of tensors [bs, c, w, h] if get_intermediate_steps=True
        """
        batch_size = batch_size or 1
        T = T or self.T
        self.alphas_hat: torch.FloatTensor = self.alphas_hat.to(self.device)
        self.alphas: torch.FloatTensor = self.alphas.to(self.device)
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
        if get_intermediate_steps:
            steps = []
        X_noise = torch.randn(batch_size, self.input_channels,   # start with random noise sampled from N(0, 1)
                              self.width, self.height, device=self.device)
        for t in range(T - 1, -1, -1):
            if get_intermediate_steps:
                steps.append(X_noise)
            t = torch.LongTensor([t]).to(self.device)
            eps, v = self.denoiser_module(X_noise, t)  # predict via nn the noise
            # if variational lower bound is present on the loss function hence v (the logit of variance) is trained
            # else the variance is taked fixed as in the original DDPM paper
            sigma = sigma_x_t(v, t, self.betas_hat, self.betas) if self.vlb else self.betas_hat[t].reshape(-1, 1, 1, 1)
            z = torch.randn_like(X_noise)
            if t == 0:
                z.fill_(0)
            alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
            X_noise = 1 / (torch.sqrt(alpha_t)) * \
                      (X_noise - ((1 - alpha_t) / torch.sqrt(1 - self.alphas_hat[t].reshape(-1, 1, 1, 1))) * eps) + \
                      sigma * z  # denoise step from x_t to x_{t-1} following the DDPM paper. Differently from the
                                 # original paper, the variance is estimated via nn instead of be fixed, as in Improved DDPM paper
        X_noise = (X_noise + 1) / 2  # rescale from [-1, 1] to [0, 1]
        if get_intermediate_steps:
            steps.append(X_noise)
            return steps
        return X_noise
