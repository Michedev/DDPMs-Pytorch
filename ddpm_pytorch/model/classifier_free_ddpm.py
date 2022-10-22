from typing import Literal, List, Union, Optional

import pytorch_lightning as pl
import torch
import torchvision
from path import Path
from torch import nn
from torch.nn.functional import one_hot

from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler
from ddpm_pytorch.utils.distributions import x0_to_xt


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
        self.gen_images = Path('training_gen_images')
        self.gen_images.mkdir_p()

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        predict the score (noise) to transition from step t to t-1
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
        """
        train/validation step of DDPM. The logic is mostly taken from the original DDPM paper,
        except for the class conditioning part.
        """
        X, y = batch
        with torch.no_grad():
            X = X * 2 - 1  # normalize to -1, 1
        y = one_hot(y, self.num_classes).float()

        # dummy flags that with probability p_uncond, we train without class conditioning
        is_class_cond = torch.rand(size=(X.shape[0],1), device=X.device) >= self.p_uncond
        y = y * is_class_cond.float()  # set to zero the batch elements not class conditioned
        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)  # sample t uniformly from [0, T-1]
        t_expanded = t.reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)  # [bs, c, w, h]
        alpha_hat_t = self.alphas_hat[t_expanded] # get \hat{\alpha}_t
        x_t = x0_to_xt(X, alpha_hat_t, eps)  # go from x_0 to x_t in a single equation thanks to the step
        pred_eps = self(x_t, t / self.T, y) # predict the noise to transition from x_t to x_{t-1}
        loss = self.mse(eps, pred_eps) # compute the MSE between the predicted noise and the real noise

        # log every batch on validation set, otherwise log every self.logging_freq batches on training set
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
        """
        Generate a new sample starting from pure random noise sampled from a normal standard distribution
        :param batch_size: the generated batch size
        :param c: the class conditional matrix [batch_size, num_classes]. By default, it will be deactivated by passing a matrix of full zeroes
        :param T: the number of generation steps. By default, it will be the number of steps of the training
        :param get_intermediate_steps: if true, it will all return the intermediate steps of the generation
        :return: the generated image or the list of intermediate steps
        """
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
                # compute unconditioned noise
                eps = self(z_t, t / T, c)  # predict via nn the noise
            else:
                # compute class conditioned noise
                eps1 = (1 + self.w) * self(z_t, t / T, c)
                eps2 = self.w * self(z_t, t / T, c * 0)
                eps = eps1 - eps2
            alpha_t = self.alphas[t_expanded]
            z = torch.randn_like(z_t)
            alpha_hat_t = self.alphas_hat[t_expanded]
            # denoise step from x_t to x_{t-1} following the DDPM paper
            z_t = (z_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps) / (torch.sqrt(alpha_t)) + \
                  self.betas[t_expanded] * z
        z_t = (z_t + 1) / 2  # bring back to [0, 1]
        if get_intermediate_steps:
            steps.append(z_t)
        return z_t if not get_intermediate_steps else steps
