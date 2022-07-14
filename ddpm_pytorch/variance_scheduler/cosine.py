from math import pi

import torch

from ddpm_pytorch.variance_scheduler.abs_var_scheduler import Scheduler

class CosineScheduler(Scheduler):

    def __init__(self, T: int, s: float = 0.0008):
        self.T = T
        self._alpha_hats = self.f(torch.arange(self.T), T, s)
        self._alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, 1, 0)
        self._alpha_hats_t_minus_1[0] = self._alpha_hats_t_minus_1[1]
        self._betas = 1.0 - self._alpha_hats / self._alpha_hats_t_minus_1
        self._alphas = 1.0 - self._betas
        self._betas_hat = (1 - self._alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas

    def f(self, t: torch.Tensor, T: int, s: float, clip_max_value=0.999):
        return torch.minimum(torch.cos((t / T + s) / (1 + s) * pi / 2.0), torch.Tensor([clip_max_value]))

    def get_alpha_hat(self):
        return self._alpha_hats

    def get_alpha_noise(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_betas_hat(self):
        return self._betas_hat