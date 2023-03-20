from math import pi

import torch

from variance_scheduler.abs_var_scheduler import Scheduler

class CosineScheduler(Scheduler):


    clip_max_value = torch.Tensor([0.999])

    def __init__(self, T: int, s: float = 0.0008):
        """
        Cosine variance scheduler.
        The equation for the variance is:
            alpha_hat = min(cos((t / T + s) / (1 + s) * pi / 2)^2, 0.999)
        The equation for the beta is:
            beta = 1 - (alpha_hat(t) / alpha_hat(t - 1))
        The equation for the beta_hat is:
            beta_hat = (1 - alpha_hat(t - 1)) / (1 - alpha_hat(t)) * beta(t)
        """
        self.T = T
        self._alpha_hats = self.f(torch.arange(self.T), T, s)
        self._alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, 1, 0) # shift by 1
        self._alpha_hats_t_minus_1[0] = self._alpha_hats_t_minus_1[1]  # to remove first NaN value
        self._betas = 1.0 - self._alpha_hats / self._alpha_hats_t_minus_1
        self._alphas = 1.0 - self._betas
        self._betas_hat = (1 - self._alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas

    def f(self, t: torch.Tensor, T: int, s: float):
        return torch.minimum(torch.cos((t / T + s) / (1 + s) * pi / 2.0).pow(2), self.clip_max_value)

    def get_alpha_hat(self):
        return self._alpha_hats

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_betas_hat(self):
        return self._betas_hat