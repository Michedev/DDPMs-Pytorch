from math import exp

import torch
from numpy import arctan

from variance_scheduler.abs_var_scheduler import Scheduler


class HyperbolicSecant(Scheduler):

    def __init__(self, T: int, lambda_min: float, lambda_max: float):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        # pg 3 section 2 for the details about the following eqns
        self.b = arctan(exp(-lambda_max / 2))
        self.a = arctan(exp(-lambda_min/2)) - self.b
        self._beta = - 2 * torch.log(torch.tan(self.a * torch.linspace(0, 1, T, dtype=torch.float) + self.b))
        self._alpha = 1.0 - self._beta
        self._alpha_hat = torch.cumprod(self._alpha, dim=0)
        self._alpha_hat_t_minus_1 = torch.roll(self._alpha_hat, shifts=1, dims=0)
        self._alpha_hat_t_minus_1[0] = self._alpha_hat_t_minus_1[1]
        self._beta_hat = (1 - self._alpha_hat_t_minus_1) / (1 - self._alpha_hat) * self._beta

    def get_alpha_hat(self):
        return self._alpha_hat

    def get_alphas(self):
        return self._alpha

    def get_betas(self):
        return self._beta

    def get_betas_hat(self):
        return self._beta_hat
