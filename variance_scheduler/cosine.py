from math import pi
import torch

from variance_scheduler.abs_var_scheduler import Scheduler

class CosineScheduler(Scheduler):


    clip_max_value = torch.Tensor([0.999])

    def __init__(self, T: int, s: float = 0.008):
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
        self._alpha_hats = self._alpha_hat_function(torch.arange(self.T), T, s)
        self._alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, shifts=1, dims=0) # shift forward by 1 so that alpha_first[t] = alpha[t-1]
        self._alpha_hats_t_minus_1[0] = self._alpha_hats_t_minus_1[1]  # to remove first NaN value
        self._betas = 1.0 - self._alpha_hats / self._alpha_hats_t_minus_1
        self._betas = torch.minimum(self._betas, self.clip_max_value)
        self._alphas = 1.0 - self._betas
        self._betas_hat = (1 - self._alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas
        self._betas_hat[torch.isnan(self._betas_hat)] = 0.0

    def _alpha_hat_function(self, t: torch.Tensor, T: int, s: float):
        """
        Compute the alpha_hat value for a given t value.
        :param t: the t value
        :param T: the total amount of noising steps
        :param s: smoothing parameter
        """
        cos_value = torch.pow(torch.cos((t / T + s) / (1 + s) * pi / 2.0), 2)
        return cos_value

    def get_alpha_hat(self):
        return self._alpha_hats

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_betas_hat(self):
        return self._betas_hat
    

if __name__ == '__main__':
    scheduler = CosineScheduler(1000)
    import matplotlib.pyplot as plt
    plt.plot(scheduler.get_alpha_hat().numpy())
    plt.ylabel('$\\alpha_t$')
    plt.xlabel('t')
    plt.show()