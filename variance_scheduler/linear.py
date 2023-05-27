from variance_scheduler.abs_var_scheduler import Scheduler
import torch


class LinearScheduler(Scheduler):
    """
    A scheduler that linearly interpolates between two values of beta over T steps.
    """

    def __init__(self, T: int, beta_start: float, beta_end: float):
        """
        Initializes the LinearScheduler.

        Args:
            T (int): The number of steps to interpolate over.
            beta_start (float): The starting value of beta.
            beta_end (float): The ending value of beta.
        """
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._beta = torch.linspace(beta_start, beta_end, T)
        self._alpha = 1.0 - self._beta
        self._alpha_hat = torch.cumprod(self._alpha, dim=0)
        self._alpha_hat_t_minus_1 = torch.roll(self._alpha_hat, shifts=1, dims=0)
        self._alpha_hat_t_minus_1[0] = self._alpha_hat_t_minus_1[1]
        self._beta_hat = (1 - self._alpha_hat_t_minus_1) / (1 - self._alpha_hat) * self._beta

    def get_alpha_hat(self):
        """
        Returns the cumulative product of (1 - beta) over time.
        """
        return self._alpha_hat

    def get_alphas(self):
        """
        Returns the values of alpha over time.
        """
        return self._alpha

    def get_betas(self):
        """
        Returns the values of beta over time.
        """
        return self._beta

    def get_betas_hat(self):
        """
        Returns the values of beta_hat over time.
        """
        return self._beta_hat