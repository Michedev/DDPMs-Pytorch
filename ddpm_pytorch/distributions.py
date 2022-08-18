from math import sqrt, log

import torch


def mu_x_t(x_t: torch.Tensor, t: int, model_noise: torch.Tensor, alphas_hat: torch.Tensor, betas: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """

    :param x_t: the noised image
    :param t: the time step of $x_t$
    :param model_noise: the model estimated noise
    :param alphas_hat: sequence of $\hat{\alpha}$ used for variance scheduling
    :param betas: sequence of $\beta$ used for variance scheduling
    :param alphas: sequence of $\alpha$ used for variance scheduling
    :return: the mean of $q(x_t | x_0)$
    """
    x = 1 / sqrt(alphas[t]) * (x_t - betas[t] / sqrt(1 - alphas_hat[t]) * model_noise)
    # tg.guard(x, "B, C, W, H")
    return x


def sigma_x_t(v: torch.Tensor, t: int, betas_hat: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """
    Compute the varaince at time step t as defined in "Improving Denoising Diffusion probabilistic Models", eqn 15 page 4
    :param v: the neural network "logits" used to compute the variance
    :param t: the target time step
    :param betas_hat: sequence of $\hat{\beta}$ used for variance scheduling
    :param betas: sequence of $\beta$ used for variance scheduling
    :return: the estimated variance at time step t
    """
    x = torch.exp(v * log(betas[t]) + (1 - v) * log(betas_hat[t]))
    # tg.guard(x, "B, C, W, H")
    return x


def mu_hat_xt_x0(x_t: torch.Tensor, x_0: torch.Tensor, t: int, alphas_hat: torch.Tensor, alphas: torch.Tensor,
                 betas: torch.Tensor):
    """
    Compute $\hat{mu}(x_t, x_0)$ from "Improving Denoising Diffusion probabilistic Models", eqn 11 page 2
    :param x_t: The noised image at step t
    :param x_0: the original image
    :param t: the time step of $x_t$
    :param alphas_hat: sequence of $\hat{\alpha}$ used for variance scheduling
    :param alphas: sequence of $\alpha$ used for variance scheduling
    :param betas: sequence of $\beta$ used for variance scheduling
    :return: the mean of distribution $q(x_{t-1} | x_t, x_0)$
    """
    x = sqrt(alphas_hat[t - 1]) * betas[t] / (1 - alphas_hat[t]) * x_0 + \
        sqrt(alphas[t]) * (1 - alphas_hat[t - 1]) / (1 - alphas_hat[t]) * x_t
    # tg.guard(x, "B, C, W, H")
    return x


def sigma_hat(t: int, betas_hat: torch.Tensor) -> float:
    return betas_hat[t]