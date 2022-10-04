from typing import Union, Optional, List

from ddpm_pytorch.model.ddpm import GaussianDDPM
import torch

from distributions import sigma_x_t


class GaussianDDIM(GaussianDDPM):


    def generate(self, T: int, batch_size: Optional[int] = None,
                 get_intermediate_steps: bool = False, eta: float = 0.0) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a batch of images via denoising diffusion probabilistic model
        :param batch_size: batch size of generated images. The default value is 1
        :param T: number of diffusion steps to generated images. The default value is the training diffusion steps
        :param get_intermediate_steps: return all the denoising steps instead of the final step output
        :return: The tensor [bs, c, w, h] of generated images or a list of tensors [bs, c, w, h] if get_intermediate_steps=True
        """
        assert eta >= 0
        batch_size = batch_size or 1
        if get_intermediate_steps:
            steps = []
        X_noise = torch.randn(batch_size, self.input_channels,  # start with random noise sampled from N(0, 1)
                              self.width, self.height, device=self.device)
        for t in range(T - 1, -1, -1):
            t = int(t * self.T / T) # scale t to the training diffusion steps
            if get_intermediate_steps:
                steps.append(X_noise)
            t = torch.LongTensor([t]).to(self.device)
            eps, v = self.denoiser_module(X_noise, t)  # predict via nn the noise
            # if variational lower bound is present on the loss function hence v (the logit of variance) is trained
            # else the variance is taked fixed as in the original DDPM paper
            X_noise = self.denoise_step(X_noise, eps, t, eta, T)
            # original paper, the variance is estimated via nn instead of be fixed, as in Improved DDPM paper
        X_noise = (X_noise + 1) / 2  # rescale from [-1, 1] to [0, 1]
        if get_intermediate_steps:
            steps.append(X_noise)
            return steps
        return X_noise

    def denoise_step(self, X_noise, eps, t, eta, T):
        """
        Implementation of DDIM denoising step formula
        """
        t = t.reshape(-1, 1, 1, 1)
        t_min_1 = t - self.T / T # go to previous diffusion scaled difffuison step
        alpha_hat_t = self.alphas_hat[t]
        alpha_hat_t_min_1 = self.alphas_hat[t_min_1]
        sigma_t = 0 if eta == 0.0 else eta * torch.sqrt((1 - alpha_hat_t_min_1) / (1 - alpha_hat_t)) * torch.sqrt(1 - alpha_hat_t / alpha_hat_t_min_1)
        X_noise = torch.sqrt(alpha_hat_t_min_1) * \
                  (X_noise - (((1 - alpha_hat_t).sqrt() * eps) / alpha_hat_t.sqrt())) + \
                  (1 - alpha_hat_t_min_1 - sigma_t**2).sqrt() * eps
        if sigma_t != 0:
            z = torch.randn_like(X_noise)
            X_noise += sigma_t * z
        return X_noise
