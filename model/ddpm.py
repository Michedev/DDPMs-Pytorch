from math import sqrt
from typing import Callable, Tuple, Optional, Type, Union, List, ClassVar

import pytorch_lightning as pl
import torch
import torchvision.utils
from torch import nn

from model.distributions import sigma_x_t, mu_x_t, mu_hat_xt_x0, sigma_hat_xt_x0, x0_to_xt
from variance_scheduler.abs_var_scheduler import Scheduler

class GaussianDDPM(pl.LightningModule):
    """
    Gaussian De-noising Diffusion Probabilistic Model
    This class implements both original DDPM model (by setting vlb=False) and Improved DDPM paper
    """

    def __init__(self, denoiser_module: nn.Module, opt: Union[Type[torch.optim.Optimizer], Callable[[], torch.optim.Optimizer], "partial[torch.optim.optimzer]"], T: int, variance_scheduler: Scheduler, lambda_variational: float, width: int, height: int, input_channels: int, logging_freq: int, vlb: bool, init_step_vlb: int):
        """
        :param denoiser_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, t)
        :param T: the amount of noising steps
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        :param lambda_variational: the coefficient in from of variational loss
        :param width: image width
        :param height: image height
        :param input_channels: image input channels
        :param logging_freq: frequency of logging loss function during training
        :param vlb: true to include the variational lower bound into the loss function
        :param init_step_vlb: the step at which the variational lower bound is included into the loss function
        """
        super().__init__()
        self.input_channels = input_channels
        self.denoiser_module = denoiser_module
        self.T = T
        self.opt_class = opt

        self.var_scheduler = variance_scheduler
        self.lambda_variational = lambda_variational
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)
        self.mse = nn.MSELoss()
        self.width = width
        self.height = height
        self.logging_freq = logging_freq
        self.vlb = vlb
        self.init_step_vlb = init_step_vlb
        self.iteration = 0
        self.init_step_vlb = max(1, self.init_step_vlb) # make sure that init_step_vlb is at least 1

    def forward(self, x: torch.FloatTensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DDPM model.

        Args:
            x: Input image tensor.
            t: Time step tensor.

        Returns:
            Tuple of predicted noise tensor and predicted variance tensor.
        """
        return self.denoiser_module(x, t)


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Training step of the DDPM model.

        Args:
            batch: Tuple of input image tensor and target tensor.
            batch_idx: Batch index.

        Returns:
            Dictionary containing the loss.
        """
        X, y = batch
        with torch.no_grad():
            # Map image values from [0, 1] to [-1, 1]
            X = X * 2 - 1
        # Sample a random time step t from 0 to T-1 for each image in the batch
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)  # todo add importance sampling
        # Compute alpha_hat for the selected time steps
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        # Sample noise eps from a normal distribution with the same shape as X
        eps = torch.randn_like(X)
        # Compute the intermediate image x_t from the original image X, alpha_hat, and eps
        x_t = x0_to_xt(X, alpha_hat, eps)  # go from x_0 to x_t with the formula
        # Run the intermediate image x_t through the model to obtain predicted noise and scale vectors (pred_eps, v)
        pred_eps, v = self(x_t, t)
        # Compute the loss for the predicted noise
        loss = 0.0
        noise_loss = self.mse(eps, pred_eps)
        loss = loss + noise_loss
        # If using the VLB loss, compute the VLB loss and add it to the total loss
        use_vlb = self.iteration >= self.init_step_vlb and self.vlb
        if use_vlb:
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(dim=0).sum()
            loss = loss + loss_vlb
        # If it's time to log the loss, log the total loss and optionally the noise and VLB losses
        if (self.iteration % self.logging_freq) == 0:
            self.log('loss/train_loss', loss, on_step=True)
            if use_vlb:
                self.log('loss/train_loss_noise', noise_loss, on_step=True)
                self.log('loss/train_loss_vlb', loss_vlb, on_step=True)
            # Log the norm of the gradients of the parameters
            norm_params = sum([torch.norm(p.grad) for p in self.parameters() if hasattr(p, 'grad') and p.grad is not None])
            self.log('grad_norm', norm_params)
        # Increment the iteration count
        self.iteration += 1
        # Return the loss as a dictionary
        return dict(loss=loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # Generate images and log them for visualization
        if batch_idx == 0:
            gen_images = self.generate(batch_size=16)  # Generate 16 images
            gen_images = torchvision.utils.make_grid(gen_images)  # Convert to grid
            self.logger.experiment.add_image('gen_val_images', gen_images, self.current_epoch)  # Log the images
            torchvision.utils.save_image(gen_images, f'gen_images/epoch={self.current_epoch}.png')  # Save the images

        # Unpack the batch into inputs X and ground truth y
        X, y = batch

        # Normalize inputs to [-1, 1] range
        with torch.no_grad():
            X = X * 2 - 1

        # Sample a time step t uniformly from [0, T-1] for each sample in the batch
        # TODO: Replace uniform sampling with importance sampling
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)

        # Compute alpha_hat from the precomputed alphas_hat for the sampled t
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)

        # Sample a noise vector eps from the standard normal distribution with the same shape as X
        eps = torch.randn_like(X)

        # Compute x_t, the input to the model at time step t
        x_t = x0_to_xt(X, alpha_hat, eps)

        # Forward pass through the model to get predicted noise and v
        pred_eps, v = self(x_t, t)

        # Compute the reconstruction loss between the input and the predicted noise
        loss = self.mse(eps, pred_eps)

        # If using the variational lower bound (VLB), compute the VLB loss and add it to the reconstruction loss
        # self.iteration > 0 is to avoid computing the VLB loss before the first training step because gives NaNs
        if self.iteration >= self.init_step_vlb and self.vlb:
            eps_loss = loss
            self.log('loss/val_eps_loss', eps_loss, on_step=True)
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(dim=0).sum()
            self.log('loss/val_vlb_loss', loss_vlb, on_step=True)
            loss = loss + loss_vlb

        # Log the total validation loss
        self.log('loss/val_loss', loss, on_step=True)

        # Return the loss as a dictionary
        return dict(loss=loss)

    def variational_loss(self, x_t: torch.Tensor, x_0: torch.Tensor,
                        model_noise: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """
        Compute variational loss for time step t
        
        Parameters:
            - x_t (torch.Tensor): the image at step t obtained with closed form formula from x_0
            - x_0 (torch.Tensor): the input image
            - model_noise (torch.Tensor): the unet predicted noise
            - v (torch.Tensor): the unet predicted coefficients for the variance
            - t (torch.Tensor): the time step
        
        Returns:
            - vlb (torch.Tensor): the pixel-wise variational loss, with shape [batch_size, channels, width, height]
        """
        vlb = 0.0
        t_eq_0 = (t == 0).reshape(-1, 1, 1, 1)
        
        # Compute variational loss for t=0 (i.e., first time step)
        if torch.any(t_eq_0):
            p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas),
                                        sigma_x_t(v, t, self.betas_hat, self.betas))
            # Compute log probability of x_0 under the distribution p
            # and add it to the variational lower bound
            vlb += - p.log_prob(x_0) * t_eq_0.float()
            
        t_eq_last = (t == (self.T - 1)).reshape(-1, 1, 1, 1)
        
        # Compute variational loss for t=T-1 (i.e., last time step)
        if torch.any(t_eq_last):
            p = torch.distributions.Normal(0, 1)
            q = torch.distributions.Normal(sqrt(self.alphas_hat[t]) * x_0, (1 - self.alphas_hat[t]))
            # Compute KL divergence between distributions p and q
            # and add it to the variational lower bound
            vlb += torch.distributions.kl_divergence(q, p) * t_eq_last.float()
            
        # Compute variational loss for all other time steps
        mu_hat = mu_hat_xt_x0(x_t, x_0, t, self.alphas_hat, self.alphas, self.betas)
        sigma_hat = sigma_hat_xt_x0(t, self.betas_hat)
        q = torch.distributions.Normal(mu_hat, sigma_hat)  # q(x_{t-1} | x_t, x_0)
        mu = mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas).detach()
        sigma = sigma_x_t(v, t, self.betas_hat, self.betas)
        p = torch.distributions.Normal(mu, sigma)  # p(x_t | x_{t-1})
        # Compute KL divergence between distributions p and q
        # and add it to the variational lower bound
        vlb += torch.distributions.kl_divergence(q, p) * (~t_eq_last).float() * (~t_eq_0).float()
        
        return vlb

    def configure_optimizers(self):
        return self.opt_class(params=self.parameters())

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
        if get_intermediate_steps:
            steps = []
        X_noise = torch.randn(batch_size, self.input_channels,  # start with random noise sampled from N(0, 1)
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

    def on_fit_start(self) -> None:
        self.alphas_hat: torch.FloatTensor = self.alphas_hat.to(self.device)
        self.alphas: torch.FloatTensor = self.alphas.to(self.device)
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
