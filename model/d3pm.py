from typing import Callable, Iterator, Tuple, Optional, Type, Union, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tensorguard as tg
from torch import nn
from torch.nn.parameter import Parameter

from variance_scheduler.abs_var_scheduler import Scheduler

class D3PM(pl.LightningModule):
    """
    Discrete Denoising Diffusion Probabilistic Models (D3PM)
    Implementation using Uniform Transition Matrix for discrete state spaces (e.g. images)
    """

    def __init__(self, denoiser_module: nn.Module, opt: Union[Type[torch.optim.Optimizer], Callable[[Iterator[Parameter]], torch.optim.Optimizer]], 
                 T: int, variance_scheduler: Scheduler, lambda_variational: float, width: int, height: int, input_channels: int, 
                 logging_freq: int, vlb: bool, init_step_vlb: int, num_classes: int = 256, assert_shapes: bool = True):
        """
        :param denoiser_module: The nn which computes logits of p(x_0 | x_t).
        :param T: the amount of noising steps
        :param variance_scheduler: scheduler providing alphas/betas
        :param num_classes: number of discrete states (e.g. 256 for 8-bit images)
        """
        super().__init__()
        self.input_channels = input_channels
        self.denoiser_module = denoiser_module
        self.T = T
        self.opt_class = opt

        self.var_scheduler = variance_scheduler
        self.lambda_variational = lambda_variational
        
        # In D3PM with uniform transition, alpha_bar represents prob of staying in x_0
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        
        self.width = width
        self.height = height
        self.logging_freq = logging_freq
        self.vlb = vlb
        self.init_step_vlb = max(1, init_step_vlb)
        self.iteration = 0
        self.num_classes = num_classes
        self.assert_shapes = assert_shapes

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of indices (B, C, H, W)
            t: Time step tensor (B,)
        Returns:
            Logits for x_0 (B, C, K, H, W)
        """
        # Convert indices to one-hot float input for denoiser
        # x: (B, C, H, W) -> (B, C, K, H, W) -> (B, C*K, H, W)
        x_onehot = F.one_hot(x, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        B, K, C, H, W = x_onehot.shape
        # Input to model: -1 to 1 range is standard for neural nets, but one-hot is 0/1. 
        # Usually 0/1 is fine, or x*2-1. Let's keep 0-1 onehot.
        model_input = x_onehot.reshape(B, C * K, H, W)
        
        # Denoiser acts on features, returns logits of shape (B, C*K, H, W)
        model_output = self.denoiser_module(model_input, t)
        
        output = model_output.reshape(B, C, K, H, W)
        if self.assert_shapes: tg.guard(output, "B, C, K, H, W")
        return output

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) using Uniform Transition Matrix
        """
        # alpha_bar_t is probability of staying at x_0 (plus uniform component)
        # Total prob of x_t == x_0 is alpha_bar + (1-alpha_bar)/K
        # We sample: with prob alpha_bar, keep x_0. Else uniform.
        
        alpha_bar_t = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        
        # Mask: 1 to keep x_0, 0 to replace
        # We need to sample mask per pixel? Yes.
        u = torch.rand_like(x_0.float())
        mask = (u < alpha_bar_t).long()
        
        uniform_noise = torch.randint(0, self.num_classes, x_0.shape, device=x_0.device)
        x_t = mask * x_0 + (1 - mask) * uniform_noise
        return x_t

    def q_posterior_logits(self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute logits of q(x_{t-1} | x_t, x_0)
        """
        if self.assert_shapes: tg.guard(x_t, "B, C, H, W")
        if self.assert_shapes: tg.guard(x_0, "B, C, H, W")
        if self.assert_shapes: tg.guard(t, "B")

        K = self.num_classes
        t_min_1 = torch.clamp(t - 1, min=0)
        
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        alpha_bar_tm1 = self.alphas_hat[t_min_1].view(-1, 1, 1, 1, 1)
        
        # p(x_{t-1} | x_0) part
        p1_eq = alpha_bar_tm1 + (1 - alpha_bar_tm1)/K
        p1_neq = (1 - alpha_bar_tm1)/K
        
        # p(x_t | x_{t-1}) part
        p2_eq = alpha_t + (1 - alpha_t)/K
        p2_neq = (1 - alpha_t)/K
        
        # Logits computation
        eps = 1e-30
        log_p1_eq = torch.log(p1_eq + eps)
        log_p1_neq = torch.log(p1_neq + eps)
        log_p2_eq = torch.log(p2_eq + eps)
        log_p2_neq = torch.log(p2_neq + eps)
        
        # Base logit
        logits = log_p1_neq + log_p2_neq
        
        # Expand to match x_0 spatial dimensions and channels
        # Target shape: (B, C, K, H, W)
        B, C, H, W = x_0.shape
        logits = logits.expand(B, C, K, H, W).clone()
        
        x_0_expanded = x_0.unsqueeze(2)
        x_t_expanded = x_t.unsqueeze(2)
        
        if self.assert_shapes: 
            tg.guard(logits, "B, C, K, H, W")
            tg.guard(x_0_expanded, "B, C, 1, H, W")
            tg.guard(x_t_expanded, "B, C, 1, H, W")
        
        # Add delta 1 at x_0
        logits.scatter_add_(2, x_0_expanded, (log_p1_eq - log_p1_neq).expand_as(x_0_expanded))
        # Add delta 2 at x_t
        logits.scatter_add_(2, x_t_expanded, (log_p2_eq - log_p2_neq).expand_as(x_t_expanded))
        
        if self.assert_shapes: tg.guard(logits, "B, C, K, H, W")
        return logits

    def compute_loss(self, x_0, x_t, t, logits_x0):
        if self.assert_shapes: tg.guard(x_0, "B, C, H, W")
        if self.assert_shapes: tg.guard(x_t, "B, C, H, W")
        if self.assert_shapes: tg.guard(t, "B")
        if self.assert_shapes: tg.guard(logits_x0, "B, C, K, H, W")

        # 1. Prediction Loss (Hybrid: Cross Entropy)
        # Often used as L_simple or auxiliary
        ce_loss = F.cross_entropy(logits_x0.transpose(1, 2), x_0, reduction='none')
        ce_mean = ce_loss.mean()
        
        if not self.vlb or (self.vlb and self.global_step <= self.init_step_vlb):
            return dict(loss=ce_mean, noise_loss=ce_mean, vlb_loss=None)
        
        # 2. VLB (KL Divergence)
        # q_true = q(x_{t-1} | x_t, x_0)
        true_logits = self.q_posterior_logits(x_t, x_0, t)
        
        # p_pred = p(x_{t-1} | x_t) computed from predicted x_0
        probs_x0 = F.softmax(logits_x0, dim=2)
        K = self.num_classes
        t_min_1 = torch.clamp(t - 1, min=0)
        alpha_bar_tm1 = self.alphas_hat[t_min_1].view(-1, 1, 1, 1, 1)
        
        # Expected value of q(x_{t-1} | x_0) over predicted x_0
        pred_x_tm1_part1 = alpha_bar_tm1 * probs_x0 + (1 - alpha_bar_tm1) / K
        
        # Multiply by q(x_t | x_{t-1})
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        p_neq = (1 - alpha_t) / K
        p_eq = alpha_t + p_neq
        
        fact = torch.ones_like(probs_x0) * p_neq
        fact.scatter_add_(2, x_t.unsqueeze(2), (p_eq - p_neq).expand_as(x_t.unsqueeze(2)))
        
        pred_probs_unnorm = pred_x_tm1_part1 * fact
        pred_logits = torch.log(pred_probs_unnorm + 1e-30)
        
        # KL
        q_probs = F.softmax(true_logits, dim=2)
        q_log_probs = F.log_softmax(true_logits, dim=2)
        p_log_probs = F.log_softmax(pred_logits, dim=2)
        
        kl = (q_probs * (q_log_probs - p_log_probs)).sum(dim=2)
        
        # For t=0, use reconstruction loss (NLL)
        # VLB is usually sum of KLs + L0.
        loss_t = torch.where(t[:, None, None, None] == 0, ce_loss, kl)
        vlb_mean = loss_t.mean()
        
        # Hybrid loss: L_simple + lambda * L_vlb
        loss = ce_mean + self.lambda_variational * vlb_mean

        return dict(loss=loss, noise_loss=ce_mean, vlb_loss=vlb_mean)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if tg.has_dim('B'): tg.del_dim('B')
        X, y = batch
        # Assume inputs are float [0, 1] images, convert to indices
        with torch.no_grad():
            X = (X * (self.num_classes - 1)).round().long().clamp(0, self.num_classes - 1)
            
        t = torch.randint(0, self.T, (X.shape[0],), device=X.device)
        x_t = self.q_sample(X, t)
        
        logits_x0 = self(x_t, t)
        loss_dict = self.compute_loss(X, x_t, t, logits_x0)
        
        self.iteration += 1
        return loss_dict

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if tg.has_dim('B'): tg.del_dim('B')
        X, y = batch
        with torch.no_grad():
            X = (X * (self.num_classes - 1)).round().long().clamp(0, self.num_classes - 1)
            
        t = torch.randint(0, self.T, (X.shape[0],), device=X.device)
        x_t = self.q_sample(X, t)
        logits_x0 = self(x_t, t)
        loss_dict = self.compute_loss(X, x_t, t, logits_x0)
        return loss_dict

    def configure_optimizers(self):
        return self.opt_class(params=self.parameters())

    def generate(self, batch_size: Optional[int] = None, T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        tg.set_dim('B', batch_size)
        batch_size = batch_size or 1
        T = T or self.T
        
        # Start with uniform noise
        x = torch.randint(0, self.num_classes, (batch_size, self.input_channels, self.height, self.width), device=self.device)
        
        steps = []
        if get_intermediate_steps: steps.append(x.float() / (self.num_classes - 1))
            
        for t in range(T - 1, -1, -1):
            timestamp = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            logits_x0 = self(x, timestamp)
            probs_x0 = F.softmax(logits_x0, dim=2)
            
            # Posterior calculation for p(x_{t-1} | x_t)
            # For t=0, just return x_0 estimate
            if t == 0:
                # Sample x_0
                probs_perm = probs_x0.permute(0, 3, 4, 1, 2)
                x = torch.distributions.Categorical(probs_perm).sample().permute(0, 3, 1, 2)
            else:
                alpha_bar_tm1 = self.alphas_hat[t-1]
                pred_x_tm1_part1 = alpha_bar_tm1 * probs_x0 + (1 - alpha_bar_tm1) / self.num_classes
                
                alpha_t = self.alphas[t]
                p_neq = (1 - alpha_t) / self.num_classes
                p_eq = alpha_t + p_neq
                
                fact = torch.ones_like(probs_x0) * p_neq
                fact.scatter_add_(2, x.unsqueeze(2), (p_eq - p_neq).expand_as(x.unsqueeze(2)))
                
                pred_probs = pred_x_tm1_part1 * fact
                pred_probs = pred_probs / (pred_probs.sum(dim=2, keepdim=True) + 1e-30)
                
                probs_perm = pred_probs.permute(0, 3, 4, 1, 2)
                x = torch.distributions.Categorical(probs_perm).sample().permute(0, 3, 1, 2)
            
            if get_intermediate_steps:
                # Normalize back to [0,1] for visualization
                steps.append(x.float() / (self.num_classes - 1))
                
        if get_intermediate_steps:
            return steps
        return x.float() / (self.num_classes - 1)

    def on_fit_start(self) -> None:
        self.alphas_hat = self.alphas_hat.to(self.device)
        self.alphas = self.alphas.to(self.device)
