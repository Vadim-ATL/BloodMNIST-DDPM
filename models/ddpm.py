import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM)
    Executes forward and reverse diffusion using UNet to predict the noise
    """
    def __init__(self, T=1000, beta_start = 0.0001, beta_end=0.02, device = 'cpu'):
        self.T = T
        self.device = device
        # Variance schedule - betas increase linearly
        self.betas = th.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas

        # Cum.prod of alphas over time - closed-form sampling 
        self.alpha_bars = th.cumprod(self.alphas, dim=0)
        self.to(device)
    
    def to(self, device):
        """Moves the scheduler's tensors to the specified device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def sample_noise(self, x0: th.Tensor) -> th.Tensor:

        """Generate the Gaussian noise the same shape as x0"""
        epsilon = th.randn_like(x0)

        return epsilon

    def forward_diffusion(self, x0: th.Tensor, t: th.Tensor, epsilon: th.Tensor) -> th.Tensor:
        """
        Apply the forward diffusion process:
        q(x_t|x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
        """

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        xT = th.sqrt(alpha_bar_t)*x0 + th.sqrt(1-alpha_bar_t)*epsilon
        return xT

    def reverse_step(self, x_t, t, eps_theta) -> th.Tensor:
        """
        One reverse denoising step:
        1. Predict x_0
        2. Compute the mean (mu_theta) of q(x_t-1 | x_t, x_0)

        """
        alpha_t = self.alphas[t]
        alpha_t_prev = self.alpha_bars[t-1] if t>0 else th.tensor(1.0, device=self.device)

        beta_t = self.betas[t]
        alpha_bar_t = self.alpha_bars[t]

        # Predict clean image x_0 from noisy x_t
        x0_pred= (x_t - th.sqrt(1-alpha_bar_t)*eps_theta)/th.sqrt(alpha_bar_t)

        # Compute mean of the reverse process for future x_t
        mu_theta = (th.sqrt(alpha_t_prev) * beta_t / (1 - alpha_bar_t)) * x0_pred \
                + (th.sqrt(alpha_t) * (1 - alpha_t_prev) / (1 - alpha_bar_t)) * x_t
        
        return x0_pred, mu_theta

    def sample(self, unet: nn.Module, shape):
        """
        Generate a sample from Guassian noise and applying the reverse diffusion process step-by-step
        """

        batch_size = shape[0]
        x_t = th.randn(shape, device = self.device)

        for t in reversed(range(self.T)):

            time_tensor = th.tensor([t] * batch_size, device=self.device)
            # Predict the noise with UNet, input to the Unet is x_t and timesteps
            eps_theta = unet(x_t, time_tensor)
            # Compute reverse mean
            _, mu_theta = self.reverse_step(x_t, t, eps_theta)

            # Compute variance for stochastic sampling
            if t > 0:
                sigma_t = th.sqrt(self.betas[t] * (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]))
                noise = th.randn_like(x_t)
            else:
                sigma_t = 0
                noise = 0       

            # Sample from N (mu_theta, sigma_t2)
            x_t = mu_theta + sigma_t * noise

        
        return x_t