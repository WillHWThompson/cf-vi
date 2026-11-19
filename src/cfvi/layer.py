"""A Bayesian ReLU layer with a deterministic linear readout, whose output law is a Latent."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .latent import Latent


class BayesReLU(nn.Module):
    """eta = V ReLU(W x + b) + c, with mean-field Gaussian q(W), q(b) and the prior N(0, prior_scale^2) on every
    entry of W and b (prior_scale is a standard deviation, not rescaled by fan-in). V and c are deterministic.

    forward(x) with x of shape (B, in_features) returns the Latent law of eta given x, with batch shape
    (B, out_features). Its m and s, of shape (B, hidden), are the means W_mean x + b_mean and the sds
    sqrt(W_sd^2 x^2 + b_sd^2) of z = W x + b; Latent(m, s, V[k], c[k]) is the law of output k alone, batch shape (B,).

    Init ("narrow", as in the research code): the means of W and b, and V and c, as nn.Linear; the posterior log-sds
    uniform on [-2.5, -0.5], so sds of 0.08 to 0.6."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, prior_scale: float = 1.0):
        super().__init__()
        self.prior_scale = prior_scale
        hidden = nn.Linear(in_features, hidden_features)
        self.w_mean, self.b_mean = hidden.weight, hidden.bias
        self.w_log_sd = nn.Parameter(torch.empty_like(hidden.weight).uniform_(-2.5, -0.5))
        self.b_log_sd = nn.Parameter(torch.empty_like(hidden.bias).uniform_(-2.5, -0.5))
        self.readout = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> Latent:
        m = F.linear(x, self.w_mean, self.b_mean)
        s = F.linear(x * x, (2 * self.w_log_sd).exp(), (2 * self.b_log_sd).exp()).sqrt()
        return Latent(m, s, self.readout.weight, self.readout.bias)

    def kl(self) -> torch.Tensor:
        """KL(q || prior) summed over W and b, in closed form."""
        p = self.prior_scale
        return sum((math.log(p) - log_sd + ((2 * log_sd).exp() + mean**2) / (2 * p**2) - 0.5).sum()
                   for mean, log_sd in ((self.w_mean, self.w_log_sd), (self.b_mean, self.b_log_sd)))
