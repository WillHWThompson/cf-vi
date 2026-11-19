"""Likelihoods p(y | eta) whose expected log-likelihood under the Latent law of eta is closed form (for Categorical, a
lower bound): it needs only the mean, the variance and the log-MGF of eta."""
import math

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, constraints
from torch.nn import functional as F

from .latent import Latent
from .predictive import Predictive


class Likelihood(nn.Module):
    """p(y | eta) for y in support.
    - build(in_features): called once by the head.
    - expected_log_prob(latent, y, x): E log p(y | eta) over the latent, of shape (B,).
    - conditional(eta, x): the law of y | eta, for eta with a trailing grid axis.
    - moments(latent, x): the mean and variance of predictive(latent, x), the mixture of conditional over eta."""

    support = constraints.real

    def build(self, in_features: int) -> None:
        pass

    def predictive(self, latent: Latent, x: Tensor) -> Distribution:
        return Predictive(latent, lambda eta: self.conditional(eta, x), self.support, *self.moments(latent, x))


class Gaussian(Likelihood):
    """y | eta ~ N(eta, sd^2), with sd = softplus(raw) + 1e-6 and raw one learned scalar, or a linear map of x if
    heteroscedastic. E log p = -log(2 pi sd^2) / 2 - ((y - E eta)^2 + Var eta) / (2 sd^2)."""

    def __init__(self, heteroscedastic: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic

    def build(self, in_features: int) -> None:
        self.raw_sd = nn.Linear(in_features, 1) if self.heteroscedastic else nn.Parameter(torch.zeros(()))

    def sd(self, x: Tensor) -> Tensor:
        raw = self.raw_sd(x).squeeze(-1) if self.heteroscedastic else self.raw_sd
        return F.softplus(raw) + 1e-6

    def expected_log_prob(self, latent: Latent, y: Tensor, x: Tensor) -> Tensor:
        var = self.sd(x) ** 2
        return -torch.log(2 * math.pi * var) / 2 - ((y - latent.mean) ** 2 + latent.variance) / (2 * var)

    def conditional(self, eta: Tensor, x: Tensor) -> Distribution:
        return torch.distributions.Normal(eta, self.sd(x).unsqueeze(-1), validate_args=False)  # sd (B,) -> (B, 1)

    def moments(self, latent: Latent, x: Tensor) -> tuple[Tensor, Tensor]:
        return latent.mean, latent.variance + self.sd(x) ** 2


class Poisson(Likelihood):
    """y | eta ~ Poisson(e^eta). E log p = y E eta - E e^eta - lgamma(y + 1), with E e^eta = exp(log_mgf(1))."""

    support = constraints.nonnegative_integer

    def expected_log_prob(self, latent: Latent, y: Tensor, x: Tensor) -> Tensor:
        return y * latent.mean - latent.log_mgf(1.0).exp() - torch.lgamma(y + 1)

    def conditional(self, eta: Tensor, x: Tensor) -> Distribution:
        return _Poisson(eta.exp(), validate_args=False)

    def moments(self, latent: Latent, x: Tensor) -> tuple[Tensor, Tensor]:
        m1, m2 = latent.log_mgf(1.0).exp(), latent.log_mgf(2.0).exp()  # E e^eta, E e^(2 eta)
        return m1, m1 + m2 - m1**2


class _Poisson(torch.distributions.Poisson):
    """torch's Poisson with a cdf: P(Y <= y) = Q(floor(y) + 1, rate), Q the regularized upper incomplete gamma."""

    def cdf(self, y: Tensor) -> Tensor:
        return torch.where(y >= 0, torch.special.gammaincc(y.clamp_min(0).floor() + 1, self.rate), 0.0)


class Gamma(Likelihood):
    """y | eta ~ Gamma(shape k, rate k e^-eta), of mean e^eta, with k = exp(log_k) learned (init 1).
    E log p = k log k - lgamma(k) + (k - 1) log y - k E eta - k y E e^-eta, with E e^-eta = exp(log_mgf(-1)).
    The predictive's cdf holds k constant: torch's igamma has no gradient in its first argument."""

    support = constraints.positive

    def __init__(self):
        super().__init__()
        self.log_k = nn.Parameter(torch.zeros(()))

    def expected_log_prob(self, latent: Latent, y: Tensor, x: Tensor) -> Tensor:
        k = self.log_k.exp()
        return (k * self.log_k - torch.lgamma(k) + (k - 1) * y.log() - k * latent.mean
                - k * y * latent.log_mgf(-1.0).exp())

    def conditional(self, eta: Tensor, x: Tensor) -> Distribution:
        k = self.log_k.exp()
        return _Gamma(k, k * (-eta).exp(), validate_args=False)

    def moments(self, latent: Latent, x: Tensor) -> tuple[Tensor, Tensor]:
        m1, m2 = latent.log_mgf(1.0).exp(), latent.log_mgf(2.0).exp()  # Var(y | eta) = e^(2 eta) / k
        return m1, m2 * (1 + (-self.log_k).exp()) - m1**2


class _Gamma(torch.distributions.Gamma):
    """torch's Gamma, with cdf 0 for y < 0 (torch's is NaN there) and the shape k held constant in the cdf."""

    def cdf(self, y: Tensor) -> Tensor:
        k = self.concentration.detach()
        return torch.special.gammainc(k, k * self.rate / self.concentration * y.clamp_min(0))  # rate / k = e^-eta


class Categorical(Likelihood):
    """y | eta ~ softmax(eta) over K classes, with eta of batch shape (B, K) and y in 0..K-1. By Jensen, E log p(y) =
    E eta_y - E log sum_k e^eta_k >= E eta_y - log sum_k E e^eta_k = E eta_y - logsumexp_k log_mgf(1)_k.
    The predictive probabilities are proportional to E e^eta_k."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.support = constraints.integer_interval(0, num_classes - 1)

    def expected_log_prob(self, latent: Latent, y: Tensor, x: Tensor) -> Tensor:
        return latent.mean.gather(-1, y[:, None])[:, 0] - latent.log_mgf(1.0).logsumexp(-1)

    def predictive(self, latent: Latent, x: Tensor):
        return torch.distributions.Categorical(logits=latent.log_mgf(1.0))
