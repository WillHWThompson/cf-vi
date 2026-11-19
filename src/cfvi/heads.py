"""The Regression and Classification heads and their Output, in the style of vbll."""
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import nn

from .latent import Latent
from .layer import BayesReLU
from .likelihoods import Categorical


class Regression(nn.Module):
    """A BayesReLU layer with one output eta and a likelihood p(y | eta). forward(x), x of shape (B, in_features),
    returns an Output whose latent has batch shape (B,); num_data, the training set size, scales the KL."""

    def __init__(self, in_features: int, hidden_features: int, likelihood: nn.Module, num_data: int,
                 prior_scale: float = 1.0):
        super().__init__()
        self.layer = BayesReLU(in_features, hidden_features, 1, prior_scale)
        self.likelihood = likelihood
        likelihood.build(in_features)
        self.num_data = num_data

    def forward(self, x: torch.Tensor) -> "Output":
        law = self.layer(x)
        latent = Latent(law.m, law.s, law.V[0], law.c[0])
        return Output(latent, self.likelihood, x, self.layer.kl(), self.num_data)


class Classification(nn.Module):
    """A BayesReLU layer with num_classes outputs eta_k and the likelihood y | eta ~ softmax(eta). forward(x) returns an
    Output whose latent has batch shape (B, num_classes); y is a LongTensor of class indices of shape (B,)."""

    def __init__(self, in_features: int, hidden_features: int, num_classes: int, num_data: int,
                 prior_scale: float = 1.0):
        super().__init__()
        self.layer = BayesReLU(in_features, hidden_features, num_classes, prior_scale)
        self.likelihood = Categorical(num_classes)
        self.num_data = num_data

    def forward(self, x: torch.Tensor) -> "Output":
        return Output(self.layer(x), self.likelihood, x, self.layer.kl(), self.num_data)


@dataclass(frozen=True)
class Output:
    """A head's output for inputs x: latent is the law of eta given x, kl the layer's KL(q || prior).
    train_loss_fn(y) is the negative ELBO per data point (for Categorical, an upper bound on it),
    -(batch mean of E log p(y | eta)) + kl / num_data; val_loss_fn(y) is -(batch mean of predictive.log_prob(y)).
    y has shape (B,)."""

    latent: Latent
    likelihood: nn.Module
    x: torch.Tensor
    kl: torch.Tensor
    num_data: int

    def _check(self, y: torch.Tensor) -> None:
        if y.shape != self.x.shape[:1]:
            raise ValueError(f"y has shape {tuple(y.shape)}, expected ({len(self.x)},)")
        if not self.likelihood.support.check(y).all():
            raise ValueError(f"y is outside the support of the {type(self.likelihood).__name__} likelihood")

    def train_loss_fn(self, y: torch.Tensor) -> torch.Tensor:
        self._check(y)
        return -self.likelihood.expected_log_prob(self.latent, y, self.x).mean() + self.kl / self.num_data

    def val_loss_fn(self, y: torch.Tensor) -> torch.Tensor:
        self._check(y)
        return -self.predictive.log_prob(y).mean()

    @cached_property
    def predictive(self):
        """The law of y given x, built on first access."""
        return self.likelihood.predictive(self.latent, self.x)
