"""The predictive law of y: a mixture of the law of y | eta over the Latent's eta grid."""
import math

import torch
from torch.distributions import Distribution, constraints


class Predictive(Distribution):
    """The law of y when eta has the Latent law and y | eta ~ conditional(eta), a torch Distribution built with
    validate_args=False for eta with a trailing grid axis. Over latent.grid(), p(y) = exp(log_mass) p(y | loc) +
    sum_i density_i dx p(y | eta_i), with these weights normalized to sum to 1: log_prob by log-sum-exp, cdf the same
    weighted sum of the conditional's cdf, sf = 1 - cdf. log_prob is -inf for y outside support. mean and variance are
    the likelihood's closed forms, passed in.

    Accuracy is the eta grid's: see the floors in Latent's docstring. sf is unreliable below about 1e-12 in float64 and
    1e-5 in float32. A Gaussian conditional also needs its sd well above the grid step dx; a narrower normal sees the
    grid as a comb of points, not as a smooth density."""

    arg_constraints = {}
    support = constraints.real
    mean = property(lambda self: self._mean)
    variance = property(lambda self: self._variance)

    def __init__(self, latent, conditional, support, mean, variance):
        eta, density, loc, log_mass, dx = latent.grid()
        self.latent, self.conditional, self.support = latent, conditional, support
        self._mean, self._variance = mean, variance
        w = torch.cat([log_mass.exp().unsqueeze(-1), density * dx.unsqueeze(-1)], -1)
        self._w = w / w.sum(-1, keepdim=True)  # the density's clamp at 0 adds mass, which would floor sf = 1 - cdf
        safe = torch.where(self._w > 0, self._w, 1.0)  # log'(0) would give a NaN gradient
        self._log_w = torch.where(self._w > 0, safe.log(), -math.inf)
        self._given = conditional(torch.cat([loc.unsqueeze(-1), eta], -1))  # the point mass is one more grid point
        super().__init__(latent.batch_shape, validate_args=False)

    def log_prob(self, y):
        log_p = (self._log_w + self._given.log_prob(y.unsqueeze(-1))).logsumexp(-1)
        return torch.where(self.support.check(y), log_p, -math.inf)

    def cdf(self, y):
        return (self._w * self._given.cdf(y.unsqueeze(-1))).sum(-1).clamp(0, 1)  # round-off can pass 1

    def sf(self, y):
        return 1 - self.cdf(y)

    def sample(self, sample_shape=torch.Size()):
        return self.conditional(self.latent.sample(sample_shape).unsqueeze(-1)).sample().squeeze(-1)
