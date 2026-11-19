import copy
import math

import numpy as np
import pytest
import torch
from scipy import integrate, optimize, stats

from cfvi import Gamma, Gaussian, Latent, Poisson, Regression

F64 = dict(dtype=torch.float64)
N = 10**6
G = torch.Generator().manual_seed(0)
LATENT = Latent(torch.randn(3, 8, generator=G, **F64) - 0.3, 0.2 + torch.rand(3, 8, generator=G, **F64),
                0.5 * torch.randn(8, generator=G, **F64), torch.tensor(0.2, **F64))  # h = 8, batch shape (3,)
X = torch.randn(3, 2, generator=G, **F64)


def built(likelihood):
    likelihood.build(2)
    return likelihood.double()


GAUSSIAN, HETERO, POISSON, GAMMA = built(Gaussian()), built(Gaussian(True)), built(Poisson()), built(Gamma())
with torch.no_grad():
    GAMMA.log_k.fill_(math.log(2.0))  # k = 2, so that a shape-rate mix-up shows
LIKELIHOODS = (GAUSSIAN, HETERO, POISSON, GAMMA)


def oracle(lik, eta, y):
    """scipy's log p(y | eta), for eta of shape (draws, 3) and y of shape (3,)."""
    if isinstance(lik, Gaussian):
        return stats.norm.logpdf(y, eta, lik.sd(X).detach().numpy())
    if isinstance(lik, Poisson):
        return stats.poisson.logpmf(y, np.exp(eta))
    return stats.gamma.logpdf(y, 2.0, scale=np.exp(eta) / 2)  # shape k = 2, mean e^eta


def quantiles(y, q):
    """The q-quantiles of the draws y (draws, 3), as values that y takes: shape (len(q), 3)."""
    return y.quantile(torch.tensor(q, dtype=y.dtype), dim=0, interpolation="lower")


@pytest.fixture(scope="module")
def eta():
    torch.manual_seed(1)
    return LATENT.sample((N,)).numpy()


@pytest.fixture(scope="module")
def draws():
    """Each likelihood, its predictive and N draws of y from the predictive."""
    torch.manual_seed(2)
    predictives = [lik.predictive(LATENT, X) for lik in LIKELIHOODS]
    return [(lik, pred, pred.sample((N,))) for lik, pred in zip(LIKELIHOODS, predictives)]


def assert_matches_rao_blackwell(lik, pred, eta, y):
    """|log_prob(y) - log mean_draws p(y | eta)| <= 4 SE, for draws eta (N, B) and y (B,)."""
    p = np.exp(oracle(lik, eta, y.numpy()))  # p(y | eta) at each draw of eta
    se = p.std(0) / math.sqrt(N) / p.mean(0)  # of log mean p, by the delta method
    err = np.abs(pred.log_prob(y).detach().numpy() - np.log(p.mean(0)))
    assert (err <= 4 * se).all(), (type(lik).__name__, err, se)


def test_bulk_vs_rao_blackwell_mc(eta, draws):
    for lik, pred, y in draws:
        for yq in quantiles(y, (0.05, 0.5, 0.95)):
            assert_matches_rao_blackwell(lik, pred, eta, yq)


def test_folded_unit():
    """Unit 2 is always on and narrower than a grid step, so it folds into the point mass at loc = 0.2 + 0.5 * 3."""
    latent = Latent(torch.tensor([[0.3, 3.0]], **F64), torch.tensor([[0.8, 1e-4]], **F64),
                    torch.tensor([1.0, 0.5], **F64), torch.tensor(0.2, **F64))
    assert torch.allclose(latent.grid().loc, torch.tensor(1.7, **F64))
    torch.manual_seed(3)
    eta, pred = latent.sample((N,)).numpy(), GAUSSIAN.predictive(latent, X[:1])
    for y in (0.5, 1.7, 3.0):
        assert_matches_rao_blackwell(GAUSSIAN, pred, eta, torch.tensor([y], **F64))


def test_cdf_vs_mc(draws):
    for lik, pred, y in draws:
        for yq in quantiles(y, (0.01, 0.5, 0.99)):
            fraction = (y <= yq).double().mean(0)
            se = (fraction * (1 - fraction) / N).sqrt()
            err = (pred.cdf(yq).detach() - fraction).abs()
            assert (err <= 4 * se).all(), (type(lik).__name__, err, se)


def test_sf_vs_quad():
    """One unit, (m, s, V, c) = (0.3, 0.8, 1, 0.2): sf where its true value is about 1e-3, 1e-6 and 1e-9, to 1%, against
    quad over the exact law: the atom Phi(-m/s) at c, then the density N(eta - c; m, s^2) above c."""
    m, s, c = 0.3, 0.8, 0.2
    latent = Latent(torch.tensor([[m]], **F64), torch.tensor([[s]], **F64), torch.ones(1, **F64),
                    torch.tensor(c, **F64))
    sd = GAUSSIAN.sd(X).item()
    for lik, sf in ((GAUSSIAN, lambda y, e: stats.norm.sf(y, e, sd)),
                    (POISSON, lambda y, e: stats.poisson.sf(y, np.exp(e))),
                    (GAMMA, lambda y, e: stats.gamma.sf(y, 2.0, scale=np.exp(e) / 2))):  # the sf of y | eta
        exact = lambda y: stats.norm.cdf(-m / s) * sf(y, c) + integrate.quad(
            lambda e: stats.norm.pdf(e - c, m, s) * sf(y, e), c, c + m + 12 * s, epsabs=0, epsrel=1e-10, limit=500)[0]
        pred = lik.predictive(latent, X[:1])
        for target in (1e-3, 1e-6, 1e-9):
            y = optimize.brentq(lambda y: exact(y) / target - 1, 0, 1e4, rtol=1e-3)  # Poisson sf is a step in y
            got = pred.sf(torch.tensor([y], **F64)).item()
            assert abs(got / exact(y) - 1) <= 0.01, (type(lik).__name__, target, got, exact(y))


def test_gamma_cdf_backward():
    """torch's Gamma cdf has no gradient in the shape k, so the predictive's cdf holds k constant (gradient 0)."""
    gamma = copy.deepcopy(GAMMA)
    gamma.predictive(LATENT, X).cdf(torch.ones(3, **F64)).sum().backward()
    assert gamma.log_k.grad.abs() <= 1e-12


def test_cdf_monotone_and_limits(draws):
    for lik, pred, y in draws:
        lo = -1.0 if isinstance(lik, (Poisson, Gamma)) else y.min().item() - 10
        grid = torch.linspace(lo, 2 * y.max().item() + 10, 101, **F64)[:, None]  # (101, 1) against batch (3,)
        F = pred.cdf(grid).detach()
        assert (F.diff(dim=0) >= 0).all(), type(lik).__name__
        assert F[0].abs().max() <= 1e-6 and (F[-1] - 1).abs().max() <= 1e-6, (type(lik).__name__, F[0], F[-1])
        assert torch.equal(pred.sf(grid).detach(), 1 - F)
        assert (pred.sf(grid) >= 0).all() and (pred.latent.sf(grid) >= 0).all(), type(lik).__name__  # far tails too


def test_float32_finite():
    torch.manual_seed(0)
    x = torch.randn(3, 2)
    for lik in LIKELIHOODS:
        head = Regression(2, 8, copy.deepcopy(lik), num_data=10)
        out32, out64 = head.float()(x), copy.deepcopy(head).double()(x.double())
        y = quantiles(out64.predictive.sample((10**5,)), (0.001, 0.5, 0.999))
        for name in ("log_prob", "cdf", "sf"):
            v32, v64 = getattr(out32.predictive, name)(y.float()), getattr(out64.predictive, name)(y)
            assert v32.dtype == torch.float32 and v32.isfinite().all(), (type(lik).__name__, name, v32)
            assert (v32.double() - v64).abs().max() <= 1e-3, (type(lik).__name__, name, v32, v64)


def test_out_of_support_is_neg_inf():
    gamma = copy.deepcopy(GAMMA)
    with torch.no_grad():
        gamma.log_k.fill_(math.log(0.5))  # k < 1: p(0 | eta) = +inf, so unmasked, log_prob(0) is NaN, not -inf
    for lik, y in ((gamma, [-1.0, 0.0, 2.0]), (POISSON, [-0.5, 1.5, 2.0])):
        log_p = lik.predictive(LATENT, X).log_prob(torch.tensor(y, **F64))
        assert (log_p[:2] == -math.inf).all() and log_p[2].isfinite(), (type(lik).__name__, log_p)


def test_mean_variance_vs_mc(draws):
    for lik, pred, y in draws:
        mean, var = y.mean(0), y.var(0)
        se_var = (((y - mean) ** 4).mean(0) - var**2).sqrt() / math.sqrt(N)
        assert ((pred.mean - mean).abs() <= 4 * (var / N).sqrt()).all(), (type(lik).__name__, pred.mean, mean)
        assert ((pred.variance - var).abs() <= 4 * se_var).all(), (type(lik).__name__, pred.variance, var)
