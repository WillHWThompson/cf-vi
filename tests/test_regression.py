import math

import numpy as np
import pytest
import torch
from scipy import stats

from cfvi import Gamma, Gaussian, Latent, Poisson, Regression

F64 = dict(dtype=torch.float64)
G = torch.Generator().manual_seed(0)
LATENT = Latent(torch.randn(3, 4, generator=G, **F64), 0.3 + 0.5 * torch.rand(3, 4, generator=G, **F64),
                0.7 * torch.randn(4, generator=G, **F64), torch.tensor(0.2, **F64))  # batch shape (3,)
X = torch.randn(3, 2, generator=G, **F64)


@pytest.fixture(scope="module")
def eta():
    torch.manual_seed(1)
    return LATENT.sample((10**6,)).numpy()


def built(likelihood):
    likelihood.build(2)
    return likelihood.double()


def assert_matches_mc(ell, log_p):
    """ell has shape (B,) and |ell - mean of log_p over the draws| <= 4 standard errors in every row; log_p is
    (draws, B)."""
    assert ell.shape == log_p.shape[1:]
    se = log_p.std(0) / math.sqrt(len(log_p))
    assert (np.abs(ell.detach().numpy() - log_p.mean(0)) <= 4 * se).all()


def test_ell_gaussian_vs_mc(eta):
    torch.manual_seed(0)
    y = torch.tensor([-1.0, 0.5, 4.0], **F64)
    for heteroscedastic in (False, True):
        lik = built(Gaussian(heteroscedastic))
        sd = lik.sd(X).detach().numpy()
        assert_matches_mc(lik.expected_log_prob(LATENT, y, X), stats.norm.logpdf(y.numpy(), eta, sd))


def test_ell_poisson_vs_mc(eta):
    y = torch.tensor([0.0, 3.0, 17.0], **F64)
    assert_matches_mc(built(Poisson()).expected_log_prob(LATENT, y, X), stats.poisson.logpmf(y.numpy(), np.exp(eta)))


def test_ell_gamma_vs_mc(eta):
    y, lik = torch.tensor([0.1, 2.0, 50.0], **F64), built(Gamma())
    for k in (0.5, 3.0):
        with torch.no_grad():
            lik.log_k.fill_(math.log(k))
        log_p = stats.gamma.logpdf(y.numpy(), k, scale=np.exp(eta) / k)  # shape k, mean e^eta
        assert_matches_mc(lik.expected_log_prob(LATENT, y, X), log_p)


def test_loss_is_elbo():
    torch.manual_seed(0)
    x = torch.randn(5, 2)
    for lik, y in ((Gaussian(), torch.randn(5)), (Poisson(), torch.arange(5.0)), (Gamma(), torch.rand(5) + 0.1)):
        head = Regression(2, 8, lik, num_data=50)
        out = head(x)
        assert out.latent.batch_shape == (5,)
        expected = -lik.expected_log_prob(out.latent, y, x).mean() + head.layer.kl() / 50
        assert torch.equal(out.train_loss_fn(y), expected)


def test_shape_and_support_errors():
    torch.manual_seed(0)
    x = torch.randn(4, 2)
    with pytest.raises(ValueError, match="shape"):
        Regression(2, 8, Gaussian(), num_data=10)(x).train_loss_fn(torch.zeros(4, 1))
    with pytest.raises(ValueError, match="shape"):
        Regression(2, 8, Gaussian(), num_data=10)(x).val_loss_fn(torch.zeros(4, 1))
    with pytest.raises(ValueError, match="support"):
        Regression(2, 8, Gamma(), num_data=10)(x).train_loss_fn(torch.tensor([1.0, 0.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="support"):
        Regression(2, 8, Poisson(), num_data=10)(x).train_loss_fn(torch.tensor([1.0, 1.5, 2.0, 3.0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trains_toy(dtype):
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 200, dtype=dtype)[:, None]
    f = 2 + torch.sin(3 * x[:, 0])  # the mean is f for Gaussian, e^f for Poisson and Gamma
    data = ((Gaussian(), f + 0.1 * torch.randn_like(f)), (Poisson(), torch.poisson(f.exp())),
            (Gamma(), torch.distributions.Gamma(2.0, 2 * (-f).exp()).sample()))
    for lik, y in data:
        head = Regression(1, 16, lik, num_data=200).to(dtype)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        losses = []
        for _ in range(300):
            opt.zero_grad()
            loss = head(x).train_loss_fn(y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert loss.dtype == dtype
        assert losses[-1] < 0.7 * losses[0], (type(lik).__name__, losses[0], losses[-1])


def test_head_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(4, 2, **F64)
    data = ((Gaussian(heteroscedastic=True), torch.randn(4, **F64)), (Poisson(), torch.tensor([0.0, 1, 2, 5], **F64)),
            (Gamma(), torch.rand(4, **F64) + 0.5))
    for lik, y in data:
        head = Regression(2, 3, lik, num_data=20).double()
        params = tuple(head.parameters())  # gradcheck perturbs these in place, so the head sees each perturbation
        assert torch.autograd.gradcheck(lambda *_: head(x).train_loss_fn(y), params)
