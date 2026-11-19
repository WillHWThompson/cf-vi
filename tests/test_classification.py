import math

import pytest
import torch
from scipy.special import logsumexp

from cfvi import Classification, Latent
from cfvi.likelihoods import Categorical

F64 = dict(dtype=torch.float64)


def two_moons(n, noise=0.1):
    """n points, half on each of two interleaved arcs, with Gaussian noise; labels 0 and 1."""
    y = torch.arange(n) % 2
    t, sign = math.pi * torch.rand(n), 1 - 2 * y
    x = torch.stack([y + sign * t.cos(), y / 2 + sign * t.sin()], 1)
    return x + noise * torch.randn(n, 2), y


def test_bound_below_mc():
    G = torch.Generator().manual_seed(0)
    latent = Latent(torch.randn(3, 2, generator=G, **F64), 0.3 + 0.5 * torch.rand(3, 2, generator=G, **F64),
                    torch.randn(3, 2, generator=G, **F64), torch.randn(3, generator=G, **F64))  # batch (3, 3)
    y = torch.tensor([0, 1, 2])
    torch.manual_seed(1)
    eta = latent.sample((10**6,)).numpy()  # (draws, inputs, classes)
    log_p = eta[:, range(3), y] - logsumexp(eta, axis=-1)  # log softmax(eta)_y
    bound = Categorical(3).expected_log_prob(latent, y, None).numpy()
    se = log_p.std(0) / math.sqrt(len(log_p))
    assert (bound <= log_p.mean(0) + 4 * se).all()
    assert (bound > log_p.mean(0) - 0.1).all()  # a bound, not a vacuous one


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_probs_valid(dtype):
    torch.manual_seed(0)
    x = torch.randn(5, 2, dtype=dtype)
    probs = Classification(2, 8, 3, num_data=10).to(dtype)(x).predictive.probs
    assert probs.dtype == dtype and probs.shape == (5, 3)
    assert (probs > 0).all() and torch.allclose(probs.sum(-1), torch.ones((), dtype=dtype))


def test_support_error():
    with pytest.raises(ValueError, match="support"):
        Classification(2, 8, 3, num_data=10)(torch.randn(4, 2)).train_loss_fn(torch.tensor([0, 1, 2, 3]))


def test_trains_two_moons():
    torch.manual_seed(0)
    (x, y), (x_test, y_test) = two_moons(300), two_moons(300)
    head = Classification(2, 16, 2, num_data=300)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        head(x).train_loss_fn(y).backward()
        opt.step()
    accuracy = (head(x_test).predictive.probs.argmax(-1) == y_test).float().mean()
    assert accuracy >= 0.9, accuracy
