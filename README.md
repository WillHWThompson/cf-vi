# cfvi

`cfvi` is a Bayesian last layer for PyTorch. It is one Bayesian ReLU layer, eta = V ReLU(W x + b) + c, with a
mean-field Gaussian posterior on W and b and a deterministic readout V, c. Given x, the law of eta is known exactly
through its characteristic function. So the training loss is the negative ELBO in closed form, with no sampling, and the
predictive distribution of y has a density, a cdf and a survival function, which give tail probabilities such as
P(y > t | x) without Monte Carlo. The API follows [vbll](https://github.com/VectorInstitute/vbll) (variational Bayesian
last layers). `cfvi` is the companion to the paper *Exact Bayesian layers via characteristic functions*.

## Install

```bash
pip install git+https://github.com/WillHWThompson/cf-vi
```

This installs the package `cfvi` (`import cfvi`); it will also be on PyPI as `cfvi`.

From a clone of this repository, `pip install -e ".[test]"` adds the test dependencies (pytest and scipy); then run
`pytest`. Tested on Python 3.11 with torch 2.4.

## Usage

Gamma regression on synthetic data. In practice, x is the output of your own network, which trains with the head's
loss. This is `examples/regression.py`.

```python
import torch

import cfvi

torch.manual_seed(0)
x = 4 * torch.rand(1000, 1) - 2
y = torch.distributions.Gamma(2.0, 2.0 / torch.exp(torch.sin(2 * x[:, 0]))).sample()  # shape 2, mean e^sin(2x)

head = cfvi.Regression(in_features=1, hidden_features=32, likelihood=cfvi.Gamma(), num_data=len(x))
optimizer = torch.optim.Adam(head.parameters(), lr=2e-2)
for _ in range(500):
    optimizer.zero_grad()
    loss = head(x).train_loss_fn(y)  # the negative ELBO per data point, in closed form
    loss.backward()
    optimizer.step()

head.double()  # float64 resolves smaller tail probabilities than float32
x_new = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
with torch.no_grad():
    predictive = head(x_new).predictive
    print(predictive.mean)  # E(y | x)
    print(predictive.sf(torch.tensor(4.0, dtype=torch.float64)))  # P(y > 4 | x)
```

## API

```python
cfvi.Regression(in_features, hidden_features, likelihood, num_data, prior_scale=1.0)
cfvi.Classification(in_features, hidden_features, num_classes, num_data, prior_scale=1.0)
```

`hidden_features` is the number of ReLU units. The prior is N(0, prior_scale^2) on each entry of W and b
(`prior_scale` is a standard deviation, not rescaled by fan-in). `num_data` is the size of the training set; it scales
the KL. `Regression` has one output and a likelihood. `Classification` has `num_classes` outputs and a softmax
likelihood, with y a LongTensor of class indices.

`out = head(x)`, for x of shape (B, in_features), is a `cfvi.Output`:

- `out.train_loss_fn(y)`: the negative ELBO per data point, -(batch mean of E log p(y | eta)) + KL / num_data, for y of
  shape (B,).
- `out.val_loss_fn(y)`: -(batch mean of `out.predictive.log_prob(y)`).
- `out.predictive`: the law of y given x, built on first access. For `Regression`, a `torch.distributions.Distribution`
  with `log_prob`, `cdf`, `sf`, `mean`, `variance` and `sample`; its grid can reach 65,536 points per input, so query
  large sets of inputs in batches (`val_loss_fn` too). For `Classification`, a `torch.distributions.Categorical` with
  `probs`, `log_prob` and `sample`, and no `cdf`, `sf`, `mean` or `variance`.
- `out.latent`: a `cfvi.Latent`, the law of eta given x, with `mean`, `variance`, `log_mgf(u)`, `cf(t)`, `log_atom`
  (log P(eta = c)), `log_prob` (of the continuous part), `cdf`, `sf` and `sample`.

The likelihoods are `nn.Module`s, so their parameters train with the head:

- `cfvi.Gaussian(heteroscedastic=False)`: y | eta ~ N(eta, sd^2), with sd one learned scalar or, if heteroscedastic,
  the softplus of a linear map of x.
- `cfvi.Poisson()`: y | eta ~ Poisson(e^eta).
- `cfvi.Gamma()`: y | eta ~ Gamma with mean e^eta and a learned shape.

## Accuracy

- The loss is the exact negative ELBO for `Gaussian`, `Poisson` and `Gamma`. For `Classification` it is an upper bound
  on it, by Jensen's inequality, and the predictive probabilities are proportional to E e^eta_k, not E softmax(eta).
- The regression predictive's mean and variance are exact. The predictive density and cdf come from an FFT inversion
  of the characteristic function on a grid, with the accuracy floors given in the docstrings of `cfvi.Latent` and of
  `out.predictive`. Away from the atom, the tests show the latent cdf within 1e-4 and the density within 6e-4 of its
  peak.
- `sf` is 1 - cdf. Against quadrature for one unit, `out.predictive.sf` in float64 is within 0.4% down to 1e-9 and 5%
  at 1e-12, and unreliable below; in float32, within 1% down to 1e-4 and about 6% at 1e-5. `out.latent.sf` is
  unreliable below about 1e-11 in float64 and 1e-5 in float32.

## Citation

```bibtex
@article{thompson2026exact,
  title   = {Exact {B}ayesian layers via characteristic functions},
  author  = {Thompson, William H. W. and St-Onge, Guillaume and H{\'e}bert-Dufresne, Laurent and Young, Jean-Gabriel},
  journal = {arXiv preprint arXiv:2611.26426},
  year    = {2026},
}
```

## License

MIT
