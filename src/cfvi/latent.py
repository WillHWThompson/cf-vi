"""The exact law of a Bayesian ReLU layer's output, eta = c + sum_j V_j ReLU(z_j), z_j ~ N(m_j, s_j^2) independent.

Mean, variance, log-MGF and the atom P(eta = c) are exact. Density and cdf: one FFT inversion of the continuous part's
characteristic function, damped by exp(-2 dx^2 t^2), on a grid of step dx over mean +- (8 sd + the farthest a unit moves
eta). The grid has 4096 to 65536 points: 64 steps across s|V| of each unit that is on at times and moves eta by a step
at 4096 points. A unit narrower than a step and almost surely on is folded into the point mass as the constant V m."""
import math
from collections import namedtuple
from functools import cached_property

import torch
from torch.distributions import Distribution, constraints
from torch.special import log_ndtr

from ._relu import relu_log_mgf, relu_moments

_SDS = 8.0  # Phi(-8) = 6e-16: a normal is negligible beyond 8 sds
_N_MIN, _N_MAX = 4096, 65536
_STEPS = 64  # grid steps across a unit's scale s|V|; the damping then biases its density by about 5e-4 of its peak
Grid = namedtuple("Grid", "eta density loc log_mass dx")  # what Latent.grid() returns


class Latent(Distribution):
    """Law of eta = c + V ReLU(z) given the input, z ~ N(m, diag s^2).

    m, s: (B, h). V: (h,) with scalar c, for batch shape (B,); or (K, h) with c of shape (K,), for batch shape (B, K)
    (the K outputs share the units). log_prob is the log density of the continuous part (-inf where it underflows),
    without the point mass; cdf and sf include it. The grid is cached with its autograd graph: one Latent per step.

    Accuracy floors (dx = grid step; zones = within 10 steps of c and of loc):
    - exp(log_prob) is the continuous part's density smoothed by N(0, (2 dx)^2), interpolated linearly.
    - In the zones the cdf is off by up to about peak density * dx, from the blurred jump at c, plus two effects: after
      a fold, the cdf is a step at loc where the true law has a spike of width s|V| (off by up to exp(log_mass) / 2);
      and an unfolded unit narrower than a step has its on-mass blurred over about 2 dx.
    - Outside the zones, in the tests (one and two units): up to 6e-4 of the peak density, 1e-4 in the cdf."""

    arg_constraints = {}
    support = constraints.real

    def __init__(self, m, s, V, c):
        self.m, self.s, self.V, self.c = m, s, V, c
        self._m, self._s = (m.unsqueeze(-2), s.unsqueeze(-2)) if V.dim() == 2 else (m, s)
        super().__init__(torch.broadcast_shapes(self._m.shape[:-1], V.shape[:-1], c.shape), validate_args=False)

    @property
    def mean(self):
        return self.c + (self.V * relu_moments(self._m, self._s)[0]).sum(-1)

    @property
    def variance(self):
        return (self.V**2 * relu_moments(self._m, self._s)[1]).sum(-1)

    @property
    def log_atom(self):
        """The exact log P(eta = c). cdf and the predictives use grid()'s point mass (loc, log_mass) instead, which is
        the same unless a unit folds."""
        return torch.where(self.V != 0, log_ndtr(-self._m / self._s), 0.0).sum(-1)

    def log_mgf(self, u):
        """log E exp(u eta) for real or complex u broadcastable to the batch shape."""
        units = u.unsqueeze(-1) * self.V if torch.is_tensor(u) else u * self.V
        return relu_log_mgf(units, self._m, self._s).sum(-1) + u * self.c

    def cf(self, t):
        return self.log_mgf(1j * t).exp()

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            m, s = torch.broadcast_tensors(self._m, self._s)
            z = m + s * torch.randn(torch.Size(sample_shape) + m.shape, dtype=m.dtype, device=m.device)
            return (torch.relu(z) * self.V).sum(-1) + self.c

    def _layout(self):
        """(lo, dx, n, unfolded, loc, log_mass): the grid lo + dx * arange(n), with lo and dx of shape (*batch, 1);
        V with the folded units' weights set to 0; and the point mass exp(log_mass) at loc."""
        V, m, s = self.V, self._m, self._s
        k, sv = m / s, s * V.abs()  # a unit's z-score of 0, and its width s|V| when on
        reach = V.abs() * torch.relu(m + _SDS * s)  # how far a unit moves eta
        var = self.variance.unsqueeze(-1)
        safe = torch.where(var > 0, var, 1.0)  # sqrt'(0) would give a NaN gradient
        sd = torch.where(var > 0, safe.sqrt(), 0.0)
        half = _SDS * sd + reach.amax(-1, keepdim=True)  # sd alone misses a far continuous part next to a heavy atom
        half = torch.where(half > 0, half, 1.0)  # eta = c up to round-off: every unit off, or every V_j = 0
        resolve = (k > -_SDS) & (reach >= 2 * half / _N_MIN)  # on at times, and moves eta by a step at _N_MIN points
        points = (2 * _STEPS * half / torch.where(resolve, sv, math.inf)).max().item()  # _STEPS steps across each s|V|
        n = 2 ** math.ceil(math.log2(min(max(points, _N_MIN), _N_MAX)))  # a power of 2 in [_N_MIN, _N_MAX]
        dx = 2 * half / n
        fold = (sv < dx) & (k >= _SDS)  # narrower than a step and almost surely on: the constant V m
        unfolded = torch.where(fold, 0.0, V)  # V without the folded units
        loc = self.c + torch.where(fold, V * m, 0.0).sum(-1)
        log_mass = torch.where(unfolded != 0, log_ndtr(-k), 0.0).sum(-1)  # log P(every unfolded unit is off)
        return self.mean.unsqueeze(-1) - half, dx, n, unfolded, loc, log_mass

    def _continuous_cf(self, t, lo, unfolded, loc, log_mass):
        """At t (*batch, T): the CF of eta - lo, with the folded units in loc, minus the point mass's term."""
        m, s = self._m, self._s
        log_cf = sum(relu_log_mgf(1j * t * unfolded[..., j, None], m[..., j, None], s[..., j, None])
                     for j in range(m.shape[-1]))  # one unit at a time, to bound the memory
        return torch.exp(1j * t * (loc.unsqueeze(-1) - lo)) * (log_cf.exp() - log_mass.exp().unsqueeze(-1))

    @cached_property
    def _inverted(self):
        """(grid(), the continuous part's cdf on the grid)."""
        lo, dx, n, unfolded, loc, log_mass = self._layout()
        eps = torch.finfo(dx.dtype).eps
        mass = -torch.expm1(log_mass).unsqueeze(-1)  # of the continuous part
        index = torch.arange(n // 2 + 1, dtype=dx.dtype, device=dx.device)
        blocks, start = [], 0
        while start < len(index):  # blocks of 64, 128, 256, ... frequencies, until the CF is below round-off
            t = index[start:2 * start + 64] * (2 * math.pi / n) / dx
            cf = self._continuous_cf(t, lo, unfolded, loc, log_mass) * torch.exp(-2 * (dx * t) ** 2)
            blocks.append(torch.where(mass > eps, cf, 0.0))  # with a continuous mass below eps, the CF is round-off
            start = 2 * start + 64
            if (blocks[-1].abs() <= eps * mass).all():
                break
        cf = torch.cat(blocks, -1).conj()  # the density needs e^{-itx}; irfft sums with e^{+itx}
        density = (torch.fft.irfft(cf, n=n) / dx).clamp_min(0)  # irfft zero-fills the frequencies not evaluated
        continuous_cdf = dx * torch.cat([torch.zeros_like(dx), torch.cumulative_trapezoid(density)], -1)
        eta = lo + dx * torch.arange(n, dtype=dx.dtype, device=dx.device)
        return Grid(eta, density, loc, log_mass, dx.squeeze(-1)), continuous_cdf

    def grid(self):
        """The predictives' input, a Grid: the grid eta (*batch, n) of step dx (*batch); the continuous part's density
        on it, of mass 1 - exp(log_mass); the point mass exp(log_mass) at loc (*batch)."""
        return self._inverted[0]

    def _on_grid(self, values, x):
        """values (*batch, n) at the grid points, interpolated linearly at x and held constant beyond the grid."""
        grid, n = self.grid(), values.shape[-1]
        pos = ((x - grid.eta[..., 0]) / grid.dx).clamp(0, n - 1)
        i = pos.floor().clamp(max=n - 2)
        values = values.expand(pos.shape + (n,))
        i, w = i.long().unsqueeze(-1), (pos - i).unsqueeze(-1).to(values.dtype)  # x may be float64 on a float32 grid
        return torch.lerp(values.gather(-1, i), values.gather(-1, i + 1), w).squeeze(-1)

    def log_prob(self, x):
        grid = self.grid()
        f = torch.where((x >= grid.eta[..., 0]) & (x <= grid.eta[..., -1]), self._on_grid(grid.density, x), 0.0)
        safe = torch.where(f > 0, f, 1.0)  # log'(0) would give a NaN gradient
        return torch.where(f > 0, safe.log(), -math.inf)

    def cdf(self, x):
        grid, continuous_cdf = self._inverted
        point_mass = grid.log_mass.exp() * (x >= grid.loc)
        return (self._on_grid(continuous_cdf, x) + point_mass).clamp(0, 1)  # round-off can pass 1

    def sf(self, x):
        return 1 - self.cdf(x)
