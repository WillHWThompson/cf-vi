import math
import warnings

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from cfvi._relu import relu_log_mgf
from cfvi import latent as latent_module
from cfvi.latent import Latent
from cfvi.layer import BayesReLU

F64 = dict(dtype=torch.float64)
KS_BOUND = 1.36 / math.sqrt(2e5) * 1.5
ONE_UNIT = (((-1.5,), (0.7,), (2.0,), 0.3), ((0.3,), (1.2,), (-0.8,), -1.0), ((2.0,), (0.5,), (1.3,), 4.0))
TWO_UNITS = (((0.3, -0.5), (1.0, 0.7), (1.2, 0.8), 0.1), ((-1.0, 0.5), (0.6, 1.1), (2.0, -1.5), -0.4),
             ((1.0, -0.5), (0.02, 1.0), (1.0, 1.0), 0.0), ((1.0, -0.5), (0.05, 1.0), (1.0, 1.0), 0.0))


def random_latent(h=8, K=None, B=3, shift=-0.3, seed=0):
    """B rows of h units with random m, s, V, c; V is (h,) with scalar c, or (K, h) with c of shape (K,)."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(B, h, generator=g, **F64) + shift
    s = 0.2 + torch.rand(B, h, generator=g, **F64)
    rows = (K,) if K else ()
    return Latent(m, s, torch.randn(rows + (h,), generator=g, **F64), torch.randn(rows, generator=g, **F64))


def latent_of(m, s, V, c):
    return Latent(torch.tensor([m], **F64), torch.tensor([s], **F64), torch.tensor(V, **F64), torch.tensor(c, **F64))


def draws(latent, n, seed=1):
    torch.manual_seed(seed)
    return latent.sample((n,))


def zones(latent, x):
    """True within 10 grid steps of c or of loc, where the 2-step damping blurs the jump at c and the point mass."""
    grid = latent.grid()
    near = 10 * grid.dx
    return ((x - latent.c).abs() < near) | ((x - grid.loc).abs() < near)


def ks_distance(latent, x, outside_zones=False):
    """sup |F - F_n| for a law with an atom at c: F(y-) = F(y) - atom at y = c, so ties at c compare on both sides.
    With outside_zones, the sup is over the draws outside zones(): KS at grid resolution."""
    y = torch.sort(x, dim=0).values
    keep = (~zones(latent, y) if outside_zones else torch.ones_like(y, dtype=torch.bool)).numpy()
    F = latent.cdf(y).numpy()
    y = y.numpy()
    F_left = F - np.exp(latent.log_atom.numpy()) * (y == latent.c.numpy())
    n = len(y)
    out = []
    for b in range(y.shape[1]):
        right = np.searchsorted(y[:, b], y[:, b], side="right") / n
        left = np.searchsorted(y[:, b], y[:, b], side="left") / n
        out.append(max(np.abs(F[:, b] - right)[keep[:, b]].max(), np.abs(F_left[:, b] - left)[keep[:, b]].max()))
    return max(out)


def unit_pdf(d, m, s, V):
    """Density of V ReLU(z), z ~ N(m, s^2), at d, off the atom at 0."""
    return np.where(d / V > 0, stats.norm.pdf(d / V, m, s) / abs(V), 0.0)


def unit_cdf(d, m, s, V):
    u = d / V
    return np.where(u >= 0, stats.norm.cdf(u, m, s), 0.0) if V > 0 else np.where(u <= 0, 1.0, stats.norm.sf(u, m, s))


def exact(x, m, s, V, c):
    """(density, cdf) of eta at x for one or two units: closed form, or quad over z_1 of unit 2's law given z_1."""
    d = x - c
    if len(m) == 1:
        return unit_pdf(d, m[0], s[0], V[0]), unit_cdf(d, m[0], s[0], V[0])
    q0, q1, top = stats.norm.cdf(-m[0] / s[0]), stats.norm.cdf(-m[1] / s[1]), m[0] + 12 * s[0]
    out = []
    with warnings.catch_warnings():  # quad warns about the kink at z_1 = d/V_1 even with it as a breakpoint
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        for e in d:
            def given(f):
                pts = [p for p in (m[0], e / V[0]) if 0 < p < top] or None
                return integrate.quad(lambda z: stats.norm.pdf(z, m[0], s[0]) * f(e - V[0] * z, m[1], s[1], V[1]), 0,
                                      top, points=pts, epsabs=1e-11, limit=200)[0]
            out.append((q0 * unit_pdf(e, m[1], s[1], V[1]) + q1 * unit_pdf(e, m[0], s[0], V[0]) + given(unit_pdf),
                        q0 * unit_cdf(e, m[1], s[1], V[1]) + given(unit_cdf)))
    return np.array(out).T


def errors(case, points):
    """On mean +- 6 sd and around c: outside zones(), max |density error| / peak and max |cdf error|; inside, max
    |cdf error| / (peak density * step), and whether log_prob has a NaN."""
    latent = latent_of(*case)
    mean, sd, step = latent.mean.item(), latent.variance.sqrt().item(), latent.grid().dx.item()
    x = np.concatenate([np.linspace(mean - 6 * sd, mean + 6 * sd, points), case[3] + step * np.linspace(-12, 12, 49)])
    pdf, cdf = exact(x, *case)
    xt = torch.from_numpy(x)[:, None]
    zone = zones(latent, xt)[:, 0].numpy()
    lp = latent.log_prob(xt)[:, 0]
    f, F = lp.exp().numpy(), latent.cdf(xt)[:, 0].numpy()
    return (np.abs(f - pdf)[~zone].max() / pdf.max(), np.abs(F - cdf)[~zone].max(),
            np.abs(F - cdf)[zone].max() / (pdf.max() * step), torch.isnan(lp).any().item())


def test_mean_var_match_mc():
    latent = random_latent(K=1)
    assert latent.batch_shape == (3, 1) and latent.event_shape == ()
    x = draws(latent, 10**6)
    d = x - x.mean(0)
    var = (d**2).mean(0)
    assert torch.all((latent.mean - x.mean(0)).abs() <= 4 * x.std(0) / 1e3)
    assert torch.all((latent.variance - var).abs() <= 4 * ((d**4).mean(0) - var**2).sqrt() / 1e3)
    flat = Latent(latent.m, latent.s, latent.V[0], latent.c[0])  # the (h,) form gives the same law with batch (B,)
    assert flat.batch_shape == (3,) and torch.allclose(flat.variance, latent.variance[:, 0], rtol=1e-14, atol=0)


def test_log_mgf_matches_units():
    latent = random_latent(K=2)
    for u in (torch.randn(3, 2, **F64), 1j * torch.randn(3, 2, **F64)):  # (B, K)-shaped u
        ref = [[latent.c[k] * u[b, k] + sum(relu_log_mgf(u[b, k] * latent.V[k, j], latent.m[b, j], latent.s[b, j])
                                            for j in range(8)) for k in range(2)] for b in range(3)]
        assert torch.allclose(latent.log_mgf(u), torch.tensor(ref), rtol=1e-14, atol=1e-14)


def test_atom_mass_vs_mc():
    latent = random_latent(shift=-1.0)
    x = draws(latent, 10**6)
    p = latent.log_atom.exp()
    assert torch.all(p > 0.01)
    assert torch.all(((x == latent.c).double().mean(0) - p).abs() <= 4 * (p * (1 - p) / 1e6).sqrt())


def test_density_integrates():
    """exp(log_prob) is linear between grid points and 0 beyond: quad cannot get through its kinks to 1e-8, so each
    grid cell gets 2-point Gauss-Legendre, which is exact on it."""
    latent = random_latent()
    eta, density, _, log_mass, dx = latent.grid()
    node, weight = np.polynomial.legendre.leggauss(2)
    for b in range(3):
        mid, half = (eta[b, 1:] + eta[b, :-1]).numpy() / 2, dx[b].item() / 2
        x = torch.from_numpy(mid[:, None] + half * node)[..., None]
        total = (latent.log_prob(x)[..., b].exp().numpy() * half * weight).sum()
        assert abs(total + latent.log_atom[b].exp().item() - 1) <= 1e-8
    assert torch.allclose(density.sum(-1) * dx + log_mass.exp(), torch.ones((), **F64), atol=1e-10)


def test_cdf_ks_vs_mc():
    """Random units; then three units, one of them ten times narrower than the others and on half the time, which
    drives the grid size (KS 0.011 at 4096 points)."""
    narrow = latent_of((-1.0, -0.5, 0.0), (1.0, 0.8, 0.1), (1.0, -1.2, 1.0), 0.3)
    for latent in (random_latent(), narrow):
        assert ks_distance(latent, draws(latent, 2 * 10**5)) <= KS_BOUND


@pytest.mark.parametrize("case", ONE_UNIT + TWO_UNITS)
def test_vs_exact(case):
    """test_one_unit_exact and test_two_units_vs_quad: one unit against its closed form, two against quad of their
    convolution. Outside zones(): density to 1e-3 of the peak and cdf to 1e-4; inside: cdf to peak * step. The last
    two have an almost surely on unit (s = 0.02, 0.05) 4 and 10 grid steps wide at n = 4096, which must raise n."""
    density, cdf, cdf_in_zone, nan = errors(case, 2001 if len(case[0]) == 1 else 201)
    assert density <= 1e-3 and cdf <= 1e-4 and cdf_in_zone <= 1 and not nan


@pytest.mark.parametrize("narrow_m", [(0.8, -1e-5, 1.3), (-1e-5, -1e-5, -1e-5)])
def test_near_deterministic_units(narrow_m):
    """s = 1e-6 on 3 of 8 units, far below the grid step: two on and one off (m/s = -10), or all three off. The two on
    units are folded into a point mass at loc, where the draws spread over 1e-6: KS at grid resolution."""
    base = random_latent()
    m, s = base.m.clone(), base.s.clone()
    m[:, :3], s[:, :3] = torch.tensor(narrow_m, **F64), 1e-6
    latent = Latent(m, s, base.V, base.c)
    x = draws(latent, 2 * 10**5)
    assert not torch.isnan(latent.log_prob(x[:10**4])).any()
    assert ks_distance(latent, x, outside_zones=True) <= KS_BOUND


def test_almost_always_off():
    """m/s = -40: the continuous mass, about 8 Phi(-40) = 3e-349, underflows; log_prob may be -inf there, never NaN."""
    base = random_latent(B=1, seed=2)
    s, V, c = base.s, base.V, base.c
    latent = Latent(-40 * s, s, V, c)
    atom = latent.log_atom.exp()
    assert torch.allclose(atom, torch.ones((), **F64), rtol=1e-12, atol=0)
    assert torch.allclose(latent.cdf(c) - latent.cdf(c - 1e-9), atom, rtol=1e-12, atol=0)
    assert not torch.isnan(latent.log_prob((c + V * s[0] * torch.linspace(-1, 1, 21, **F64)[:, None]))).any()


def test_negligible_unit():
    """A ninth unit with V = 1e-5 leaves the grid size and, beyond 10 steps of c, the cdf unchanged. Within them the
    cdf does change: the unit is not resolved, and its on-mass, up to 1e-2 here, is blurred over about 2 steps."""
    base = random_latent(h=7, seed=2)
    g = torch.Generator().manual_seed(5)
    m = torch.cat([base.m, torch.randn(3, 1, generator=g, **F64)], -1)
    s = torch.cat([base.s, 0.2 + torch.rand(3, 1, generator=g, **F64)], -1)
    latent = Latent(m, s, torch.cat([base.V, torch.tensor([1e-5], **F64)]), base.c)
    grid = base.grid()
    assert latent.grid().eta.shape == grid.eta.shape
    x = base.mean + base.variance.sqrt() * torch.linspace(-6, 6, 2001, **F64)[:, None]
    far = (x - base.c).abs() >= 10 * grid.dx
    assert (latent.cdf(x) - base.cdf(x))[far].abs().max() <= 1e-4


def test_zero_weight():
    """Output 0 has V_j = 0 for one unit, output 1 for all of them (eta = c surely, zero variance): finite values
    (log_prob is -inf for output 1, which has no continuous part) and gradients."""
    base = random_latent(K=2)
    m, s, V = base.m.clone().requires_grad_(), base.s.clone().requires_grad_(), base.V.clone()
    V[0, 2], V[1] = 0, 0
    V.requires_grad_()
    latent = Latent(m, s, V, base.c)
    x = latent.mean.detach() + latent.variance.sqrt().detach() * torch.tensor([-1.0, 0.0, 1.0], **F64)[:, None, None]
    lp, F = latent.log_prob(x), latent.cdf(x)
    (torch.where(torch.isfinite(lp), lp, 0).sum() + F.sum()).backward()
    assert torch.isfinite(lp[..., 0]).all() and (lp[..., 1] == -math.inf).all()
    assert all(torch.isfinite(v).all() for v in (F, m.grad, s.grad, V.grad))


def test_early_stop(monkeypatch):
    """A dense layer's CF is negligible well before the grid's last frequency; a readout row of zeros, whose psi is
    round-off, must not hold up the early stop."""
    seen = []
    count = lambda u, m, s: seen.append(u.shape[-1]) or relu_log_mgf(u, m, s)
    monkeypatch.setattr(latent_module, "relu_log_mgf", count)
    torch.manual_seed(0)
    layer = BayesReLU(10, 64, 2).double()
    with torch.no_grad():
        layer.readout.weight[1] = 0
        layer(torch.randn(20, 10, dtype=torch.float64)).grid()
    assert sum(seen) / 64 < 1024


def test_kl_vs_torch():
    layer = BayesReLU(5, 7, 2, prior_scale=0.7).double()
    N = torch.distributions.Normal
    ref = sum(torch.distributions.kl_divergence(N(mean, log_sd.exp()), N(torch.zeros((), **F64), 0.7)).sum()
              for mean, log_sd in ((layer.w_mean, layer.w_log_sd), (layer.b_mean, layer.b_log_sd)))
    assert torch.allclose(layer.kl(), ref, rtol=1e-12, atol=0)


def test_layer_dtype_follows_input():
    default = torch.get_default_dtype()
    for dtype, complex_ in ((torch.float32, torch.complex64), (torch.float64, torch.complex128)):
        torch.manual_seed(0)
        layer = BayesReLU(4, 6, 2).to(dtype)
        latent = layer(torch.randn(5, 4, dtype=dtype))
        assert latent.batch_shape == (5, 2)
        outs = (latent.mean, latent.variance, latent.log_atom, latent.log_prob(latent.mean), latent.cdf(latent.mean),
                latent.sf(latent.mean), latent.log_mgf(torch.ones((), dtype=dtype)), layer.kl(),
                latent.cdf(latent.mean.double()))  # a float64 x on the float32 latent too
        assert all(o.dtype == dtype and torch.isfinite(o).all() for o in outs)
        assert latent.cf(torch.ones((), dtype=dtype)).dtype == complex_
    assert torch.get_default_dtype() == default
