"""Log-MGF and moments of h = ReLU(z), z ~ N(m, s^2): the atom Phi(-m/s) at 0 plus N(m, s^2) on (0, inf)."""
import math

import torch
from torch.special import erfcx, log_ndtr

from ._faddeeva import faddeeva

_SQRT2 = math.sqrt(2)
_LOG2 = math.log(2)
_SQRT_2PI = math.sqrt(2 * math.pi)
_SQRT_HALF_PI = math.sqrt(math.pi / 2)


def relu_log_mgf(u: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """log E exp(u h): real for real u (exactly 0 at u = 0), a complex log (any branch) for complex u; broadcasts.

    E exp(u h) = Phi(-m/s) + exp(m u + s^2 u^2/2) Phi((m + s^2 u)/s). With alpha = m/(sqrt2 s) and
    beta = alpha + s u/sqrt2, the second term is e^{-alpha^2} w(-i beta)/2. w is only evaluated at Im >= 0: as is when
    Re beta <= 0, else by the reflection w(-z) = 2 e^{-z^2} - w(z), as exp(m u + s^2 u^2/2) - e^{-alpha^2} w(i beta)/2.
    For u = i t, Re beta = alpha: this is the alpha-branch. The terms are summed in log space."""
    k = m / s
    gauss = m * u + (s * u) ** 2 / 2  # log E exp(u z)
    if not u.is_complex():
        lq = log_ndtr(k)
        tail = gauss + log_ndtr(k + s * u)  # log of the second term
        d = tail - lq  # E exp(u h) = Phi(-k) + Phi(k) e^d, so log1p(Phi(k) expm1(d)) is exactly 0 at u = 0
        near = d.abs() < 0.5
        small = torch.log1p(lq.exp() * torch.expm1(torch.where(near, d, 0.0)))
        return torch.where(near, small, torch.logaddexp(log_ndtr(-k), tail))
    alpha = k / _SQRT2
    beta = alpha + s * u / _SQRT2
    up = beta.real > 0
    w = faddeeva(1j * torch.where(up, beta, -beta))
    terms = (log_ndtr(-k), torch.log(torch.where(up, -w, w)) - alpha**2 - _LOG2, torch.where(up, gauss, -math.inf))
    top = torch.maximum(torch.maximum(terms[0], terms[1].real), terms[2].real).detach()
    return top + torch.log(sum(torch.exp(t - top) for t in terms))


def relu_moments(m: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E h, Var h), with k = m/s, x = |k| and the Mills ratio r = Phi(-x)/phi(x) = sqrt(pi/2) erfcx(x/sqrt2):
    E h/s = max(k, 0) + e and Var h/s^2 = [k >= 0](1 - 2 phi r) + phi ((x^2 + 1) r - x) - e^2, with e = phi (1 - x r).

    The leading k and 1 are exact and the rest carry a factor phi(k), so nothing cancels for m/s >> 1. For m/s << 0,
    (x^2 + 1) r - x cancels to 2/x^3: relative error ~ eps x^4 (5e-12 at m/s = -20), absolute error ~ eps phi(k)."""
    k = m / s
    up = k >= 0
    x = torch.where(up, k, -k)  # |k| with slope +1 at 0, so the gradient at m = 0 is right
    p = torch.exp(-k * k / 2) / _SQRT_2PI
    r = erfcx(x / _SQRT2) * _SQRT_HALF_PI
    e = p * (1 - x * r)
    var = torch.where(up, 1 - 2 * p * r, 0.0) + p * ((x * x + 1) * r - x) - e * e
    return s * (torch.where(up, k, 0.0) + e), s * s * var
