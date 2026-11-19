"""Faddeeva w(z), Im z >= 0: Weideman's (1994) N = 40 rational approximation, one analytic formula, ~1e-15 relative."""
import math

import torch

_N = 40
_L = math.sqrt(_N / math.sqrt(2))
_SQRT_PI = math.sqrt(math.pi)


def _coefficients():
    """Fourier coefficients a_N, ..., a_1 of (L^2 + t^2) exp(-t^2) in powers of (L + it) / (L - it)."""
    m = 2 * _N
    t = _L * torch.tan(torch.arange(1 - m, m, dtype=torch.float64) * math.pi / (2 * m))
    f = torch.cat([t.new_zeros(1), (_L**2 + t**2) * torch.exp(-t**2)])
    return (torch.fft.fft(torch.fft.fftshift(f)).real[1 : _N + 1] / (2 * m)).tolist()[::-1]


_P = _coefficients()  # p(zeta), highest power first
_DP = [(_N - 1 - k) * a for k, a in enumerate(_P[:-1])]  # p'(zeta), highest power first


def _horner(coefficients, zeta):
    p = torch.zeros_like(zeta)
    for a in coefficients:
        p = p * zeta + a
    return p


def _d_zeta(z):
    d = _L - 1j * z
    return d, (_L + 1j * z) / d


class _Faddeeva(torch.autograd.Function):
    """w = (2 p(zeta) / d + 1 / sqrt(pi)) / d with d = L - iz and zeta = (L + iz) / d.

    Dividing by d twice, never by d^2, keeps very large |z| finite."""

    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        d, zeta = _d_zeta(z)
        return (2 * _horner(_P, zeta) / d + 1 / _SQRT_PI) / d

    @staticmethod
    def backward(ctx, grad):
        # w' from the rational form itself: the identity -2 z w + 2i/sqrt(pi) cancels to relative error eps |z|^2.
        d, zeta = _d_zeta(ctx.saved_tensors[0])
        dw = 1j * ((4 * _L * _horner(_DP, zeta) / d + 4 * _horner(_P, zeta)) / d + 1 / _SQRT_PI) / d / d
        return grad * dw.conj()


def faddeeva(z: torch.Tensor) -> torch.Tensor:
    """w(z) = exp(-z^2) erfc(-iz) for complex z with Im z >= 0, differentiable; dtype and device follow z."""
    return _Faddeeva.apply(z)
