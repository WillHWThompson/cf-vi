import numpy as np
import torch
from scipy.special import wofz

from cfvi._faddeeva import faddeeva


def grid():
    """The real axis, tiny and large Im z, and |z| in {1e3, 1e5} on rays across the upper half-plane."""
    x = np.linspace(-60, 60, 241)
    y = np.array([0, 1e-12, 1e-6, 1e-3, 0.1, 1, 5, 20, 100])
    rays = np.multiply.outer([1e3, 1e5], np.exp(1j * np.linspace(0, np.pi, 13)))
    return np.concatenate([(x[:, None] + 1j * y).ravel(), rays.ravel()])


def test_w_matches_scipy_float64():
    z = grid()
    w = faddeeva(torch.from_numpy(z)).numpy()
    ref = wofz(z)
    assert np.all(np.abs(w - ref) <= 1e-13 * np.abs(ref))


def test_w_matches_scipy_float32():
    z = grid().astype(np.complex64)
    w = faddeeva(torch.from_numpy(z)).numpy()
    assert np.max(np.abs(w - wofz(z.astype(np.complex128)))) <= 2e-6


def test_w_gradcheck():
    gen = torch.Generator().manual_seed(0)
    x = 20 * torch.rand(20, generator=gen, dtype=torch.float64) - 10
    y = 5 * torch.rand(20, generator=gen, dtype=torch.float64)
    y[:4] = 0  # the real axis, where the ReLU CF switches branch
    z = torch.complex(x, y).requires_grad_()
    assert torch.autograd.gradcheck(faddeeva, (z,))


def test_w_derivative_relative_accuracy():
    """Autograd w' against its asymptotic series -(2i/sqrt(pi)) sum_n (2n-1)!!/(2z^2)^n, exact to roundoff here."""
    for dtype, radius, rtol in ((np.complex128, 1e2, 1e-12), (np.complex128, 1e4, 1e-12), (np.complex64, 600, 1e-4)):
        z = (radius * np.exp(1j * np.linspace(0, np.pi, 13))).astype(dtype)
        zt = torch.from_numpy(z).requires_grad_()
        w = faddeeva(zt)
        dw = torch.autograd.grad(w, zt, torch.ones_like(w))[0].numpy().conj()
        terms = np.cumprod([(2 * n - 1) / (2 * z.astype(np.complex128) ** 2) for n in range(1, 8)], axis=0)
        ref = -2j / np.sqrt(np.pi) * terms.sum(axis=0)
        assert np.max(np.abs(dw - ref) / np.abs(ref)) <= rtol


def test_w_huge_z():
    for dtype, radius, rtol in ((np.complex64, 1e20, 1e-6), (np.complex128, 1e160, 1e-13)):
        z = np.array([radius * np.exp(1j * np.pi / 4)]).astype(dtype)
        w = faddeeva(torch.from_numpy(z)).numpy()
        ref = wofz(z.astype(np.complex128))
        assert np.all(np.isfinite(w)) and np.all(np.abs(w - ref) <= rtol * np.abs(ref))


def test_w_dtype_device():
    default = torch.get_default_dtype()
    for dtype in (torch.complex64, torch.complex128):
        for device in ("cpu", "meta"):
            z = torch.full((2, 3), 0.5 + 1j, dtype=dtype, device=device)
            w = faddeeva(z)
            assert (w.dtype, w.device, w.shape) == (dtype, z.device, z.shape)
    assert torch.get_default_dtype() == default
