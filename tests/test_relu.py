import warnings

import numpy as np
import torch
from scipy import integrate, special

from cfvi._relu import relu_log_mgf, relu_moments

CASES = [(m, s) for m in (-3.0, -0.5, 0.0, 0.7, 4.0) for s in (0.05, 1.0, 3.0)]
QUAD = dict(epsabs=0, epsrel=1e-13, limit=1000)
F64 = dict(dtype=torch.float64)


def phi(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def quad_mgf(u, m, s):
    """E exp(u ReLU(z)) over the exact law: an atom Phi(-m/s) at 0 plus N(m, s^2) on (0, inf), in x = (h - m)/s.

    quad covers whichever side of x = -m/s holds less of the mass of exp(Re(u) s x) phi(x), and the Gaussian MGF
    exp(m u + s^2 u^2 / 2) supplies the other side: integrating (0, inf) alone loses 8e-11 relative to oscillation
    at m = 4, s = 1, u = 1 + 20i (checked against mpmath)."""
    k, a, b = m / s, u.real, u.imag
    side = 1 if k + a * s <= 0 else -1
    lo, hi = sorted((-k, a * s + 40 * side))
    f = lambda x: np.exp(a * s * x) * phi(x)
    with warnings.catch_warnings():  # QAWO flags roundoff at epsrel 1e-13; the result is within 4e-16 of mpmath
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        if b == 0:
            integral = integrate.quad(f, lo, hi, **QUAD)[0]
        else:
            integral = complex(integrate.quad(f, lo, hi, weight="cos", wvar=b * s, **QUAD)[0],
                               integrate.quad(f, lo, hi, weight="sin", wvar=b * s, **QUAD)[0])
    gauss = np.exp(m * u + (s * u) ** 2 / 2) if side < 0 else 0
    return special.ndtr(-k) + gauss + side * np.exp(m * u) * integral


def lmgf_and_quad(us):
    """relu_log_mgf, one broadcast call, and its quad reference at every u in us and (m, s) in CASES."""
    points = [(u, m, s) for m, s in CASES for u in us]
    u, m, s = zip(*points)
    dtype = torch.complex128 if isinstance(us[0], complex) else torch.float64
    out = relu_log_mgf(torch.tensor(u, dtype=dtype), torch.tensor(m, **F64), torch.tensor(s, **F64))
    return out.numpy(), np.array([quad_mgf(*p) for p in points])


def test_log_mgf_real_vs_quad():
    out, ref = lmgf_and_quad((-8.0, -1.0, 0.3, 2.0, 6.0))
    assert np.max(np.abs(np.expm1(out - np.log(ref)))) <= 1e-10  # relative error of the MGF
    m, s = torch.tensor(CASES, **F64).T
    assert torch.all(relu_log_mgf(torch.zeros_like(m), m, s) == 0)


def test_log_mgf_complex_vs_quad():
    out, ref = lmgf_and_quad([complex(a, b) for a in (-2.0, 1.0) for b in (0.5, 20.0)])
    assert np.all(np.abs(np.exp(out) - ref) <= 1e-11 * np.abs(ref))
    out, ref = lmgf_and_quad([1j * t for t in (0.01, 1.0, 10.0, 300.0)])  # the CF
    assert np.max(np.abs(np.exp(out) - ref)) <= 1e-11


def test_alpha_branch_continuous():
    """m = +-1e-9 straddles the branch switch at alpha = 0. The CF itself moves between them by
    2 m dphi/dm = 1e-9 i t w(t/sqrt 2) (exact to O(m^3)), up to 1e-9, so the test removes that move and bounds the rest,
    the seam between the two branches."""
    t = torch.logspace(-2, 3, 50, **F64)
    m = torch.tensor([[1e-9], [-1e-9]], **F64)
    cf = torch.exp(relu_log_mgf(1j * t, m, torch.ones((), **F64))).numpy()
    move = 1e-9j * t.numpy() * special.wofz(t.numpy() / np.sqrt(2))
    assert np.max(np.abs(cf[0] - cf[1] - move)) <= 1e-12


def quad_moments(m, s):
    """(E h, Var h) for h = ReLU(z) by quad over h = s y; the variance is the central moment, with no cancellation."""
    k = m / s
    lo, hi = max(0.0, k - 40), max(0.0, k + 40)
    mean = integrate.quad(lambda y: s * y * phi(y - k), lo, hi, **QUAD)[0]
    spread = integrate.quad(lambda y: (s * y - mean) ** 2 * phi(y - k), lo, hi, **QUAD)[0]
    return mean, special.ndtr(-k) * mean**2 + spread


def test_moments_vs_quad():
    """CASES, then m/s >> 1, where the textbook E h^2 - (E h)^2 cancels: its relative Var error is 5.4e-7 at
    m/s = 1e5, s = 0.05 in float64, and 1.8e-4 to 4.8e-4 at m/s = 1e2 in float32, where ours is at most 4.1e-8."""
    wide = [(k * s, s) for k in (1e2, 1e3, 1e5) for s in (0.05, 1.0, 3.0)]
    mean, var = relu_moments(*torch.tensor(CASES + wide, **F64).T)
    ref_mean, ref_var = np.array([quad_moments(*c) for c in CASES + wide]).T
    n = len(CASES)
    assert np.all(np.abs(mean.numpy()[:n] - ref_mean[:n]) <= 1e-10 * ref_mean[:n])
    assert np.all(np.abs(var.numpy()[:n] - ref_var[:n]) <= 1e-10 * ref_var[:n])
    assert np.all(np.abs(var.numpy()[n:] - ref_var[n:]) <= 1e-9 * ref_var[n:])
    m, s = torch.tensor([(1e2 * s, s) for s in (0.05, 0.37, 2.3)], dtype=torch.float32).T
    ref_var = np.array([quad_moments(*c)[1] for c in zip(m.tolist(), s.tolist())])  # at the float32 inputs, exactly
    assert np.all(np.abs(relu_moments(m, s)[1].numpy() - ref_var) <= 1e-6 * ref_var)


def test_relu_gradcheck():
    """Both real-u forms (u = 0 and away), both complex branches, and the seam Re beta = 0 (m = 0, u = 1.5i)."""
    m = torch.tensor([-2.0, -0.3, 0.0, 1e-9, 0.4, 3.0, 0.4, 1.0], **F64, requires_grad=True)
    s = torch.tensor([0.5, 1.0, 2.0, 1.0, 0.1, 1.5, 0.1, 0.3], **F64, requires_grad=True)
    u = torch.tensor([-3.0, 0.0, 0.5, 1.0, -50.0, 2.0, 7.0, -0.2], **F64, requires_grad=True)
    assert torch.autograd.gradcheck(relu_log_mgf, (u, m, s))
    uc = torch.tensor([0.5j, 3j, 1.5j, 1 + 0.5j, -50 + 2j, -0.5 + 1j, 10j, 4 - 6j], dtype=torch.complex128)
    assert torch.autograd.gradcheck(relu_log_mgf, (uc.requires_grad_(), m, s))
    assert torch.autograd.gradcheck(relu_moments, (m, s))


def test_relu_dtype_device():
    default = torch.get_default_dtype()
    for real, complex_ in ((torch.float32, torch.complex64), (torch.float64, torch.complex128)):
        for device in ("cpu", "meta"):
            m = torch.zeros(2, 3, dtype=real, device=device)
            s = torch.ones(3, dtype=real, device=device)
            for dtype in (real, complex_):
                out = relu_log_mgf(torch.ones(2, 1, dtype=dtype, device=device), m, s)
                assert (out.dtype, out.device, out.shape) == (dtype, m.device, m.shape)
            assert all((x.dtype, x.device, x.shape) == (real, m.device, m.shape) for x in relu_moments(m, s))
    assert torch.get_default_dtype() == default
