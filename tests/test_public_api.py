import inspect

import torch

import cfvi

SIGNATURES = {  # of each public class's __init__
    "Regression": "(self, in_features: int, hidden_features: int, likelihood: torch.nn.modules.module.Module, "
                  "num_data: int, prior_scale: float = 1.0)",
    "Classification": "(self, in_features: int, hidden_features: int, num_classes: int, num_data: int, "
                      "prior_scale: float = 1.0)",
    "Gaussian": "(self, heteroscedastic: bool = False)",
    "Poisson": str(inspect.signature(torch.nn.Module.__init__)),  # Poisson has no __init__ of its own
    "Gamma": "(self)",
    "Latent": "(self, m, s, V, c)",
    "Output": "(self, latent: cfvi.latent.Latent, likelihood: torch.nn.modules.module.Module, x: torch.Tensor, "
              "kl: torch.Tensor, num_data: int) -> None",
}


def test_public_api_snapshot():
    assert cfvi.__all__ == list(SIGNATURES)
    assert cfvi.__version__ == "0.1.0"
    for name, signature in SIGNATURES.items():
        assert str(inspect.signature(getattr(cfvi, name).__init__)) == signature, name
