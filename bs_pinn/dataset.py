from __future__ import annotations

import numpy as np
import torch
from .utils import black_scholes_price


def generate_synthetic(config: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate noisy samples from the closed-form Black-Scholes solution."""
    S = np.random.uniform(config["min_S"], config["max_S"], (config["N_data"], 1))
    t = np.random.uniform(0, config["T"], (config["N_data"], 1))
    C = black_scholes_price(S, config["K"], config["T"] - t, config["r"], config["sigma"])
    C += np.random.normal(config["bias"], config["noise_variance"], size=C.shape)
    return (
        torch.tensor(S, dtype=torch.float32, requires_grad=True),
        torch.tensor(t, dtype=torch.float32, requires_grad=True),
        torch.tensor(C, dtype=torch.float32),
    )


def collocation_points(config: dict, n: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample collocation points for enforcing the PDE."""
    S = torch.tensor(
        np.random.uniform(config["min_S"], config["max_S"], (n, 1)),
        dtype=torch.float32,
        requires_grad=True,
    )
    t = torch.tensor(
        np.random.uniform(0, config["T"], (n, 1)),
        dtype=torch.float32,
        requires_grad=True,
    )
    return S, t
