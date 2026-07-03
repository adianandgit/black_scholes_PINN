from __future__ import annotations

import numpy as np
from scipy.stats import norm
import torch


def black_scholes_price(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    """Closed-form solution for European call option prices."""
    S = np.array(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def evaluation_grid(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a uniform grid of prices for evaluation purposes."""
    S_eval = torch.linspace(config["min_S"], config["max_S"], 200).view(-1, 1)
    t_eval = torch.zeros_like(S_eval)
    return S_eval, t_eval
