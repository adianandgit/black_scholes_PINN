import torch
import torch.nn as nn

class PINN(nn.Module):
    """Simple feed-forward neural network used as the physics-informed model."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([S, t], dim=1))
