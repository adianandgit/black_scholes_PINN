from __future__ import annotations

import torch
from .network import PINN
from .dataset import generate_synthetic, collocation_points
from .losses import total_loss


class BlackScholesSolver:
    """High-level class that wraps training and prediction for a PINN."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = PINN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.S_data, self.t_data, self.C_data = generate_synthetic(config)

    def train(self) -> None:
        for epoch in range(self.config["epochs"]):
            self.optimizer.zero_grad()
            S_colloc, t_colloc = collocation_points(self.config)
            loss, loss_data, loss_pde = total_loss(
                self.model,
                self.S_data,
                self.t_data,
                self.C_data,
                S_colloc,
                t_colloc,
                self.config["r"],
                self.config["sigma"],
            )
            loss.backward()
            self.optimizer.step()

            if epoch % self.config["log_interval"] == 0:
                print(
                    f"Epoch {epoch} | Total: {loss.item():.6f} | Data: {loss_data.item():.6f} | PDE: {loss_pde.item():.6f}"
                )

    def save(self, path: str = "model.pth") -> None:
        torch.save(self.model.state_dict(), path)

    def predict(self, S_eval: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(S_eval, t_eval)
