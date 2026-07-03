import torch
from torch import autograd
from .network import PINN


def pde_residual(model: PINN, S: torch.Tensor, t: torch.Tensor, r: float, sigma: float) -> torch.Tensor:
    C = model(S, t)
    dC_dt = autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    dC_dS = autograd.grad(C, S, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    d2C_dS2 = autograd.grad(dC_dS, S, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    return dC_dt + 0.5 * sigma**2 * S**2 * d2C_dS2 + r * S * dC_dS - r * C


def total_loss(model: PINN, S_data: torch.Tensor, t_data: torch.Tensor, C_data: torch.Tensor,
               S_colloc: torch.Tensor, t_colloc: torch.Tensor, r: float, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    C_pred = model(S_data, t_data)
    loss_data = torch.mean((C_pred - C_data) ** 2)
    pde = pde_residual(model, S_colloc, t_colloc, r, sigma)
    loss_pde = torch.mean(pde ** 2)
    return loss_data + loss_pde, loss_data, loss_pde
