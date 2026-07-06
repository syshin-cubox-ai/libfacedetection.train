from __future__ import annotations

from typing import Any

import torch
from torch import optim


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization of a 2D matrix via 5 Newton-Schulz iterations.

    Ported from the YOLO26 MuSGD optimizer (ultralytics/optim/muon.py).
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum.clone()
    if update.ndim == 4:  # conv filters: flatten to (out_channels, -1)
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    update = update * max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    """Hybrid Muon + SGD optimizer from YOLO26.

    Parameter groups with ``use_muon=True`` (>=2D weights) receive both an
    orthogonalized Muon update (scaled by ``muon``) and a standard SGD momentum
    update (scaled by ``sgd``). Groups with ``use_muon=False`` (biases, norm
    params) receive plain SGD.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    lr = group["lr"]
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)

                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))

                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    lr = group["lr"]
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss


def build_musgd_param_groups(
    model: torch.nn.Module,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool = True,
) -> list[dict[str, Any]]:
    """Split params like YOLO26: >=2D weights get Muon, 1D params get plain SGD without decay."""
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    sgd_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups: list[dict[str, Any]] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
                "use_muon": True,
            }
        )
    if sgd_params:
        groups.append(
            {
                "params": sgd_params,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": 0.0,
                "nesterov": nesterov,
                "use_muon": False,
            }
        )
    return groups
