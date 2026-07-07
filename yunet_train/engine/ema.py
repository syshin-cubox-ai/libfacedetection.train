from __future__ import annotations

import copy
import math

import torch


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    """Exponential moving average of model weights, following Ultralytics ModelEMA.

    Keeps an eval-mode deep copy of the model. After every optimizer step,
    floating-point parameters and buffers are updated as
    ``v = d * v + (1 - d) * v_model`` where the decay ramps up from 0 as
    ``d = decay * (1 - exp(-updates / tau))`` so early updates follow the live
    model closely and later updates average over ~1/(1-decay) steps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float = 0.9999,
        tau: float = 2000.0,
        updates: int = 0,
    ):
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.ema = copy.deepcopy(_unwrap(model)).eval()
        self.decay = decay
        self.tau = tau
        self.updates = updates
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        model_state = _unwrap(model).state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(model_state[key].detach(), alpha=1.0 - d)
            else:
                value.copy_(model_state[key])

    def state_dict(self) -> dict[str, object]:
        return {
            "state_dict": self.ema.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
            "tau": self.tau,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.ema.load_state_dict(state["state_dict"])
        self.updates = int(state["updates"])
        self.decay = float(state.get("decay", self.decay))
        self.tau = float(state.get("tau", self.tau))
