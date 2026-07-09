from __future__ import annotations

from typing import Any

import torch


class LinearWarmupMultiStepLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        milestones: tuple[int, ...],
        gamma: float = 0.1,
        warmup_iters: int = 1500,
        warmup_ratio: float = 0.001,
    ):
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if not 0 < warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in (0, 1]")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if any(milestone <= 0 for milestone in milestones):
            raise ValueError("milestones must be positive epoch numbers")

        self.optimizer = optimizer
        self.milestones = tuple(sorted(milestones))
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.iter_count = 0
        self.last_epoch = 0
        self.last_lrs = self.base_lrs.copy()

    def step(self, *, epoch: int) -> list[float]:
        if epoch <= 0:
            raise ValueError("epoch must be one-based and positive")
        factor = self._lr_factor(epoch=epoch, iter_index=self.iter_count)
        self.last_lrs = [base_lr * factor for base_lr in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, self.last_lrs):
            group["lr"] = lr
        self.iter_count += 1
        self.last_epoch = epoch
        return self.last_lrs.copy()

    def get_last_lr(self) -> list[float]:
        return self.last_lrs.copy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "milestones": self.milestones,
            "gamma": self.gamma,
            "warmup_iters": self.warmup_iters,
            "warmup_ratio": self.warmup_ratio,
            "base_lrs": self.base_lrs,
            "iter_count": self.iter_count,
            "last_epoch": self.last_epoch,
            "last_lrs": self.last_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.milestones = tuple(state_dict["milestones"])
        self.gamma = float(state_dict["gamma"])
        self.warmup_iters = int(state_dict["warmup_iters"])
        self.warmup_ratio = float(state_dict["warmup_ratio"])
        self.base_lrs = [float(lr) for lr in state_dict["base_lrs"]]
        self.iter_count = int(state_dict["iter_count"])
        self.last_epoch = int(state_dict["last_epoch"])
        self.last_lrs = [float(lr) for lr in state_dict["last_lrs"]]
        for group, lr in zip(self.optimizer.param_groups, self.last_lrs):
            group["lr"] = lr

    def _lr_factor(self, *, epoch: int, iter_index: int) -> float:
        warmup_factor = 1.0
        if self.warmup_iters > 0 and iter_index < self.warmup_iters:
            progress = iter_index / self.warmup_iters
            warmup_factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * progress
        decay_power = sum(epoch >= milestone for milestone in self.milestones)
        return warmup_factor * (self.gamma**decay_power)


class WarmupMultiStepLR:
    """Multi-step LR decay with Ultralytics-style per-iteration warmup.

    For the first ``warmup_iters`` iterations, each param group's LR ramps
    linearly to ``base_lr * gamma**k(epoch)`` — from 0.0 for ordinary groups
    and from the absolute ``warmup_bias_lr`` for groups tagged
    ``param_group="bias"`` (biases start high and ramp *down*, as in
    Ultralytics). Groups that carry a ``momentum`` value simultaneously ramp
    it from ``warmup_momentum`` to their own base momentum. After warmup the
    schedule is a plain multi-step decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        milestones: tuple[int, ...],
        gamma: float = 0.1,
        warmup_iters: int = 0,
        warmup_bias_lr: float = 0.1,
        warmup_momentum: float = 0.8,
    ):
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if warmup_bias_lr < 0:
            raise ValueError("warmup_bias_lr must be non-negative")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if any(milestone <= 0 for milestone in milestones):
            raise ValueError("milestones must be positive epoch numbers")

        self.optimizer = optimizer
        self.milestones = tuple(sorted(milestones))
        self.gamma = gamma
        self.warmup_iters = int(warmup_iters)
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.base_momentums = [
            float(group["momentum"]) if "momentum" in group else None
            for group in optimizer.param_groups
        ]
        self.iter_count = 0
        self.last_epoch = 0
        self.last_lrs = self.base_lrs.copy()

    def step(self, *, epoch: int) -> list[float]:
        if epoch <= 0:
            raise ValueError("epoch must be one-based and positive")
        decay_power = sum(epoch >= milestone for milestone in self.milestones)
        decay = self.gamma**decay_power
        in_warmup = self.iter_count < self.warmup_iters
        progress = self.iter_count / self.warmup_iters if in_warmup else 1.0

        lrs: list[float] = []
        groups = zip(self.optimizer.param_groups, self.base_lrs, self.base_momentums)
        for group, base_lr, base_momentum in groups:
            target_lr = base_lr * decay
            if in_warmup:
                start_lr = (
                    self.warmup_bias_lr
                    if group.get("param_group") == "bias"
                    else 0.0
                )
                lr = start_lr + (target_lr - start_lr) * progress
                if base_momentum is not None:
                    group["momentum"] = (
                        self.warmup_momentum
                        + (base_momentum - self.warmup_momentum) * progress
                    )
            else:
                lr = target_lr
                if base_momentum is not None:
                    group["momentum"] = base_momentum
            group["lr"] = lr
            lrs.append(lr)

        self.iter_count += 1
        self.last_epoch = epoch
        self.last_lrs = lrs.copy()
        return lrs.copy()

    def get_last_lr(self) -> list[float]:
        return self.last_lrs.copy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "milestones": self.milestones,
            "gamma": self.gamma,
            "warmup_iters": self.warmup_iters,
            "warmup_bias_lr": self.warmup_bias_lr,
            "warmup_momentum": self.warmup_momentum,
            "base_lrs": self.base_lrs,
            "base_momentums": self.base_momentums,
            "iter_count": self.iter_count,
            "last_epoch": self.last_epoch,
            "last_lrs": self.last_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.milestones = tuple(state_dict["milestones"])
        self.gamma = float(state_dict["gamma"])
        self.warmup_iters = int(state_dict["warmup_iters"])
        self.warmup_bias_lr = float(state_dict["warmup_bias_lr"])
        self.warmup_momentum = float(state_dict["warmup_momentum"])
        self.base_lrs = [float(lr) for lr in state_dict["base_lrs"]]
        self.base_momentums = [
            None if momentum is None else float(momentum)
            for momentum in state_dict["base_momentums"]
        ]
        self.iter_count = int(state_dict["iter_count"])
        self.last_epoch = int(state_dict["last_epoch"])
        self.last_lrs = [float(lr) for lr in state_dict["last_lrs"]]
        for group, lr in zip(self.optimizer.param_groups, self.last_lrs):
            group["lr"] = lr
