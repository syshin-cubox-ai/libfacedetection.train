from __future__ import annotations

import pytest
import torch

from yunet_train.engine import LinearWarmupMultiStepLR, WarmupLinearLR, WarmupMultiStepLR


def test_linear_warmup_multi_step_lr_matches_yunet_schedule_shape() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.01)
    scheduler = LinearWarmupMultiStepLR(
        optimizer,
        milestones=(3,),
        gamma=0.1,
        warmup_iters=4,
        warmup_ratio=0.1,
    )

    assert scheduler.step(epoch=1) == [0.001]
    assert scheduler.step(epoch=1) == [0.0032500000000000003]
    assert scheduler.step(epoch=1) == [0.0055000000000000005]
    assert scheduler.step(epoch=1) == [0.007750000000000001]
    assert scheduler.step(epoch=1) == [0.01]
    assert scheduler.step(epoch=3) == [0.001]
    assert optimizer.param_groups[0]["lr"] == 0.001


def test_linear_warmup_multi_step_lr_restores_state() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.01)
    scheduler = LinearWarmupMultiStepLR(optimizer, milestones=(3,), warmup_iters=4)
    scheduler.step(epoch=1)
    scheduler.step(epoch=1)

    new_optimizer = torch.optim.SGD([param], lr=0.5)
    restored = LinearWarmupMultiStepLR(new_optimizer, milestones=(99,), warmup_iters=0)
    restored.load_state_dict(scheduler.state_dict())

    assert restored.iter_count == scheduler.iter_count
    assert restored.get_last_lr() == scheduler.get_last_lr()
    assert new_optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0]


def _two_group_optimizer(lr: float = 0.01, momentum: float = 0.9) -> torch.optim.SGD:
    weight = torch.nn.Parameter(torch.ones(2, 2))
    bias = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.SGD(
        [
            {"params": [weight], "lr": lr, "momentum": momentum, "param_group": "weights"},
            {"params": [bias], "lr": lr, "momentum": momentum, "param_group": "bias"},
        ],
        lr=lr,
        momentum=momentum,
    )


def test_warmup_multi_step_lr_ramps_bias_down_and_momentum_up() -> None:
    optimizer = _two_group_optimizer(lr=0.01, momentum=0.9)
    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=(100,),
        gamma=0.1,
        warmup_iters=4,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
    )

    # iter 0: weights start at 0, biases at warmup_bias_lr, momentum at warmup_momentum.
    lrs = scheduler.step(epoch=1)
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[1] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.8)

    # iter 2: halfway through warmup.
    scheduler.step(epoch=1)
    lrs = scheduler.step(epoch=1)
    assert lrs[0] == pytest.approx(0.005)
    assert lrs[1] == pytest.approx(0.1 + (0.01 - 0.1) * 0.5)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.85)

    # after warmup: every group at the scheduled LR and base momentum.
    scheduler.step(epoch=1)
    lrs = scheduler.step(epoch=1)
    assert lrs == [pytest.approx(0.01), pytest.approx(0.01)]
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert optimizer.param_groups[1]["momentum"] == pytest.approx(0.9)


def test_warmup_multi_step_lr_applies_milestone_decay() -> None:
    optimizer = _two_group_optimizer(lr=0.01, momentum=0.9)
    scheduler = WarmupMultiStepLR(optimizer, milestones=(3, 5), gamma=0.1, warmup_iters=0)

    assert scheduler.step(epoch=1) == [pytest.approx(0.01)] * 2
    assert scheduler.step(epoch=3) == [pytest.approx(0.001)] * 2
    assert scheduler.step(epoch=5) == [pytest.approx(0.0001)] * 2


def test_warmup_multi_step_lr_restores_state() -> None:
    optimizer = _two_group_optimizer()
    scheduler = WarmupMultiStepLR(
        optimizer, milestones=(3,), warmup_iters=4, warmup_bias_lr=0.05
    )
    scheduler.step(epoch=1)
    scheduler.step(epoch=1)

    new_optimizer = _two_group_optimizer(lr=0.5)
    restored = WarmupMultiStepLR(new_optimizer, milestones=(99,), warmup_iters=0)
    restored.load_state_dict(scheduler.state_dict())

    assert restored.iter_count == scheduler.iter_count
    assert restored.warmup_bias_lr == scheduler.warmup_bias_lr
    assert restored.get_last_lr() == scheduler.get_last_lr()
    assert new_optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0]


def test_warmup_linear_lr_decays_linearly_to_lrf() -> None:
    optimizer = _two_group_optimizer(lr=0.01, momentum=0.9)
    scheduler = WarmupLinearLR(optimizer, epochs=10, lrf=0.1, warmup_iters=0)

    # factor(e) = max(1 - (e-1)/epochs, 0) * (1 - lrf) + lrf
    assert scheduler.step(epoch=1) == [pytest.approx(0.01)] * 2
    assert scheduler.step(epoch=6) == [pytest.approx(0.01 * (0.5 * 0.9 + 0.1))] * 2
    assert scheduler.step(epoch=10) == [pytest.approx(0.01 * (0.1 * 0.9 + 0.1))] * 2


def test_warmup_linear_lr_warmup_ramps_toward_scheduled_lr() -> None:
    optimizer = _two_group_optimizer(lr=0.01, momentum=0.9)
    scheduler = WarmupLinearLR(
        optimizer,
        epochs=100,
        lrf=0.01,
        warmup_iters=4,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
    )

    lrs = scheduler.step(epoch=1)
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[1] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.8)

    scheduler.step(epoch=1)
    lrs = scheduler.step(epoch=1)
    assert lrs[0] == pytest.approx(0.005)
    assert lrs[1] == pytest.approx(0.1 + (0.01 - 0.1) * 0.5)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.85)

    scheduler.step(epoch=1)
    lrs = scheduler.step(epoch=1)
    assert lrs == [pytest.approx(0.01), pytest.approx(0.01)]
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)


def test_warmup_linear_lr_restores_state() -> None:
    optimizer = _two_group_optimizer()
    scheduler = WarmupLinearLR(
        optimizer, epochs=10, lrf=0.05, warmup_iters=4, warmup_bias_lr=0.05
    )
    scheduler.step(epoch=1)
    scheduler.step(epoch=1)

    new_optimizer = _two_group_optimizer(lr=0.5)
    restored = WarmupLinearLR(new_optimizer, epochs=99, warmup_iters=0)
    restored.load_state_dict(scheduler.state_dict())

    assert restored.iter_count == scheduler.iter_count
    assert restored.epochs == scheduler.epochs
    assert restored.lrf == scheduler.lrf
    assert restored.get_last_lr() == scheduler.get_last_lr()
    assert new_optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0]
