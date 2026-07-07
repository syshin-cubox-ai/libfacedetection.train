from __future__ import annotations

import math

import pytest
import torch

from yunet_train.engine import ModelEMA


def _model(value: float) -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)
    return model


def test_ema_update_moves_toward_model_with_ramped_decay() -> None:
    model = _model(1.0)
    ema = ModelEMA(model, decay=0.9999, tau=2000.0)

    with torch.no_grad():
        for param in model.parameters():
            param.fill_(2.0)
    ema.update(model)

    d = 0.9999 * (1.0 - math.exp(-1 / 2000.0))
    expected = d * 1.0 + (1.0 - d) * 2.0
    for param in ema.ema.parameters():
        assert torch.allclose(param, torch.full_like(param, expected))
    assert ema.updates == 1


def test_ema_copies_integer_buffers() -> None:
    model = _model(1.0)
    ema = ModelEMA(model)
    model.train()
    model(torch.randn(4, 2))  # bumps num_batches_tracked
    ema.update(model)
    assert int(ema.ema[1].num_batches_tracked) == int(model[1].num_batches_tracked)


def test_ema_parameters_do_not_require_grad() -> None:
    ema = ModelEMA(_model(1.0))
    assert all(not p.requires_grad for p in ema.ema.parameters())
    assert not ema.ema.training


def test_ema_state_dict_roundtrip() -> None:
    model = _model(1.0)
    ema = ModelEMA(model, decay=0.999, tau=500.0)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(3.0)
    ema.update(model)

    restored = ModelEMA(_model(0.0), decay=0.9999, tau=2000.0)
    restored.load_state_dict(ema.state_dict())

    assert restored.updates == ema.updates
    assert restored.decay == pytest.approx(0.999)
    assert restored.tau == pytest.approx(500.0)
    for restored_param, original_param in zip(
        restored.ema.parameters(), ema.ema.parameters()
    ):
        assert torch.equal(restored_param, original_param)


def test_ema_unwraps_ddp_style_module_attribute() -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module):
            super().__init__()
            self.module = module

    inner = _model(1.0)
    ema = ModelEMA(Wrapper(inner))
    with torch.no_grad():
        for param in inner.parameters():
            param.fill_(5.0)
    ema.update(Wrapper(inner))
    for key in ema.ema.state_dict():
        assert not key.startswith("module.")
