from __future__ import annotations

import torch

from yunet_train.engine import build_musgd_param_groups, build_sgd_param_groups


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, bias=True),
        torch.nn.BatchNorm2d(4),
        torch.nn.Conv2d(4, 4, 3, bias=False),
    )


def test_sgd_param_groups_exclude_norm_and_bias_from_decay() -> None:
    model = _model()
    groups = build_sgd_param_groups(model, lr=0.01, momentum=0.9, weight_decay=5e-4)

    by_tag = {group["param_group"]: group for group in groups}
    assert set(by_tag) == {"weights", "norm", "bias"}
    assert by_tag["weights"]["weight_decay"] == 5e-4
    assert by_tag["norm"]["weight_decay"] == 0.0
    assert by_tag["bias"]["weight_decay"] == 0.0

    # weights = the two conv kernels; norm = BN gamma; bias = conv bias + BN beta.
    assert len(by_tag["weights"]["params"]) == 2
    assert all(p.ndim >= 2 for p in by_tag["weights"]["params"])
    assert len(by_tag["norm"]["params"]) == 1
    assert len(by_tag["bias"]["params"]) == 2

    total = sum(len(group["params"]) for group in groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)


def test_musgd_param_groups_exclude_norm_and_bias_from_decay() -> None:
    model = _model()
    groups = build_musgd_param_groups(model, lr=0.01, momentum=0.9, weight_decay=5e-4)

    by_tag = {group["param_group"]: group for group in groups}
    assert set(by_tag) == {"weights", "norm", "bias"}
    assert by_tag["weights"]["use_muon"] is True
    assert by_tag["weights"]["weight_decay"] == 5e-4
    assert by_tag["norm"]["weight_decay"] == 0.0
    assert by_tag["bias"]["weight_decay"] == 0.0

    total = sum(len(group["params"]) for group in groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)
