from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfig,
    get_default_qat_qconfig,
    prepare_qat,
)


@dataclass
class PrecisionPolicy:
    mapping: dict[str, str]

    @staticmethod
    def default() -> "PrecisionPolicy":
        return PrecisionPolicy(
            mapping={
                "backbone.conv1": "int4",
                "backbone.conv2": "int4",
                "backbone.conv3": "int4",
                "reliability_head.fc1": "fp16",
                "reliability_head.fc_out": "fp16",
            }
        )


def _int4_qconfig() -> QConfig:
    act = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=15,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )
    weight = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-8,
        quant_max=7,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )
    return QConfig(activation=act, weight=weight)


def apply_heterogeneous_qat_policy(model: nn.Module) -> nn.Module:
    model.train()
    qconfig_int4 = _int4_qconfig()

    model.qconfig = None
    model.quant.qconfig = get_default_qat_qconfig("fbgemm")
    model.dequant.qconfig = None
    model.backbone.qconfig = qconfig_int4
    model.reliability_head.qconfig = None

    prepared = prepare_qat(model, inplace=False)
    return prepared


def apply_uniform_int8_qat_policy(model: nn.Module) -> nn.Module:
    model.train()
    model.qconfig = get_default_qat_qconfig("fbgemm")
    prepared = prepare_qat(model, inplace=False)
    return prepared


def has_fake_quant_in_backbone_only(model: nn.Module) -> bool:
    backbone_has = False
    head_has = False
    for name, module in model.named_modules():
        if "activation_post_process" in name or "weight_fake_quant" in name:
            if name.startswith("backbone"):
                backbone_has = True
            if name.startswith("reliability_head"):
                head_has = True
    return backbone_has and not head_has


def structured_prune_backbone(model: nn.Module, amount: float = 0.15) -> nn.Module:
    if amount <= 0.0:
        return model
    convs = [
        model.backbone.conv1,
        model.backbone.conv2,
        model.backbone.conv3,
        model.backbone.bridge,
    ]
    for layer in convs:
        prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=0)
        prune.remove(layer, "weight")
    return model
