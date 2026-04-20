from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import DeQuantStub, QuantStub


@dataclass
class ModelConfig:
    in_channels: int
    num_classes: int
    hidden_dim: int = 128
    dropout_p: float = 0.25
    bottleneck_channels: int = 16
    head_pool_size: int = 2


class CompressionBackbone(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, head_pool_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bridge = nn.Conv2d(128, bottleneck_channels, kernel_size=1)
        self.bridge_act = nn.ReLU()
        self.head_pool = nn.AdaptiveAvgPool2d((head_pool_size, head_pool_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bridge_act(self.bridge(x))
        x = self.head_pool(x)
        return x


class ReliabilityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_p: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout_p = dropout_p
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        if self.training or mc_dropout:
            x = F.dropout(x, p=self.dropout_p, training=(mc_dropout or self.training))
        return self.fc_out(x)


class SafetyCriticalCNN(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.backbone = CompressionBackbone(
            cfg.in_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            head_pool_size=cfg.head_pool_size,
        )

        flattened_dim = cfg.bottleneck_channels * cfg.head_pool_size * cfg.head_pool_size
        self.reliability_head = ReliabilityHead(flattened_dim, cfg.hidden_dim, cfg.num_classes, cfg.dropout_p)
        self.dequant = DeQuantStub()

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        z = self.backbone(x)
        return z.flatten(1)

    def forward_head(self, features: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        logits = self.reliability_head(features, mc_dropout=mc_dropout)
        return self.dequant(logits)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        features = self.forward_backbone(x)
        return self.forward_head(features, mc_dropout=mc_dropout)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, mc_passes: int = 20) -> dict[str, torch.Tensor]:
        self.eval()
        probs = []
        for _ in range(mc_passes):
            logits = self.forward(x, mc_dropout=True)
            probs.append(torch.softmax(logits, dim=1))
        stack = torch.stack(probs, dim=0)
        mean_prob = stack.mean(dim=0)
        var_prob = stack.var(dim=0)
        entropy = -(mean_prob * torch.log(mean_prob.clamp_min(1e-8))).sum(dim=1)
        pred = mean_prob.argmax(dim=1)
        conf = mean_prob.max(dim=1).values
        return {
            "mean_prob": mean_prob,
            "var_prob": var_prob,
            "entropy": entropy,
            "prediction": pred,
            "confidence": conf,
        }


def zone_parameter_count(model: nn.Module) -> dict[str, int]:
    compression = 0
    reliability = 0
    other = 0
    for name, p in model.named_parameters():
        if name.startswith("backbone"):
            compression += p.numel()
        elif name.startswith("reliability_head"):
            reliability += p.numel()
        else:
            other += p.numel()
    return {
        "compression_zone": compression,
        "reliability_zone": reliability,
        "other": other,
        "total": compression + reliability + other,
    }
