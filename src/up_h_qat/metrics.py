from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> tuple[float, list[dict[str, float]]]:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)

    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    bin_stats: list[dict[str, float]] = []

    for i in range(num_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        in_bin = (confidences > left) & (confidences <= right)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            bin_acc = accuracies[in_bin].float().mean()
            bin_conf = confidences[in_bin].mean()
            ece += (bin_conf - bin_acc).abs() * prop
            bin_stats.append(
                {
                    "bin_left": float(left.item()),
                    "bin_right": float(right.item()),
                    "accuracy": float(bin_acc.item()),
                    "confidence": float(bin_conf.item()),
                    "proportion": float(prop.item()),
                }
            )
        else:
            bin_stats.append(
                {
                    "bin_left": float(left.item()),
                    "bin_right": float(right.item()),
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "proportion": 0.0,
                }
            )
    return float(ece.item()), bin_stats


def plot_reliability_diagram(
    bin_stats: list[dict[str, float]],
    ece: float,
    title: str,
    out_file: Path,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    centers = np.array([(b["bin_left"] + b["bin_right"]) / 2.0 for b in bin_stats])
    acc = np.array([b["accuracy"] for b in bin_stats])
    width = 1.0 / max(len(bin_stats), 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    ax.bar(centers, acc, width=width * 0.9, alpha=0.7, label="Empirical accuracy")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} | ECE={ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_calibration_gap(
    bin_stats: list[dict[str, float]],
    title: str,
    out_file: Path,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    centers = np.array([(b["bin_left"] + b["bin_right"]) / 2.0 for b in bin_stats])
    gap = np.array([b["confidence"] - b["accuracy"] for b in bin_stats])
    width = 1.0 / max(len(bin_stats), 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
    ax.bar(centers, gap, width=width * 0.9, alpha=0.8)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Confidence - Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp_min(1e-4)
        return logits / temp

    def set_temperature(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        logits = logits.detach()
        labels = labels.detach()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-4)
        return float(self.temperature.item())


def fit_temperature_scaler(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    del max_iter
    logits = logits.detach()
    labels = labels.detach()
    criterion = nn.CrossEntropyLoss()

    candidates = torch.linspace(0.5, 5.0, steps=91, device=logits.device)
    best_t = 1.0
    best_nll = float("inf")
    for t in candidates:
        nll = criterion(logits / t, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t.item())
    return best_t


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-4)
    return logits / t


class VectorScaler(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.ones(num_classes, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.W + self.b

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        l2_reg: float = 1e-4,
    ) -> dict[str, list[float]]:
        logits = logits.detach()
        labels = labels.detach()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.W, self.b], lr=0.05, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            calibrated = self.forward(logits)
            reg = l2_reg * ((self.W - 1.0).pow(2).mean() + self.b.pow(2).mean())
            loss = criterion(calibrated, labels) + reg
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.W.clamp_(min=0.05, max=5.0)
            self.b.clamp_(min=-3.0, max=3.0)
        return {
            "W": self.W.detach().cpu().tolist(),
            "b": self.b.detach().cpu().tolist(),
        }


def apply_vector_scaling(logits: torch.Tensor, W: list[float], b: list[float]) -> torch.Tensor:
    w_t = torch.tensor(W, dtype=logits.dtype, device=logits.device)
    b_t = torch.tensor(b, dtype=logits.dtype, device=logits.device)
    return logits * w_t + b_t


def fit_vector_scaler(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> dict[str, list[float]]:
    scaler = VectorScaler(num_classes=logits.shape[1])
    return scaler.fit(logits, labels, max_iter=max_iter)
