from __future__ import annotations

import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .data import DataConfig, make_dataloaders, seed_everything
from .metrics import (
    apply_temperature,
    apply_vector_scaling,
    expected_calibration_error,
    fit_vector_scaler,
    fit_temperature_scaler,
    plot_calibration_gap,
    plot_reliability_diagram,
)
from .model import ModelConfig, SafetyCriticalCNN, zone_parameter_count
from .quantization import (
    PrecisionPolicy,
    apply_heterogeneous_qat_policy,
    apply_uniform_int8_qat_policy,
    has_fake_quant_in_backbone_only,
    structured_prune_backbone,
)


@dataclass
class TrainConfig:
    dataset: str = "mnist"
    mode: str = "fp32"
    epochs: int = 2
    batch_size: int = 128
    lr: float = 1e-3
    num_workers: int = 2
    seed: int = 42
    max_train_samples: int | None = 12000
    max_test_samples: int | None = 4000
    output_dir: str = "outputs"
    mc_passes: int = 20
    ece_bins: int = 15
    prune_amount: float = 0.15
    val_split: float = 0.1
    enable_temperature_scaling: bool = True
    temperature_max_iter: int = 50
    label_smoothing: float = 0.0
    dropout_p: float = 0.4
    mixup_alpha: float = 0.0
    calibration_method: str = "temperature"
    vector_max_iter: int = 100
    # Heterogeneous Fine-Tuning (H-FT) parameters
    enable_hetero_ft: bool = False
    hetero_ft_epochs: int = 2
    hetero_ft_mixup_alpha: float = 0.05
    hetero_ft_label_smoothing: float = 0.05
    hetero_ft_lr: float = 1e-4


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_backbone_trainable(model: SafetyCriticalCNN, trainable: bool = True) -> None:
    """Freeze or unfreeze all parameters in the compression backbone."""
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def _build_model(in_channels: int, num_classes: int, dropout_p: float) -> SafetyCriticalCNN:
    cfg = ModelConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=96,
        dropout_p=dropout_p,
        bottleneck_channels=16,
        head_pool_size=2,
    )
    return SafetyCriticalCNN(cfg)


def _train_one_epoch(
    model: SafetyCriticalCNN,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    mode: str,
    cfg: TrainConfig,
) -> float:
    model.train()
    scaler = GradScaler(device="cuda", enabled=(mode == "hetero_int4_fp16" and device.type == "cuda"))
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if cfg.mixup_alpha > 0.0:
            lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
            index = torch.randperm(x.size(0), device=device)
            x_in = lam * x + (1.0 - lam) * x[index, :]
            y_a = y
            y_b = y[index]
        else:
            lam = 1.0
            x_in = x
            y_a = y
            y_b = y

        def mixed_loss(logits: torch.Tensor) -> torch.Tensor:
            if cfg.mixup_alpha > 0.0:
                return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            return criterion(logits, y)

        if mode == "hetero_int4_fp16" and device.type == "cuda":
            # Backbone stays in regular precision + fake quantization; head runs in fp16 autocast.
            features = model.forward_backbone(x_in)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model.forward_head(features, mc_dropout=False)
                loss = mixed_loss(logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_in)
            loss = mixed_loss(logits)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _collect_logits_targets(model: SafetyCriticalCNN, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


@torch.no_grad()
def _evaluate_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = probs.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def _model_size_bytes(model: nn.Module) -> int:
    buff = io.BytesIO()
    torch.save(model.state_dict(), buff)
    return buff.getbuffer().nbytes


def _estimated_deploy_size_bytes(model: nn.Module, mode: str) -> int:
    total_bits = 0
    for name, param in model.named_parameters():
        if mode in {"uniform_int8", "hetero_int4_fp16"} and name.startswith("backbone"):
            numel = int(param.count_nonzero().item())
        else:
            numel = param.numel()
        if mode == "fp32":
            bits = 32
        elif mode == "uniform_int8":
            bits = 8
        elif mode == "hetero_int4_fp16":
            if name.startswith("backbone"):
                bits = 4
            elif name.startswith("reliability_head"):
                bits = 16
            else:
                bits = 16
        else:
            bits = 32
        total_bits += numel * bits
    return int(math.ceil(total_bits / 8.0))


def _peak_vram_bytes() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def train_and_evaluate(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = _device()

    data_cfg = DataConfig(
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_train_samples=cfg.max_train_samples,
        max_test_samples=cfg.max_test_samples,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    train_loader, val_loader, test_loader, in_channels, num_classes = make_dataloaders(data_cfg)

    model = _build_model(in_channels, num_classes, dropout_p=cfg.dropout_p)
    policy = PrecisionPolicy.default()

    if cfg.mode in {"uniform_int8", "hetero_int4_fp16"}:
        model = structured_prune_backbone(model, amount=cfg.prune_amount)

    if cfg.mode == "hetero_int4_fp16":
        model = apply_heterogeneous_qat_policy(model)
    elif cfg.mode == "uniform_int8":
        model = apply_uniform_int8_qat_policy(model)
    elif cfg.mode != "fp32":
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    losses = []
    for _ in range(cfg.epochs):
        epoch_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.mode, cfg)
        losses.append(epoch_loss)
    
    # === HETEROGENEOUS FINE-TUNING (H-FT) RECOVERY PHASE ===
    if cfg.enable_hetero_ft and cfg.mode == "hetero_int4_fp16" and cfg.hetero_ft_epochs > 0:
        print("[H-FT] Starting Head-Only Recovery Phase")
        # Freeze the backbone (INT4 feature extractor)
        set_backbone_trainable(model, trainable=False)
        
        # Create optimizer for only the head parameters
        head_optimizer = torch.optim.Adam(model.reliability_head.parameters(), lr=cfg.hetero_ft_lr)
        
        # Use reduced regularization for recovery
        recovery_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.hetero_ft_label_smoothing)
        recovery_cfg = TrainConfig(
            dataset=cfg.dataset,
            mode=cfg.mode,
            epochs=cfg.hetero_ft_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.hetero_ft_lr,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            max_train_samples=cfg.max_train_samples,
            max_test_samples=cfg.max_test_samples,
            output_dir=cfg.output_dir,
            mc_passes=cfg.mc_passes,
            ece_bins=cfg.ece_bins,
            prune_amount=cfg.prune_amount,
            val_split=cfg.val_split,
            enable_temperature_scaling=cfg.enable_temperature_scaling,
            temperature_max_iter=cfg.temperature_max_iter,
            label_smoothing=cfg.hetero_ft_label_smoothing,
            dropout_p=cfg.dropout_p,
            mixup_alpha=cfg.hetero_ft_mixup_alpha,
            calibration_method=cfg.calibration_method,
            vector_max_iter=cfg.vector_max_iter,
            enable_hetero_ft=False,  # Don't recurse H-FT
        )
        
        for _ in range(cfg.hetero_ft_epochs):
            epoch_loss = _train_one_epoch(
                model, train_loader, head_optimizer, recovery_criterion, device, cfg.mode, recovery_cfg
            )
            losses.append(epoch_loss)
        
        # Re-enable backbone parameters for consistency (though backbone won't be updated)
        set_backbone_trainable(model, trainable=True)
        print("[H-FT] Recovery phase complete")
    
    train_time_s = time.perf_counter() - start

    test_logits, targets = _collect_logits_targets(model, test_loader, device)
    probs = torch.softmax(test_logits, dim=1)
    accuracy = _evaluate_accuracy(probs, targets)
    ece_raw, bin_stats_raw = expected_calibration_error(probs, targets, num_bins=cfg.ece_bins)

    calibrated_temperature = None
    vector_scaler_params = None
    ece_calibrated = None
    probs_calibrated = None
    bin_stats_calibrated = None
    if cfg.enable_temperature_scaling and val_loader is not None and cfg.calibration_method != "none":
        val_logits, val_targets = _collect_logits_targets(model, val_loader, device)
        if cfg.calibration_method == "vector":
            vector_scaler_params = fit_vector_scaler(
                val_logits,
                val_targets,
                max_iter=cfg.vector_max_iter,
            )
            calibrated_test_logits = apply_vector_scaling(
                test_logits,
                vector_scaler_params["W"],
                vector_scaler_params["b"],
            )
        else:
            calibrated_temperature = fit_temperature_scaler(
                val_logits,
                val_targets,
                max_iter=cfg.temperature_max_iter,
            )
            calibrated_test_logits = apply_temperature(test_logits, calibrated_temperature)
        probs_calibrated = torch.softmax(calibrated_test_logits, dim=1)
        ece_calibrated, bin_stats_calibrated = expected_calibration_error(
            probs_calibrated,
            targets,
            num_bins=cfg.ece_bins,
        )

    out_dir = Path(cfg.output_dir) / cfg.dataset / cfg.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = out_dir / "model_state.pt"
    torch.save(model.cpu().state_dict(), checkpoint_file)

    plot_reliability_diagram(
        bin_stats_raw,
        ece_raw,
        title=f"{cfg.dataset} {cfg.mode} raw",
        out_file=out_dir / "reliability_raw.png",
    )
    plot_calibration_gap(
        bin_stats_raw,
        title=f"{cfg.dataset} {cfg.mode} raw calibration gap",
        out_file=out_dir / "calibration_gap_raw.png",
    )
    if ece_calibrated is not None and bin_stats_calibrated is not None:
        plot_reliability_diagram(
            bin_stats_calibrated,
            ece_calibrated,
            title=f"{cfg.dataset} {cfg.mode} calibrated",
            out_file=out_dir / "reliability_calibrated.png",
        )
        plot_calibration_gap(
            bin_stats_calibrated,
            title=f"{cfg.dataset} {cfg.mode} calibrated calibration gap",
            out_file=out_dir / "calibration_gap_calibrated.png",
        )

    # Safety check: heterogeneous path must not quantize the reliability head.
    hetero_backbone_only = (
        has_fake_quant_in_backbone_only(model)
        if cfg.mode == "hetero_int4_fp16"
        else None
    )

    mc_example = None
    x0, _ = next(iter(test_loader))
    x0 = x0[:16].to(device)
    if cfg.mode == "hetero_int4_fp16":
        unc = model.predict_with_uncertainty(x0, mc_passes=cfg.mc_passes)
        mc_example = {
            "mean_entropy": float(unc["entropy"].mean().item()),
            "mean_confidence": float(unc["confidence"].mean().item()),
        }

    result = {
        "config": asdict(cfg),
        "precision_policy": policy.mapping,
        "loss_history": losses,
        "accuracy": accuracy,
        "ece": ece_raw,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_calibrated,
        "calibration_method": cfg.calibration_method,
        "temperature": calibrated_temperature,
        "vector_scaler": vector_scaler_params,
        "train_time_s": train_time_s,
        "peak_vram_bytes": _peak_vram_bytes(),
        "model_checkpoint_bytes": _model_size_bytes(model.cpu()),
        "model_checkpoint_path": str(checkpoint_file),
        "model_size_bytes": _estimated_deploy_size_bytes(model, cfg.mode),
        "zone_parameter_count": zone_parameter_count(model),
        "hetero_backbone_only_fake_quant": hetero_backbone_only,
        "mc_dropout_uncertainty": mc_example,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
