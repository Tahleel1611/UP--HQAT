from __future__ import annotations

import json
from pathlib import Path

from .train import TrainConfig, train_and_evaluate


def _defaults_for_dataset(dataset: str) -> dict:
    if dataset == "mnist":
        return {
            "epochs": 2,
            "batch_size": 128,
            "lr": 1e-3,
            "max_train_samples": 12000,
            "max_test_samples": 4000,
        }
    if dataset == "cifar10":
        return {
            "epochs": 3,
            "batch_size": 128,
            "lr": 8e-4,
            "max_train_samples": 20000,
            "max_test_samples": 5000,
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_all_modes(
    dataset: str,
    output_dir: Path,
    seed: int = 42,
    num_workers: int = 2,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    prune_amount: float = 0.15,
    val_split: float = 0.1,
    enable_temperature_scaling: bool = True,
    temperature_max_iter: int = 50,
    mixup_alpha: float = 0.0,
    calibration_method: str = "temperature",
    vector_max_iter: int = 100,
    label_smoothing: float = 0.0,
    dropout_p: float = 0.4,
    size_goal: float = 0.70,
    ece_delta_threshold: float = 0.03,
    hetero_ece_threshold: float = 0.02,
    max_accuracy_drop: float = 0.02,
    use_calibrated_ece: bool = True,
    enable_hetero_ft: bool = False,
    hetero_ft_epochs: int = 2,
    hetero_ft_mixup_alpha: float = 0.05,
    hetero_ft_label_smoothing: float = 0.05,
    hetero_ft_lr: float = 1e-4,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = _defaults_for_dataset(dataset)

    base = {
        "dataset": dataset,
        "seed": seed,
        "num_workers": num_workers,
        "epochs": defaults["epochs"] if epochs is None else epochs,
        "batch_size": defaults["batch_size"] if batch_size is None else batch_size,
        "lr": defaults["lr"] if lr is None else lr,
        "max_train_samples": defaults["max_train_samples"] if max_train_samples is None else max_train_samples,
        "max_test_samples": defaults["max_test_samples"] if max_test_samples is None else max_test_samples,
        "prune_amount": prune_amount,
        "val_split": val_split,
        "enable_temperature_scaling": enable_temperature_scaling,
        "temperature_max_iter": temperature_max_iter,
        "mixup_alpha": mixup_alpha,
        "calibration_method": calibration_method,
        "vector_max_iter": vector_max_iter,
        "label_smoothing": label_smoothing,
        "dropout_p": dropout_p,
        "output_dir": str(output_dir),
        "enable_hetero_ft": enable_hetero_ft,
        "hetero_ft_epochs": hetero_ft_epochs,
        "hetero_ft_mixup_alpha": hetero_ft_mixup_alpha,
        "hetero_ft_label_smoothing": hetero_ft_label_smoothing,
        "hetero_ft_lr": hetero_ft_lr,
    }

    fp32 = train_and_evaluate(TrainConfig(mode="fp32", **base))
    int8 = train_and_evaluate(TrainConfig(mode="uniform_int8", **base))
    hetero = train_and_evaluate(TrainConfig(mode="hetero_int4_fp16", **base))

    ece_key = "ece_calibrated" if use_calibrated_ece else "ece_raw"
    fp32_ece = fp32[ece_key] if fp32.get(ece_key) is not None else fp32["ece_raw"]
    hetero_ece = hetero[ece_key] if hetero.get(ece_key) is not None else hetero["ece_raw"]
    calibration_drop_vs_fp32 = hetero_ece - fp32_ece
    size_reduction = 1.0 - (hetero["model_size_bytes"] / max(fp32["model_size_bytes"], 1))
    fp32_acc = float(fp32["accuracy"])
    hetero_acc = float(hetero["accuracy"])
    accuracy_drop_vs_fp32 = max(0.0, fp32_acc - hetero_acc)
    size_ok = bool(size_reduction >= size_goal)
    calibration_ok = bool(abs(calibration_drop_vs_fp32) <= ece_delta_threshold)
    hetero_ece_ok = bool(hetero_ece <= hetero_ece_threshold)
    accuracy_ok = bool(accuracy_drop_vs_fp32 < max_accuracy_drop)
    overall_pass = bool(size_ok and calibration_ok and hetero_ece_ok and accuracy_ok)
    zone_counts = hetero["zone_parameter_count"]

    print("Model Summary (heterogeneous path)")
    print(
        "Compression Zone params:",
        zone_counts["compression_zone"],
        "| Reliability Zone params:",
        zone_counts["reliability_zone"],
        "| Total:",
        zone_counts["total"],
    )

    summary = {
        "dataset": dataset,
        "fp32": fp32,
        "uniform_int8": int8,
        "heterogeneous_int4_fp16": hetero,
        "ece_metric_used": ece_key,
        "fp32_ece_for_gate": fp32_ece,
        "heterogeneous_ece_for_gate": hetero_ece,
        "heterogeneous_ece_threshold": hetero_ece_threshold,
        "calibration_drop_vs_fp32": calibration_drop_vs_fp32,
        "accuracy_drop_vs_fp32": accuracy_drop_vs_fp32,
        "max_accuracy_drop": max_accuracy_drop,
        "size_goal": size_goal,
        "ece_delta_threshold": ece_delta_threshold,
        "meets_calibration_goal": calibration_ok,
        "meets_heterogeneous_ece_goal": hetero_ece_ok,
        "meets_accuracy_drop_goal": accuracy_ok,
        "size_reduction_vs_fp32": size_reduction,
        "meets_size_goal_gt_70pct": size_ok,
        "overall_pass": overall_pass,
    }

    summary_file = output_dir / dataset / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
