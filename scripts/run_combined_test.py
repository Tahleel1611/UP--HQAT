from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from up_h_qat.runner import run_all_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run combined bottleneck+pruning UP-H-QAT verification")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--output-dir", default="outputs_combined")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--prune-amount", type=float, default=0.15)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--calibration-method", choices=["temperature", "vector", "none"], default="temperature")
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--dropout-sweep", default="")
    parser.add_argument("--size-goal", type=float, default=0.70)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--ece-delta-threshold", type=float, default=0.03)
    parser.add_argument("--hetero-ece-threshold", type=float, default=0.02)
    parser.add_argument("--temperature-max-iter", type=int, default=50)
    parser.add_argument("--vector-max-iter", type=int, default=100)
    parser.add_argument("--disable-temperature-scaling", action="store_true", default=False)
    parser.add_argument("--use-calibrated-ece", action="store_true", default=True)
    parser.add_argument("--use-raw-ece", action="store_true", default=False)
    # Heterogeneous Fine-Tuning parameters
    parser.add_argument("--enable-hetero-ft", action="store_true", default=False, help="Enable head-only recovery phase")
    parser.add_argument("--hetero-ft-epochs", type=int, default=2, help="Number of head-only recovery epochs")
    parser.add_argument("--hetero-ft-mixup-alpha", type=float, default=0.05, help="MixUp alpha for recovery phase")
    parser.add_argument("--hetero-ft-label-smoothing", type=float, default=0.05, help="Label smoothing for recovery phase")
    parser.add_argument("--hetero-ft-lr", type=float, default=1e-4, help="Learning rate for head-only recovery phase")
    return parser.parse_args()


def _parse_dropout_sweep(raw: str, fallback: float) -> list[float]:
    if not raw.strip():
        return [fallback]
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values or [fallback]


def _unified_score(accuracy: float, hetero_ece: float, ece_threshold: float = 0.02) -> float:
    """
    Unified scoring function for Pareto front optimization.
    Score = Accuracy - (penalty for missing ECE gate)
    Penalizes heavily (ECE - threshold) * 10 if ECE exceeds threshold.
    Rewards recovered accuracy.
    """
    ece_penalty = 10.0 * max(0.0, hetero_ece - ece_threshold)
    return accuracy - ece_penalty


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dropout_candidates = _parse_dropout_sweep(args.dropout_sweep, args.dropout_p)

    best_record = None
    for p in dropout_candidates:
        run_dir = output_dir / f"dropout_{p:.2f}"
        summary = run_all_modes(
            dataset=args.dataset,
            output_dir=run_dir,
            seed=args.seed,
            num_workers=args.num_workers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            prune_amount=args.prune_amount,
            val_split=args.val_split,
            enable_temperature_scaling=(not args.disable_temperature_scaling),
            temperature_max_iter=args.temperature_max_iter,
            mixup_alpha=args.mixup_alpha,
            calibration_method=args.calibration_method,
            vector_max_iter=args.vector_max_iter,
            label_smoothing=args.label_smoothing,
            dropout_p=p,
            size_goal=args.size_goal,
            ece_delta_threshold=args.ece_delta_threshold,
            hetero_ece_threshold=args.hetero_ece_threshold,
            max_accuracy_drop=args.max_accuracy_drop,
            use_calibrated_ece=(args.use_calibrated_ece and not args.use_raw_ece),
            enable_hetero_ft=args.enable_hetero_ft,
            hetero_ft_epochs=args.hetero_ft_epochs,
            hetero_ft_mixup_alpha=args.hetero_ft_mixup_alpha,
            hetero_ft_label_smoothing=args.hetero_ft_label_smoothing,
            hetero_ft_lr=args.hetero_ft_lr,
        )

        size_reduction = float(summary["size_reduction_vs_fp32"])
        hetero_ece = float(summary["heterogeneous_ece_for_gate"])
        ece_delta = float(summary["calibration_drop_vs_fp32"])
        ece_metric = str(summary["ece_metric_used"])
        fp32_acc = float(summary["fp32"]["accuracy"])
        hetero_acc = float(summary["heterogeneous_int4_fp16"]["accuracy"])
        accuracy_drop = max(0.0, fp32_acc - hetero_acc)

        size_ok = size_reduction >= args.size_goal
        ece_delta_ok = abs(ece_delta) <= args.ece_delta_threshold
        hetero_ece_ok = hetero_ece <= args.hetero_ece_threshold
        acc_drop_ok = accuracy_drop < args.max_accuracy_drop

        overall = size_ok and ece_delta_ok and hetero_ece_ok and acc_drop_ok
        print(f"Combined Gate Check (dropout={p:.2f})")
        print(f"size_reduction_vs_fp32={size_reduction:.4f} | target>={args.size_goal:.2f} | pass={size_ok}")
        print(
            f"calibration_delta_vs_fp32={ece_delta:.4f} | target<={args.ece_delta_threshold:.2f} | pass={ece_delta_ok}"
        )
        print(
            f"heterogeneous_ece[{ece_metric}]={hetero_ece:.4f} | target<={args.hetero_ece_threshold:.2f} | pass={hetero_ece_ok}"
        )
        print(f"accuracy_drop_vs_fp32={accuracy_drop:.4f} | target<{args.max_accuracy_drop:.2f} | pass={acc_drop_ok}")
        print(f"overall_pass={overall}")

        unified_score = _unified_score(hetero_acc, hetero_ece, args.hetero_ece_threshold)
        record = {
            "dropout_p": p,
            "overall": overall,
            "unified_score": unified_score,
            "hetero_ece": hetero_ece,
            "accuracy_drop": accuracy_drop,
            "hetero_accuracy": hetero_acc,
            "size_reduction": size_reduction,
            "ece_delta": ece_delta,
        }
        if best_record is None or record["unified_score"] > best_record["unified_score"]:
            best_record = record

    if best_record is not None and len(dropout_candidates) > 1:
        print("\n🏆 Best model by Unified Score (ECE-penalized Accuracy)")
        print(f"  Unified Score: {best_record['unified_score']:.4f}")
        print(f"  Accuracy: {best_record['hetero_accuracy']:.4f}")
        print(f"  ECE: {best_record['hetero_ece']:.4f}")
        print(f"  Accuracy Drop: {best_record['accuracy_drop']:.4f}")
        print(f"  Size Reduction: {best_record['size_reduction']:.4f}")
        print(f"  Dropout P: {best_record['dropout_p']:.2f}")


if __name__ == "__main__":
    main()

