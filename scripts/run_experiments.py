from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from up_h_qat.runner import run_all_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UP-H-QAT comparative experiments")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=4000)
    parser.add_argument("--prune-amount", type=float, default=0.15)
    parser.add_argument("--ece-delta-threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_all_modes(
        dataset=args.dataset,
        output_dir=output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        prune_amount=args.prune_amount,
        ece_delta_threshold=args.ece_delta_threshold,
    )
    print("Experiment complete")
    print(summary)


if __name__ == "__main__":
    main()
