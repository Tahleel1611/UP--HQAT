from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast CI gate checker for deployment summary.json")
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: summary not found: {summary_path}")
        raise SystemExit(1)

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    fp32_acc = float(summary.get("fp32", {}).get("accuracy", 0.0))
    hetero_acc = float(summary.get("heterogeneous_int4_fp16", {}).get("accuracy", 0.0))
    accuracy_drop = float(summary.get("accuracy_drop_vs_fp32", max(0.0, fp32_acc - hetero_acc)))
    overall_pass = bool(summary.get("overall_pass", False))

    fail_reasons: list[str] = []
    if not overall_pass:
        fail_reasons.append("overall_pass is False")
    if accuracy_drop > args.max_accuracy_drop:
        fail_reasons.append(
            f"accuracy_drop_vs_fp32={accuracy_drop:.6f} exceeds threshold={args.max_accuracy_drop:.6f}"
        )

    print("Deployment Gate Check")
    print(f"summary={summary_path}")
    print(f"overall_pass={overall_pass}")
    print(f"accuracy_drop_vs_fp32={accuracy_drop:.6f} | max_allowed={args.max_accuracy_drop:.6f}")

    if fail_reasons:
        print("FAIL")
        for reason in fail_reasons:
            print(f"- {reason}")
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    main()
