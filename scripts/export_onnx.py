from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from up_h_qat.export import compare_pytorch_onnx_outputs, export_model_to_onnx
from up_h_qat.model import ModelConfig, SafetyCriticalCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export UP-H-QAT model to ONNX with graph and output checks")
    parser.add_argument("--checkpoint", required=True, help="Path to model state_dict (.pt)")
    parser.add_argument("--output", required=True, help="Path to output ONNX file")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout-p", type=float, default=0.4)
    parser.add_argument("--bottleneck-channels", type=int, default=16)
    parser.add_argument("--head-pool-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-dynamic-batch", action="store_true", default=False)
    parser.add_argument("--verify-atol", type=float, default=1e-4)
    parser.add_argument("--report", default="", help="Optional JSON report output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Current backbone has three MaxPool2d(2) layers, then AdaptiveAvgPool2d(2,2).
    # ONNX export path for adaptive pooling requires pre-adaptive feature map dims to be divisible by 2.
    pooled_h = args.height // 8
    pooled_w = args.width // 8
    if pooled_h < 2 or pooled_w < 2 or pooled_h % 2 != 0 or pooled_w % 2 != 0:
        raise ValueError(
            "Input size is not export-compatible for current backbone/adaptive pooling path. "
            f"Got height={args.height}, width={args.width}; after 3x pool => ({pooled_h}, {pooled_w}). "
            "Choose dimensions where floor(H/8) and floor(W/8) are even and >= 2 (e.g., 32x32)."
        )

    cfg = ModelConfig(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
        bottleneck_channels=args.bottleneck_channels,
        head_pool_size=args.head_pool_size,
    )
    model = SafetyCriticalCNN(cfg)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading checkpoint ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"Info: ignored extra checkpoint keys ({len(unexpected)}), typically from QAT: {unexpected[:5]}")

    sample_input = torch.randn(args.batch_size, args.in_channels, args.height, args.width, dtype=torch.float32)

    onnx_path = export_model_to_onnx(
        model=model,
        out_file=args.output,
        sample_input=sample_input,
        opset_version=args.opset,
        dynamic_batch=(not args.no_dynamic_batch),
        validate_graph=True,
    )

    comparison = compare_pytorch_onnx_outputs(
        model=model,
        onnx_path=onnx_path,
        sample_input=sample_input,
        atol=args.verify_atol,
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "dynamic_batch": bool(not args.no_dynamic_batch),
        "verify": comparison,
    }

    print("ONNX export complete")
    print(f"onnx_path={onnx_path}")
    print(
        f"verify_allclose={comparison['allclose']} | atol={comparison['atol']} "
        f"| max_abs_diff={comparison['max_abs_diff']:.6e}"
    )

    report_path = Path(args.report) if args.report else onnx_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not comparison["allclose"]:
        raise RuntimeError(
            "PyTorch/ONNX output mismatch exceeds tolerance. "
            f"max_abs_diff={comparison['max_abs_diff']:.6e}, atol={comparison['atol']}"
        )


if __name__ == "__main__":
    main()
