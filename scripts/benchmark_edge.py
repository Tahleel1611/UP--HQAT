from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from pathlib import Path as _Path
import sys

ROOT = _Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from up_h_qat.model import ModelConfig, SafetyCriticalCNN


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference latency and memory")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--providers", default="CPUExecutionProvider", help="Comma-separated ORT providers")
    parser.add_argument("--output", default="", help="Optional JSON output report path")
    parser.add_argument("--verify-checkpoint", default="", help="Optional PyTorch state_dict to verify ONNX output")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout-p", type=float, default=0.4)
    parser.add_argument("--bottleneck-channels", type=int, default=16)
    parser.add_argument("--head-pool-size", type=int, default=2)
    parser.add_argument("--verify-epsilon", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for benchmark_edge.py. "
            "Install dependencies from requirements.txt before benchmarking."
        ) from exc

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    session = ort.InferenceSession(str(args.model), providers=providers)

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(args.batch_size, args.in_channels, args.height, args.width).astype(np.float32)

    verification = None
    if args.verify_checkpoint:
        ckpt = Path(args.verify_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"verify checkpoint not found: {ckpt}")
        model_cfg = ModelConfig(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            dropout_p=args.dropout_p,
            bottleneck_channels=args.bottleneck_channels,
            head_pool_size=args.head_pool_size,
        )
        model = SafetyCriticalCNN(model_cfg)
        state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        _missing, _unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()
        with torch.no_grad():
            torch_out = model(torch.from_numpy(dummy)).cpu().numpy()
        ort_out = session.run(None, {input_name: dummy})[0]
        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        verification = {
            "epsilon": float(args.verify_epsilon),
            "max_abs_diff": max_abs_diff,
            "within_tolerance": bool(max_abs_diff <= args.verify_epsilon),
        }
        if not verification["within_tolerance"]:
            raise RuntimeError(
                "ONNX/PyTorch mismatch in benchmark verification: "
                f"max_abs_diff={max_abs_diff:.6e} > epsilon={args.verify_epsilon:.6e}"
            )

    for _ in range(args.warmup):
        _ = session.run(None, {input_name: dummy})

    process = psutil.Process()
    latency_ms: list[float] = []
    rss_peak_bytes = process.memory_info().rss

    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        dt = (time.perf_counter() - t0) * 1000.0
        latency_ms.append(float(dt))
        rss_now = process.memory_info().rss
        if rss_now > rss_peak_bytes:
            rss_peak_bytes = rss_now

    p50 = _percentile(latency_ms, 50)
    p99 = _percentile(latency_ms, 99)
    mean_ms = float(np.mean(np.asarray(latency_ms, dtype=np.float64))) if latency_ms else 0.0

    report = {
        "model": str(Path(args.model).resolve()),
        "providers": providers,
        "batch_size": args.batch_size,
        "shape": [args.batch_size, args.in_channels, args.height, args.width],
        "iters": args.iters,
        "warmup": args.warmup,
        "latency_ms": {
            "p50": p50,
            "p99": p99,
            "mean": mean_ms,
        },
        "peak_rss_memory_mb": float(rss_peak_bytes / (1024.0 * 1024.0)),
        "verification": verification,
    }

    print("Edge benchmark complete")
    print(f"latency_p50_ms={p50:.3f}")
    print(f"latency_p99_ms={p99:.3f}")
    print(f"peak_rss_memory_mb={report['peak_rss_memory_mb']:.3f}")

    out_path = Path(args.output) if args.output else Path(args.model).with_suffix(".benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
