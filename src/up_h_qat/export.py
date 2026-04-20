from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_model_to_onnx(
    model: torch.nn.Module,
    out_file: str | Path,
    sample_input: torch.Tensor,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    validate_graph: bool = True,
) -> Path:
    """Export a PyTorch model to ONNX with optional dynamic batch axis and topology check."""
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            }
        torch.onnx.export(
            model,
            sample_input,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

    if validate_graph:
        import onnx

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

    return out_path


def compare_pytorch_onnx_outputs(
    model: torch.nn.Module,
    onnx_path: str | Path,
    sample_input: torch.Tensor,
    atol: float = 1e-4,
) -> dict:
    """Run a numeric consistency check between PyTorch and ONNX Runtime outputs."""
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        torch_out = model(sample_input).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(["logits"], {"input": sample_input.detach().cpu().numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    mean_abs_diff = float(np.mean(np.abs(torch_out - ort_out)))
    allclose = bool(np.allclose(torch_out, ort_out, atol=atol))

    return {
        "allclose": allclose,
        "atol": float(atol),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }
