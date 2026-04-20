import torch

from up_h_qat.metrics import (
    TemperatureScaler,
    VectorScaler,
    apply_temperature,
    apply_vector_scaling,
)


def test_temperature_scale_reduces_logit_magnitude_for_t_gt_1() -> None:
    logits = torch.tensor([[3.0, -3.0], [2.0, -2.0]], dtype=torch.float32)
    scaled = apply_temperature(logits, temperature=1.5)
    assert scaled.abs().max().item() < logits.abs().max().item()


def test_set_temperature_returns_positive_value() -> None:
    logits = torch.randn(64, 10)
    labels = torch.randint(0, 10, (64,))
    scaler = TemperatureScaler(init_temperature=1.5)
    t = scaler.set_temperature(logits, labels, max_iter=10)
    assert t > 0.0


def test_vector_scaler_fit_returns_classwise_parameters() -> None:
    logits = torch.randn(64, 10)
    labels = torch.randint(0, 10, (64,))
    scaler = VectorScaler(num_classes=10)
    params = scaler.fit(logits, labels, max_iter=10)
    assert len(params["W"]) == 10
    assert len(params["b"]) == 10


def test_apply_vector_scaling_keeps_shape() -> None:
    logits = torch.randn(8, 10)
    W = [1.0] * 10
    b = [0.0] * 10
    scaled = apply_vector_scaling(logits, W, b)
    assert scaled.shape == logits.shape
