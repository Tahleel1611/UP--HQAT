import torch

from up_h_qat.metrics import expected_calibration_error


def test_ece_zero_for_perfect_confident_predictions() -> None:
    probs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 0, 1])
    ece, _ = expected_calibration_error(probs, y, num_bins=10)
    assert abs(ece) < 1e-8
