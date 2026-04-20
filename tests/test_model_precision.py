import torch

from up_h_qat.model import ModelConfig, SafetyCriticalCNN


def test_mc_dropout_is_stochastic_in_eval_mode() -> None:
    model = SafetyCriticalCNN(ModelConfig(in_channels=1, num_classes=10, dropout_p=0.5))
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    y1 = model(x, mc_dropout=True)
    y2 = model(x, mc_dropout=True)
    assert not torch.allclose(y1, y2)
