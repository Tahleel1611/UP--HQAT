from up_h_qat.model import ModelConfig, SafetyCriticalCNN
from up_h_qat.quantization import apply_heterogeneous_qat_policy, has_fake_quant_in_backbone_only


def test_heterogeneous_policy_quantizes_backbone_only() -> None:
    model = SafetyCriticalCNN(ModelConfig(in_channels=1, num_classes=10))
    prepared = apply_heterogeneous_qat_policy(model)
    assert has_fake_quant_in_backbone_only(prepared)
