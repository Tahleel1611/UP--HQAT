# UP-H-QAT Model Datasheet (Final)

## 1. Model Identity
- Model Name: UP-H-QAT Heterogeneous INT4/FP16 (CIFAR-10)
- Version: Phase F.4 Golden + Track 1/3 Deployment Pass
- Date: 2026-03-29
- Dataset: CIFAR-10
- Task: Image Classification

## 2. Architecture Summary
- Compression Zone: INT4-simulated backbone (`backbone.*`)
- Reliability Zone: FP16 head (`reliability_head.*`)
- H-FT Strategy: Enabled (Phase A + Phase B)
- Backbone Type: SafetyCriticalCNN
- Zone Parameter Counts:
  - Compression: 95,312
  - Reliability: 7,210
  - Total: 102,522

## 3. Training and Calibration Configuration
- Joint QAT Epochs (Phase A): 4
- Recovery Epochs (Phase B): 3
- MixUp Alpha (Phase A): 0.2
- Label Smoothing (Phase A): 0.15
- MixUp Alpha (Phase B): 0.05
- Label Smoothing (Phase B): 0.05
- Recovery LR: 1e-4
- Calibration Method: Vector scaling
- Validation Split: 0.1

## 4. Deployment Gates (Final)
| Gate | Metric | Value | Target | Pass |
|---|---:|---:|---:|---|
| Size | size_reduction_vs_fp32 | 0.8486 | >= 0.70 | Yes |
| Calibration Delta | calibration_drop_vs_fp32 | -0.00145 | <= 0.03 | Yes |
| Absolute ECE | heterogeneous_ece_for_gate | 0.01471 | <= 0.02 | Yes |
| Accuracy Drop | accuracy_drop_vs_fp32 | 0.01000 | < 0.02 | Yes |
| Overall | overall_pass | true | true | Yes |

## 5. Product Performance Specs (Edge Benchmark)
- ONNX Model Path: `deployment/model_final.onnx`
- Runtime Provider: CPUExecutionProvider
- Batch Size: 1
- Latency P50 (ms): 0.1174
- Latency P99 (ms): 0.2226
- Mean Latency (ms): 0.1236
- Peak RSS Memory (MB): 292.6914
- Verification Epsilon: 1e-4
- Max Abs Diff (PyTorch vs ONNX): 4.7684e-07
- Within Tolerance: true

## 6. Accuracy and Uncertainty
- FP32 Accuracy: 0.4108
- Heterogeneous Accuracy: 0.4008
- Accuracy Drop vs FP32: 0.0100
- Raw ECE (heterogeneous): 0.07320
- Calibrated ECE (heterogeneous): 0.01471
- MC Dropout Mean Entropy: 1.76395
- MC Dropout Mean Confidence: 0.32609

## 7. Artifacts
- Summary JSON: `outputs_final_prod/dropout_0.50/cifar10/summary.json`
- Heterogeneous Metrics JSON: `outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/metrics.json`
- Heterogeneous Checkpoint: `outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/model_state.pt`
- ONNX File: `deployment/model_final.onnx`
- ONNX Export Report: `deployment/model_final.export_report.json`
- Edge Benchmark Report: `deployment/model_final.benchmark.json`

## 8. Reproducibility Commands
```bash
# 1) Train final candidate
python scripts/run_combined_test.py --dataset cifar10 --epochs 4 --batch-size 128 --lr 5e-4 --max-train-samples 20000 --max-test-samples 5000 --prune-amount 0.0 --val-split 0.1 --mixup-alpha 0.2 --label-smoothing 0.15 --calibration-method vector --vector-max-iter 150 --dropout-sweep 0.5 --ece-delta-threshold 0.03 --hetero-ece-threshold 0.02 --max-accuracy-drop 0.02 --enable-hetero-ft --hetero-ft-epochs 3 --hetero-ft-mixup-alpha 0.05 --hetero-ft-label-smoothing 0.05 --hetero-ft-lr 1e-4 --output-dir outputs_final_prod

# 2) Export ONNX
python scripts/export_onnx.py --checkpoint outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/model_state.pt --output deployment/model_final.onnx --in-channels 3 --num-classes 10 --height 32 --width 32 --batch-size 1 --report deployment/model_final.export_report.json

# 3) Benchmark edge
python scripts/benchmark_edge.py --model deployment/model_final.onnx --batch-size 1 --in-channels 3 --height 32 --width 32 --iters 100 --warmup 10 --verify-checkpoint outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/model_state.pt --num-classes 10 --output deployment/model_final.benchmark.json

# 4) Enforce hard gate
python scripts/check_deployment_gates.py --summary outputs_final_prod/dropout_0.50/cifar10/summary.json --max-accuracy-drop 0.02
```

## 9. Deployment Notes
- CI smoke workflow was added at `.github/workflows/deploy_gate.yml`.
- ONNX export includes dynamic batch axis and ONNX checker validation.
- Current export path is validated for 32x32 input topology in this model.
- QAT-specific fake-quant keys in checkpoints are handled via non-strict state load for deployment export.
