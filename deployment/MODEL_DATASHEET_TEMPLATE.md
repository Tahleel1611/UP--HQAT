# UP-H-QAT Model Datasheet

## 1. Model Identity
- Model Name: 
- Version: 
- Date: 
- Commit SHA: 
- Dataset: CIFAR-10
- Task: Image Classification

## 2. Architecture Summary
- Compression Zone: INT4-simulated backbone
- Reliability Zone: FP16 head
- H-FT Strategy: Enabled (Phase A + Phase B)
- Backbone Type: SafetyCriticalCNN
- Zone Parameter Counts:
  - Compression: 
  - Reliability: 
  - Total: 

## 3. Training & Calibration Configuration
- Joint QAT Epochs (Phase A): 
- Recovery Epochs (Phase B): 
- MixUp Alpha (Phase A): 
- Label Smoothing (Phase A): 
- MixUp Alpha (Phase B): 
- Label Smoothing (Phase B): 
- Recovery LR: 
- Calibration Method: 

## 4. Deployment Gates (Final)
| Gate | Metric | Value | Target | Pass |
|---|---:|---:|---:|---|
| Size | size_reduction_vs_fp32 |  | >= 0.70 |  |
| Calibration Delta | calibration_drop_vs_fp32 |  | <= 0.03 |  |
| Absolute ECE | heterogeneous_ece_for_gate |  | <= 0.02 |  |
| Accuracy Drop | accuracy_drop_vs_fp32 |  | < 0.02 |  |
| Overall | overall_pass |  | True |  |

## 5. Product Performance Specs (Edge)
- ONNX Model Path: 
- Runtime Provider: 
- Batch Size: 
- Latency P50 (ms): 
- Latency P99 (ms): 
- Mean Latency (ms): 
- Peak RSS Memory (MB): 
- Verification Epsilon: 
- Max Abs Diff (PyTorch vs ONNX): 
- Within Tolerance: 

## 6. Accuracy and Uncertainty
- FP32 Accuracy: 
- Heterogeneous Accuracy: 
- Accuracy Drop vs FP32: 
- Raw ECE: 
- Calibrated ECE: 
- MC Dropout Uncertainty (if tracked): 

## 7. Artifacts
- Summary JSON: 
- Heterogeneous Metrics JSON: 
- Checkpoint Path: 
- ONNX File: 
- ONNX Export Report: 
- Edge Benchmark Report: 

## 8. Reproducibility Commands
```bash
# Train final candidate
python scripts/run_combined_test.py --dataset cifar10 --epochs 4 --batch-size 128 --lr 5e-4 --max-train-samples 20000 --max-test-samples 5000 --prune-amount 0.0 --val-split 0.1 --mixup-alpha 0.2 --label-smoothing 0.15 --calibration-method vector --vector-max-iter 150 --dropout-sweep 0.5 --ece-delta-threshold 0.03 --hetero-ece-threshold 0.02 --max-accuracy-drop 0.02 --enable-hetero-ft --hetero-ft-epochs 3 --hetero-ft-mixup-alpha 0.05 --hetero-ft-label-smoothing 0.05 --hetero-ft-lr 1e-4 --output-dir outputs_final_prod

# Export ONNX
python scripts/export_onnx.py --checkpoint outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/model_state.pt --output deployment/model_final.onnx --in-channels 3 --num-classes 10 --height 32 --width 32 --batch-size 1

# Benchmark edge performance
python scripts/benchmark_edge.py --model deployment/model_final.onnx --batch-size 1 --in-channels 3 --height 32 --width 32 --iters 100 --warmup 10 --verify-checkpoint outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/model_state.pt --num-classes 10 --output deployment/model_final.benchmark.json
```

## 9. Known Constraints and Notes
- Current ONNX export path is validated for 32x32 inputs with this backbone/pooling topology.
- CI smoke gate uses relaxed thresholds for speed; deployment release should use strict thresholds.
- Simulated INT4 may map to INT8/operator-level kernels on certain hardware runtimes.
