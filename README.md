# UP-H-QAT: Uncertainty-Preserving Heterogeneous Quantization-Aware Training

UP-H-QAT is a research and deployment pipeline for training compressed neural networks that retain reliable uncertainty estimates. The core idea is a zone-split quantization strategy: the convolutional backbone is trained under simulated INT4 constraints to maximise size reduction, while the classification head remains in FP16 and is equipped with Monte Carlo (MC) Dropout so that predictive confidence is preserved and measurable after quantization.

The pipeline produces three comparable checkpoints per run — FP32 baseline, uniform INT8 QAT reference, and the heterogeneous INT4/FP16 target — together with calibration metrics, ONNX export, edge-latency benchmarks, and a hard deployment gate that enforces size, calibration, and accuracy-drop targets before a model is considered production-ready.

## Table of Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
  - [Standard experiment run](#standard-experiment-run)
  - [Combined bottleneck and pruning test](#combined-bottleneck-and-pruning-test)
- [Calibration strategies](#calibration-strategies)
- [Deployment workflow](#deployment-workflow)
  - [ONNX export](#onnx-export)
  - [Edge benchmark](#edge-benchmark)
  - [Deployment gate check](#deployment-gate-check)
- [Configuration reference](#configuration-reference)
- [Project layout](#project-layout)
- [Testing](#testing)

---

## Architecture

The model (`SafetyCriticalCNN`) is divided into two zones:

**Compression Zone — `CompressionBackbone`**

Three convolutional layers (`conv1`, `conv2`, `conv3`) followed by a 1x1 bottleneck bridge (`bridge`) and an adaptive average pool. During QAT this zone receives INT4 fake-quant observers on both activations and weights, using a per-tensor affine scheme for activations and a per-channel symmetric scheme for weights.

**Reliability Zone — `ReliabilityHead`**

Two fully-connected layers (`fc1`, `fc_out`) that keep FP16 precision throughout quantized runs. The head supports MC Dropout inference: calling `predict_with_uncertainty` runs the forward pass a configurable number of times with dropout active and returns per-sample mean probability, variance, entropy, prediction, and confidence.

The `PrecisionPolicy` dataclass records which layer names map to which precision tier, and `apply_heterogeneous_qat_policy` applies the corresponding `QConfig` objects via `torch.ao.quantization.prepare_qat`.

Structured L2 filter pruning (`torch.nn.utils.prune.ln_structured`) can optionally be applied to backbone convolutional layers after QAT to further reduce parameter count.

---

## Requirements

- Python 3.10 or later
- PyTorch 2.5.1
- torchvision 0.20.1
- onnx 1.17.0 and onnxruntime 1.20.1 (for export and benchmarking)
- See `requirements.txt` for the full pinned dependency list

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Standard experiment run

`scripts/run_experiments.py` trains all three modes (FP32, uniform INT8, heterogeneous INT4/FP16) in sequence, writes per-mode metric JSON files and a `summary.json` to the output directory, and prints the final summary.

```bash
# MNIST quick validation (uses reduced sample counts by default)
python scripts/run_experiments.py --dataset mnist --output-dir outputs

# CIFAR-10 milestone run
python scripts/run_experiments.py --dataset cifar10 --output-dir outputs
```

The training protocol is strictly three-way:

- **train split** — weight updates via cross-entropy (with optional MixUp and label smoothing)
- **validation split** — post-training calibration fitting (temperature or vector scaling)
- **test split** — final accuracy and ECE reporting; never touched during training or calibration

### Combined bottleneck and pruning test

`scripts/run_combined_test.py` wraps the same `run_all_modes` call but adds an outer dropout sweep, a unified ECE-penalised accuracy score for candidate selection, and a printed gate summary after each run.

```bash
# Single dropout value
python scripts/run_combined_test.py --dataset mnist --output-dir outputs_combined

# Sweep over three dropout values and report the best candidate
python scripts/run_combined_test.py \
    --dataset cifar10 \
    --dropout-sweep 0.4,0.5,0.6 \
    --output-dir outputs_sweep
```

---

## Calibration strategies

Post-training calibration is fitted on the validation split and applied before ECE is reported. Three methods are available via `--calibration-method`:

| Method | Description |
|---|---|
| `temperature` | Scalar temperature scaling — a single parameter is fitted by grid search to minimise NLL. |
| `vector` | Class-wise vector scaling — per-class weight and bias parameters are fitted via L-BFGS with L2 regularisation. |
| `none` | No calibration; raw logits are used for ECE computation. |

ECE is computed with 15 equal-width confidence bins. Both raw ECE (pre-calibration) and calibrated ECE are recorded. The deployment gate operates on calibrated ECE by default; pass `--use-raw-ece` to gate on raw ECE instead.

---

## Deployment workflow

The full deployment sequence produces a trained checkpoint, an ONNX artefact, a latency/memory benchmark report, and a final gate decision.

### ONNX export

```bash
python scripts/export_onnx.py \
    --checkpoint outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/model_state.pt \
    --output deployment/model_final.onnx \
    --in-channels 3 --num-classes 10 \
    --height 32 --width 32 \
    --batch-size 1 \
    --report deployment/model_final.export_report.json
```

The script validates the ONNX graph with the ONNX checker and verifies that the maximum absolute difference between PyTorch and ONNX runtime outputs is within the tolerance set by `--verify-atol` (default `1e-4`). Dynamic batch axis is enabled by default; pass `--no-dynamic-batch` to disable.

The backbone topology requires that `floor(H/8)` and `floor(W/8)` are both even and at least 2. For example, 32x32 input satisfies this constraint; 28x28 MNIST input does not and must be padded or resized before export.

### Edge benchmark

```bash
python scripts/benchmark_edge.py \
    --model deployment/model_final.onnx \
    --batch-size 1 \
    --in-channels 3 --height 32 --width 32 \
    --iters 100 --warmup 10 \
    --verify-checkpoint outputs_final_prod/dropout_0.50/cifar10/hetero_int4_fp16/model_state.pt \
    --num-classes 10 \
    --output deployment/model_final.benchmark.json
```

Reports P50 and P99 latency in milliseconds and peak RSS memory in MB. When `--verify-checkpoint` is supplied the script also cross-checks ONNX runtime output against the PyTorch model for the same dummy input batch.

### Deployment gate check

```bash
python scripts/check_deployment_gates.py \
    --summary outputs_final_prod/dropout_0.50/cifar10/summary.json \
    --max-accuracy-drop 0.02
```

Reads `summary.json` produced by `run_all_modes`, enforces `overall_pass == true` and that the accuracy drop does not exceed `--max-accuracy-drop`, and exits with a non-zero status code if either condition fails. Intended for use as a CI step.

---

## Configuration reference

### `run_experiments.py`

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `mnist` | Dataset to use (`mnist` or `cifar10`). |
| `--output-dir` | `outputs` | Root directory for artefacts. |
| `--epochs` | dataset default | Training epochs (overrides per-dataset default). |
| `--batch-size` | dataset default | Mini-batch size. |
| `--lr` | dataset default | Initial learning rate. |
| `--num-workers` | `2` | DataLoader worker processes. |
| `--seed` | `42` | Global random seed. |
| `--max-train-samples` | `12000` | Cap on training samples per epoch. |
| `--max-test-samples` | `4000` | Cap on test samples. |
| `--prune-amount` | `0.15` | Structured filter pruning ratio applied to backbone convolutions after QAT. |
| `--ece-delta-threshold` | `0.03` | Maximum allowed calibration drift (heterogeneous ECE minus FP32 ECE) for the summary gate. |

### `run_combined_test.py` (additional flags)

| Flag | Default | Description |
|---|---|---|
| `--val-split` | `0.1` | Fraction of training data held out for calibration fitting. |
| `--label-smoothing` | `0.1` | Cross-entropy label smoothing factor (Phase A). |
| `--mixup-alpha` | `0.0` | MixUp interpolation strength; `0.0` disables MixUp. |
| `--calibration-method` | `temperature` | Calibration method (`temperature`, `vector`, or `none`). |
| `--temperature-max-iter` | `50` | L-BFGS iterations for temperature scaling. |
| `--vector-max-iter` | `100` | L-BFGS iterations for vector scaling. |
| `--dropout-p` | `0.5` | MC Dropout probability in the reliability head. |
| `--dropout-sweep` | `` | Comma-separated dropout values to sweep (e.g. `0.4,0.5,0.6`). |
| `--hetero-ece-threshold` | `0.02` | Absolute ECE gate for the heterogeneous model. |
| `--max-accuracy-drop` | `0.02` | Maximum acceptable accuracy drop versus FP32. |
| `--size-goal` | `0.70` | Minimum required size reduction versus FP32 (fraction). |
| `--use-raw-ece` | off | Gate on raw (uncalibrated) ECE instead of calibrated ECE. |
| `--enable-hetero-ft` | off | Enable Phase B: head-only recovery fine-tuning after joint QAT. |
| `--hetero-ft-epochs` | `2` | Recovery phase epochs. |
| `--hetero-ft-lr` | `1e-4` | Learning rate for the recovery phase. |
| `--hetero-ft-mixup-alpha` | `0.05` | MixUp alpha for the recovery phase. |
| `--hetero-ft-label-smoothing` | `0.05` | Label smoothing for the recovery phase. |

---

## Project layout

```
src/up_h_qat/
    model.py          -- SafetyCriticalCNN, CompressionBackbone, ReliabilityHead, zone parameter counter
    quantization.py   -- INT4 / INT8 QConfig factories, heterogeneous QAT policy, structured pruning
    train.py          -- TrainConfig, training loop, evaluation, calibration, MC Dropout inference
    metrics.py        -- ECE computation, reliability diagram and calibration-gap plots, TemperatureScaler, VectorScaler
    runner.py         -- run_all_modes orchestration: trains all three modes, computes gate metrics, writes summary.json
    data.py           -- Dataset loading helpers for MNIST and CIFAR-10
    export.py         -- ONNX export and PyTorch/ONNX output comparison utilities

scripts/
    run_experiments.py       -- Entry point for standard three-mode comparison runs
    run_combined_test.py     -- Entry point with dropout sweep, Pareto scoring, and gate reporting
    export_onnx.py           -- Export a trained checkpoint to ONNX with graph validation
    benchmark_edge.py        -- Measure ONNX runtime latency and memory on CPU
    check_deployment_gates.py -- CI gate enforcement from a summary.json

tests/
    test_metrics.py           -- ECE correctness and edge cases
    test_model_precision.py   -- Zone parameter counting and model forward pass shape checks
    test_quant_policy.py      -- Fake-quant placement after applying heterogeneous QAT policy
    test_temperature_scaling.py -- Temperature and vector scaler fitting behaviour

deployment/
    model_final.onnx              -- Exported production model (CIFAR-10 heterogeneous INT4/FP16)
    model_final.export_report.json -- ONNX export verification report
    model_final.benchmark.json    -- Edge latency and memory benchmark report
    MODEL_DATASHEET_FINAL.md      -- Model datasheet with training config, gate results, and reproducibility commands
```

---

## Testing

```bash
pytest
```

The test suite covers quantization policy correctness, zone parameter counting, ECE computation, and temperature/vector scaler fitting. Test paths and the `src` import root are configured in `pyproject.toml`.

---

## Results

All results are from the final production run on CIFAR-10 (20 000 train samples, 5 000 test samples, MC Dropout p=0.5, vector calibration).

### Architecture diagram

![UP-H-QAT Neural Hybrid Architecture](https://github.com/user-attachments/assets/56a6b851-a93c-4655-8728-d78b03f5535f)

### Model comparison

| Mode | Accuracy | Raw ECE | Calibrated ECE | Model size (bytes) | Size reduction vs FP32 | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 41.08 % | 0.1043 | 0.0162 | 410 088 | — | 113.3 |
| Uniform INT8 QAT | 41.40 % | 0.1174 | 0.0149 | 102 522 | 75.0 % | 88.7 |
| **Hetero INT4/FP16 (ours)** | **40.08 %** | **0.0732** | **0.0147** | **62 076** | **84.9 %** | 179.5 |

The heterogeneous model achieves the best calibrated ECE and the smallest footprint, at a cost of only 1.0 percentage-point accuracy relative to FP32.

### Deployment gate results

| Gate | Value | Target | Pass |
|---|---:|---:|:---:|
| Size reduction vs FP32 | 84.86 % | ≥ 70 % | ✅ |
| Calibration delta vs FP32 | −0.00145 | ≤ 0.03 | ✅ |
| Absolute calibrated ECE | 0.01471 | ≤ 0.02 | ✅ |
| Accuracy drop vs FP32 | 1.00 pp | < 2 pp | ✅ |
| **Overall** | — | — | **✅ PASS** |

### MC Dropout uncertainty (heterogeneous model)

| Metric | Value |
|---|---:|
| Mean predictive entropy | 1.7639 |
| Mean confidence | 0.3261 |

### Edge benchmark (ONNX, CPU, batch size 1)

| Metric | Value |
|---|---:|
| Latency P50 | 0.117 ms |
| Latency P99 | 0.223 ms |
| Mean latency | 0.124 ms |
| Peak RSS memory | 292.7 MB |
| Max abs diff (PyTorch vs ONNX) | 4.77 × 10⁻⁷ |
