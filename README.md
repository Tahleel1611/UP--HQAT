# UP-H-QAT: Uncertainty-Preserving Heterogeneous QAT

This repository implements a research-first pipeline for:

- FP32 baseline training
- Uniform INT8 QAT reference training
- Heterogeneous INT4-backbone + FP16-head QAT with MC Dropout in the reliability head

## Quick Start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run MNIST comparison (quick demo defaults):

```powershell
python scripts/run_experiments.py --dataset mnist --output-dir outputs
```

3. Run CIFAR-10 milestone:

```powershell
python scripts/run_experiments.py --dataset cifar10 --output-dir outputs
```

4. Run combined bottleneck + pruning gate test:

```powershell
python scripts/run_combined_test.py --dataset mnist --output-dir outputs_combined
```

The training pipeline uses a proper 3-way protocol:

- train split for weight updates
- validation split for temperature scaling (confidence calibration)
- test split for final reporting

Optional tuning flags:

- `--prune-amount 0.15`: structured filter pruning ratio for quantized modes
- `--ece-delta-threshold 0.03`: allowed calibration drift versus FP32
- `--hetero-ece-threshold 0.02` (combined test script): absolute heterogeneous ECE gate
- `--val-split 0.1` (combined test script): validation fraction from train split
- `--temperature-max-iter 50` (combined test script): L-BFGS steps for temperature fitting
- `--use-raw-ece` (combined test script): gate on raw ECE instead of calibrated ECE
- `--label-smoothing 0.1` (combined test script): confidence-softening target smoothing
- `--mixup-alpha 0.2` (combined test script): MixUp strength for smoother decision boundaries
- `--calibration-method vector` (combined test script): classwise vector scaling (`temperature|vector|none`)
- `--dropout-p 0.5` (combined test script): reliability head dropout probability
- `--dropout-sweep 0.4,0.5,0.6` (combined test script): run mini sweep and report best candidate
- `--vector-max-iter 100` (combined test script): optimization iterations for vector scaling
- `--max-accuracy-drop 0.02` (combined test script): acceptance gate for hetero vs FP32 accuracy drop

## Project Layout

- `src/up_h_qat/model.py`: Backbone and Reliability Zone with MC Dropout
- `src/up_h_qat/quantization.py`: INT4 fake-quant and quantization policy helpers
- `src/up_h_qat/train.py`: Training and evaluation logic for all modes
- `src/up_h_qat/metrics.py`: ECE and reliability diagram utilities
- `src/up_h_qat/runner.py`: Multi-mode experiment orchestration and artifact export
- `tests/`: Unit and smoke tests for policy, precision, and metrics behavior

## Current Scope

This phase focuses on the reliable ML engine (model + training + calibration analytics).
FastAPI and Streamlit integration are intentionally deferred to the next phase.
