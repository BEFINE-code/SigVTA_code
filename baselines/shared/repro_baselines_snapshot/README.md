# External baseline reproduction

This package evaluates external baselines on the final frozen L=5 T1/T2 benchmark.

## Safety rules

- The split digest must equal `ba64c09c53119be31f357c089987f706eb3266e4348d8398ff5d88e83dc0c48d`.
- T2 has five candidates and seven output classes; E0 seven-class accuracy is primary.
- Normalization and learned parameters use Train writers only.
- Validation selects thresholds. Test access requires an explicit final evaluation path.
- Audit metadata is never included in model features.

## Commands

```bash
CLASSICAL_PYTHON=/root/autodl-tmp/venvs/repro-baselines/bin/python
DEEP_PYTHON=/root/autodl-tmp/venvs/repro-deep/bin/python

$CLASSICAL_PYTHON -m repro_baselines.cli verify --root .
$CLASSICAL_PYTHON -m pytest -q tests/repro_baselines
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model sanity --seed 42
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model global --seed 42
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model dtw --seed 42
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model svm --seed 42
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model rf --seed 42
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model xgb --seed 42
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model two_stage_logistic --seed 42
$CLASSICAL_PYTHON -m repro_baselines.cli run --root . --model two_stage_xgb --seed 42

$DEEP_PYTHON -m repro_baselines.cli run --root . --model deepsets --seed 42
$DEEP_PYTHON -m repro_baselines.cli run --root . --model cnn --seed 42
$DEEP_PYTHON -m repro_baselines.cli run --root . --model bilstm --seed 42
$DEEP_PYTHON -m repro_baselines.cli run --root . --model transformer --seed 42
$DEEP_PYTHON -m repro_baselines.cli run --root . --model resnet18 --seed 42
$DEEP_PYTHON -m repro_baselines.cli run --root . --model tarnn --seed 42
$DEEP_PYTHON -m repro_baselines.cli run --root . --model tarnn_contrastive --seed 42
$DEEP_PYTHON -m repro_baselines.cli run --root . --model lnps_rnn --seed 42

$DEEP_PYTHON scripts/repro_baselines/prepare_synsig2vec_official.py \
  --root . \
  --official-root /root/autodl-tmp/third_party/SynSig2Vec \
  --output runs/repro_baselines_final_l5_v3/synsig2vec_official/prepared \
  --workers 6
$DEEP_PYTHON -m repro_baselines.cli run \
  --root . --model synsig2vec --seed 42 \
  --official-root /root/autodl-tmp/third_party/SynSig2Vec \
  --prepared runs/repro_baselines_final_l5_v3/synsig2vec_official/prepared

$CLASSICAL_PYTHON scripts/repro_baselines/summarize.py
```

Outputs are written below `runs/repro_baselines_final_l5_v3/`, which is ignored by Git.
The generated paper-facing table is `docs_database/BASELINE_REPRODUCTION_RESULTS.md`, with its machine-readable companion in `docs_database/baseline_reproduction_results.csv`.
The Chinese model catalog and interpretation guide is `docs_database/BASELINE_MODELS_AND_RESULTS_ZH.md`.

## Adaptation notes

- DTW is dependent multivariate DTW over x/y/pressure/speed with a 20% Sakoe-Chiba window.
- CNN, BiLSTM, and Transformer use the same fixed 128-point Train-normalized sequence representation and symmetric pair head.
- ResNet-18 uses the frozen `dynamic_rgb_v2` CSV renderer and local ImageNet weights; external PNG inputs are forbidden.
- TA-RNN is an adaptation using multivariate DTW alignment followed by the Siamese BiLSTM.
- TA-RNN contrastive keeps the same alignment but learns L2-normalized embeddings with a weighted contrastive loss.
- LNPS-RNN is an adaptation using a length-normalized 2D path, order-2 prefix signatures, and the Siamese BiLSTM.
- SynSig2Vec runs the official GPL-3.0 implementation from isolated commit `90b5a8363e57e7eb5bb82fb6168e2c2a28de624e`; this repository contains only the frozen-benchmark adapter and result metadata.
- A legacy SynSig2Vec preprocessing cache may be supplied with `SYNSIG_PREPARED` only when its manifest covers the identical 55 Train writers. Its original digest is retained in result metadata; final Validation/Test are always rerun.
- Generic pair scorers do not claim to reproduce the self-model's ABC heads. Their T2 type decision is an explicit Validation-only two-threshold adaptation over a model-specific scalar score; the candidate ranker itself is unchanged.
