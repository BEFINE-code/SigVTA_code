#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
CLASSICAL_PYTHON="${CLASSICAL_PYTHON:-/root/autodl-tmp/venvs/repro-baselines/bin/python}"
DEEP_PYTHON="${DEEP_PYTHON:-/root/autodl-tmp/venvs/repro-deep/bin/python}"
RUN_ROOT="$ROOT/runs/repro_baselines_final_l5_v3"
export PYTHONPATH="$ROOT"
export XDG_CACHE_HOME="/root/autodl-tmp/cache/repro-baselines"
export TORCH_HOME="/root/autodl-tmp/cache/torch"
export HF_HOME="/root/autodl-tmp/cache/huggingface"
export TMPDIR="/root/autodl-tmp/tmp/repro-baselines"
mkdir -p "$RUN_ROOT/logs" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$TMPDIR"

run_model() {
  local python="$1" model="$2" seed="$3"
  local metrics="$RUN_ROOT/$model/seed_$seed/metrics.json"
  local log="$RUN_ROOT/logs/${model}_seed_${seed}.log"
  if [[ -s "$metrics" ]]; then
    echo "skip completed model=$model seed=$seed"
    return
  fi
  echo "start model=$model seed=$seed"
  "$python" -m repro_baselines.cli run --root "$ROOT" --model "$model" --seed "$seed" \
    2>&1 | tee "$log"
  test -s "$metrics"
}

run_model "$CLASSICAL_PYTHON" sanity 42
for model in global dtw svm rf xgb two_stage_logistic two_stage_xgb; do
  run_model "$CLASSICAL_PYTHON" "$model" 42
done
for model in deepsets cnn bilstm transformer resnet18 tarnn tarnn_contrastive lnps_rnn; do
  run_model "$DEEP_PYTHON" "$model" 42
done
for seed in 43 44; do
  run_model "$CLASSICAL_PYTHON" xgb "$seed"
  run_model "$CLASSICAL_PYTHON" rf "$seed"
  run_model "$DEEP_PYTHON" tarnn "$seed"
  run_model "$DEEP_PYTHON" resnet18 "$seed"
done

OFFICIAL_ROOT="/root/autodl-tmp/third_party/SynSig2Vec"
PREPARED="${SYNSIG_PREPARED:-$RUN_ROOT/synsig2vec_official/prepared}"
if [[ -d "$OFFICIAL_ROOT" && -e "$PREPARED/manifest.json" ]]; then
  metrics="$RUN_ROOT/synsig2vec/seed_42/metrics.json"
  if [[ ! -s "$metrics" ]]; then
    "$DEEP_PYTHON" -m repro_baselines.cli run --root "$ROOT" --model synsig2vec --seed 42 \
      --official-root "$OFFICIAL_ROOT" --prepared "$PREPARED" \
      2>&1 | tee "$RUN_ROOT/logs/synsig2vec_seed_42.log"
    test -s "$metrics"
  fi
else
  echo "skip synsig2vec: set SYNSIG_PREPARED to a verified Train-only cache" >&2
fi

"$CLASSICAL_PYTHON" "$ROOT/scripts/repro_baselines/summarize.py"
