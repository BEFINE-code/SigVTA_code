#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${MAX_PARALLEL:-1}" != "1" ]]; then
    echo "MAX_PARALLEL must be 1" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_PATH:-python}" -u "${study_dir}/run.py" "${@:-train}"
