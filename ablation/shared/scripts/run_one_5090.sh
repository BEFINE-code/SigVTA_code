#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${script_dir}"
while [[ ! -f "${project_root}/pyproject.toml" ]]; do
    parent="$(dirname -- "${project_root}")"
    [[ "${parent}" != "${project_root}" ]] || { echo "Project root not found" >&2; exit 2; }
    project_root="${parent}"
done
python_path="${PYTHON_PATH:-python}"
workers="${WORKERS_PER_RUN:-4}"
profile_steps="${PROFILE_STEPS:-100}"
action="${1:-}"
experiment_id="${2:-}"
seed="${3:-}"
control="${project_root}/ablation/shared/scripts/ablation_control.py"
matrix_rel="${ABLATION_MATRIX:-ablation/t2/matrix.json}"
matrix="${project_root}/${matrix_rel}"

if [[ ! "${action}" =~ ^(profile|train)$ ]]; then
    echo "Usage: $0 {profile|train} EXPERIMENT_ID SEED" >&2
    exit 2
fi
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Seed must be an integer" >&2
    exit 2
fi
for required in "${control}" "${matrix}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
command -v "${python_path}" >/dev/null 2>&1 || { echo "Python not found: ${python_path}" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "${gpu_memory_mib}" -lt 30000 ]]; then
    echo "A visible GPU with at least 30000 MiB is required; memory=${gpu_memory_mib}" >&2
    exit 1
fi

cd "${project_root}"
resolved_text="$("${python_path}" "${control}" resolve --id "${experiment_id}")"
mapfile -t resolved <<< "${resolved_text}"
config="${resolved[0]}"
slug="${resolved[1]}"
parent="${project_root}/ablation/t2/v3.1_no_transfer/t1/best_t1.pt"

if [[ "${action}" == "profile" ]]; then
    output="${project_root}/ablation/t2/v3/profiles/${slug}/seed_${seed}"
    if [[ -f "${output}/stage_b_profile.json" ]]; then
        echo "Profile already exists, skipping: ${output}/stage_b_profile.json"
        exit 0
    fi
    "${python_path}" "${control}" prepare \
        --id "${experiment_id}" --seed "${seed}" --output "${output}"
    mkdir -p "${output}"
    "${python_path}" -u -m dvsrc.cli profile-stage-b \
        --config "${config}" \
        --checkpoint "${parent}" \
        --seed "${seed}" \
        --workers "${workers}" \
        --steps "${profile_steps}" \
        --stage-epoch 2 \
        --output "${output}" \
        > >(tee -a "${output}/stdout.log") \
        2> >(tee -a "${output}/stderr.log" >&2)
    exit 0
fi

output="${project_root}/ablation/t2/v3/runs/${slug}/seed_${seed}"
"${python_path}" "${control}" prepare \
    --id "${experiment_id}" --seed "${seed}" --output "${output}"

if [[ -f "${output}/stage_b_status.json" && -f "${output}/best.pt" ]]; then
    if "${python_path}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("running") is False else 1)' "${output}/stage_b_status.json"; then
        "${python_path}" "${control}" finalize-validation --output "${output}"
        ln -sfn best.pt "${output}/best_composite.pt"
        echo "Validation run already complete: ${experiment_id} seed ${seed}"
        exit 0
    fi
fi

source_args=(--stage-a-checkpoint "${parent}")
for resume_name in last.pt last_b.pt last_a.pt; do
    if [[ -f "${output}/${resume_name}" ]]; then
        source_args=(--resume "${output}/${resume_name}")
        break
    fi
done

"${python_path}" -u -m dvsrc.cli train-stage-b \
    --config "${config}" \
    "${source_args[@]}" \
    --seed "${seed}" \
    --workers "${workers}" \
    --output "${output}" \
    > >(tee -a "${output}/stdout.log") \
    2> >(tee -a "${output}/stderr.log" >&2)

"${python_path}" "${control}" finalize-validation --output "${output}"
ln -sfn best.pt "${output}/best_composite.pt"
