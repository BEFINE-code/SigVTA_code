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
control="${script_dir}/ablation_control.py"
manifest="${project_root}/ablation/t2/test_checkpoint_manifest_v2.json"

[[ -f "${manifest}" ]] || {
    echo "Frozen ablation Test checkpoint manifest is missing: ${manifest}" >&2
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${project_root}"
for seed in 42; do
    for id in C1 C2 T1 M1 M2 R1; do
        entry_text="$("${python_path}" "${control}" begin-test \
            --id "${id}" --seed "${seed}" --manifest "${manifest}")"
        mapfile -t entry <<< "${entry_text}"
        if [[ "${entry[0]}" == "SKIP" ]]; then
            echo "Test already complete, skipping: ${id} seed ${seed}"
            continue
        fi
        config="${project_root}/${entry[1]}"
        output="${entry[2]}"
        checkpoint="${entry[3]}"
        "${python_path}" -u -m dvsrc.cli evaluate-t2-release \
            --config "${config}" \
            --checkpoint "${checkpoint}" \
            --split test \
            --workers "${workers}" \
            --output "${output}" \
            > >(tee -a "${output}/test_stdout.log") \
            2> >(tee -a "${output}/test_stderr.log" >&2)
        "${python_path}" "${control}" finalize-test --output "${output}"
    done
done
