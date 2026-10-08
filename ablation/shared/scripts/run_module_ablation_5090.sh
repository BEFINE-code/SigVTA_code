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
mode="${1:-}"
max_parallel="${MAX_PARALLEL:-1}"
runner="${script_dir}/run_one_5090.sh"
control="${script_dir}/ablation_control.py"
ids=(F1 R2 R3)
matrix_rel="${ABLATION_MATRIX:-ablation/t2/module_matrix_transfer.json}"

if [[ ! "${mode}" =~ ^(profile|core)$ ]]; then
    echo "Usage: $0 {profile|core}" >&2
    exit 2
fi
if [[ "${max_parallel}" != "1" ]]; then
    echo "MAX_PARALLEL must be 1; concurrent DataLoader runs stalled on the target RTX 5090 server" >&2
    exit 2
fi

export ABLATION_MATRIX="${matrix_rel}"

for id in "${ids[@]}"; do
    if [[ "${mode}" == "profile" ]]; then
        "${runner}" profile "${id}" 42
    else
        "${runner}" train "${id}" 42
    fi
done

if [[ "${mode}" == "core" ]]; then
    cd "${project_root}"
    manifest_name="test_checkpoint_manifest_modules_transfer_v1.json"
    if [[ "${matrix_rel}" == "ablation/t2/module_matrix.json" ]]; then
        manifest_name="test_checkpoint_manifest_modules_v1.json"
    fi
    "${python_path}" "${control}" freeze-tests \
        --output "ablation/t2/${manifest_name}"
fi
