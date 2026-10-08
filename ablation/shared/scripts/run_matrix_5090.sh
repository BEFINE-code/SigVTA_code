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
serial_ids=(C1 C2 T1 M1 M2 R1)

if [[ ! "${mode}" =~ ^(profile|core)$ ]]; then
    echo "Usage: $0 {profile|core}" >&2
    exit 2
fi
if [[ "${max_parallel}" != "1" ]]; then
    echo "MAX_PARALLEL must be 1; concurrent DataLoader runs stalled on the target RTX 5090 server" >&2
    exit 2
fi

run_serial() {
    local seed id
    for seed in "$@"; do
        for id in "${serial_ids[@]}"; do
            "${runner}" train "${id}" "${seed}"
        done
    done
}

if [[ "${mode}" == "profile" ]]; then
    for id in "${serial_ids[@]}"; do
        "${runner}" profile "${id}" 42
    done
    exit 0
fi

seeds=(42)

run_serial "${seeds[@]}"

if [[ "${mode}" == "core" ]]; then
    "${python_path}" "${control}" freeze-tests
fi
