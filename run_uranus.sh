#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-${script_dir}/output}"
if [ "$#" -gt 0 ]; then shift; fi
export DEVICE=cuda
export BACKEND="${BACKEND:-ucx}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc-per-node=2 "${script_dir}/run_uranus.py" \
  --config "${script_dir}/uranus.yaml" --output-dir "${output_dir}" "$@"
