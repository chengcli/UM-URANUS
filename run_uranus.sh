#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-output}"
DEVICE=cuda torchrun --standalone --nproc-per-node=2 run_uranus.py \
  --config uranus.yaml --output-dir "${output_dir}"
