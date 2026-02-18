#!/usr/bin/env bash
set -euo pipefail
set -x

echo "Working Directory = $(pwd)"
echo "ID=$(id)"

# --------- Environment ----------
if [[ -f /opt/venv/bin/activate ]]; then
  source /opt/venv/bin/activate
else
  echo "ERROR: /opt/venv/bin/activate not found" >&2
  exit 1
fi

# --------- Torchrun args ----------
MASTER_PORT=29501
MASTER_ADDR="${1:?MASTER_ADDR required}"
NODE_RANK="${2:?NODE_RANK required}"
PROC_PER_NODE="${3:?PROC_PER_NODE required}"
NODES="${4:?NODES required}"

# --------- Run ----------
  #./run_hydro_dry.py --config=jupiter_gcm_dry.yaml \
torchrun \
  --nnodes="${NODES}" \
  --nproc_per_node="${PROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  ./shallow_splash.py --output_dir=/data/ > /data/node${NODE_RANK}.log 2>&1
