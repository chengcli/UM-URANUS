#!/usr/bin/env bash
#!/bin/bash
#SBATCH --partition=standard --time=59:00
#SBATCH -n 6
#SBATCH -o zgendatalog.txt
#SBATCH --account=chengcli1
set -euo pipefail
set -x

echo "Working Directory = $(pwd)"
echo "ID=$(id)"

# --------- Repo sync (manager node) ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -d .git ]]; then
  branch="$(git symbolic-ref --quiet --short HEAD || true)"
  if [[ -z "${branch}" ]]; then
    echo "ERROR: Detached HEAD. Checkout a branch before deploying." >&2
    exit 1
  fi

  # Abort if working tree is not clean
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Working tree not clean. Aborting."
    git status -sb
    exit 1
  fi

  echo "Repo:   ${SCRIPT_DIR}"
  echo "Branch: ${branch}"
  echo "Before: $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s)"

  git fetch --prune origin

  if ! git show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
    echo "ERROR: Remote branch origin/${branch} not found." >&2
    exit 1
  fi

  # Hard sync to remote
  git reset --hard "origin/${branch}"

  if [[ -f .gitmodules ]]; then
    git submodule update --init --recursive
  fi

  echo "After:  $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s)"
else
  echo "WARNING: Not a git repo. Skipping sync."
fi

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
TOTAL_PROCS=$((PROC_PER_NODE * NODES))

if [[ "${TOTAL_PROCS}" -ne 6 ]]; then
  echo "ERROR: run.sh is configured for exactly 6 total CPU ranks, got ${TOTAL_PROCS} (${NODES} nodes x ${PROC_PER_NODE} procs)." >&2
  exit 1
fi

# --------- Run ----------
  #./run_hydro_dry.py --config=jupiter_gcm_dry.yaml \
torchrun \
  --nnodes="${NODES}" \
  --nproc_per_node="${PROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  ./run_uranus.py --config=uranus_cpu_full_run.yaml --output-dir=output > /data/node${NODE_RANK}.log 2>&1
