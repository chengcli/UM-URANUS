#!/bin/bash
#SBATCH --job-name=uranus
#SBATCH --account=chengcli1
#SBATCH --partition=spgpu,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---- User settings ----
SIF=/scratch/chengcli_root/chengcli0/kevinsh/ubuntu22.04-cuda11.8-py3.10-canoe_2026-03-03.sif
PORT=$((10000 + SLURM_JOB_ID % 50000))

# ---- Derive rendezvous info from Slurm allocation ----
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${PORT}"
export NNODES="${SLURM_JOB_NUM_NODES}"
export GPUS_PER_NODE=1
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}  GPUS_PER_NODE=${GPUS_PER_NODE}  WORLD_SIZE=${WORLD_SIZE}"
echo "NODELIST=${SLURM_JOB_NODELIST}"

# ---- Helpful NCCL/transport knobs (adjust to your cluster) ----
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=1                 # uncomment if no InfiniBand
# export NCCL_SOCKET_IFNAME=ib0            # or eno1/bond0/etc
# export GLOO_SOCKET_IFNAME=ib0

# ---- Launch one torchrun per node; torchrun spawns 2 procs on that node ----
srun --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 \
  singularity exec --nv "${SIF}" bash -lc '
    set -euo pipefail

    echo "Running on $(hostname)  SLURM_PROCID=${SLURM_PROCID}"
    export NODE_RANK=${SLURM_PROCID}

    source /opt/venv/bin/activate

    pip install snapy==2.1.4

    torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node='"${GPUS_PER_NODE}"' \
      --node_rank=${NODE_RANK} \
      --master_addr='"${MASTER_ADDR}"' \
      --master_port='"${MASTER_PORT}"' \
      ./run_uranus.py \
        --config=./uranus_smoke_gpu.yaml \
        --output-dir=./output
  '
