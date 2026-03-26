#!/bin/bash
#SBATCH --job-name=uranus
#SBATCH --account=chengcli1
#SBATCH --partition=spgpu,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR=/gpfs/accounts/chengcli_root/chengcli0/kevinsh/UM-SUBNEPTUNE

cd "${PROJECT_DIR}"

if [ ! -d env ]; then
  echo "Missing virtual environment: ${PROJECT_DIR}/env"
  exit 1
fi

source env/bin/activate
hash -r

echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python --version
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'ngpu:', torch.cuda.device_count())"

mkdir -p output

python ./run_uranus.py \
  --config=./uranus_smoke_gpu.yaml \
  --output-dir=./output