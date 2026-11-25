# Agent Notes

## Python environment on DAIS

**Use the repository-local `.venv` for all Uranus checks and simulations.**
Run these commands from the repository root to create it on a fresh checkout.

```bash
uv venv --python /usr/bin/python3.11 .venv
uv pip install --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  'torch==2.10.0+cu128' 'snapy==2.8.7' 'kintera==2.4.2' 'pyharp==2.4.8' pyyaml
uv pip check --python .venv/bin/python
```

Use `.venv/bin/python` directly or activate it with `source .venv/bin/activate`.
Snapy 2.8.7 matches this branch's YAML schema and targets PyTorch 2.10.0.
Its compiled extension also needs the matching Kintera and PyHarp native
symbols. Keep all four pins when recreating the environment. Use the release
build metadata to confirm the PyTorch version. The CUDA 12.8 wheel is
compatible with the DAIS H200 nodes.
[Snapy build metadata](https://github.com/chengcli/snapy/blob/v2.8.7/pyproject.toml)
[PyTorch wheel releases](https://pytorch.org/get-started/previous-versions/)

## Single-GPU smoke test

Request one GPU with an explicit host memory limit, then launch the test on the
allocated node.

```bash
salloc --time=01:00:00 --gres=gpu:1 --mem=32G
srun --ntasks=1 bash run_gpu_smoke.sh
```

If you already opened a shell on the GPU node with `srun --pty bash`, run
`bash run_gpu_smoke.sh` directly. The script uses `.venv`, checks CUDA, and
uses one process and six cubed-sphere faces at `15 × 6 × 6` cells per face.
It caps the run at 10,000,000 cycles and reports progress every 10 cycles.
It uses the local `basemodel_12_19.pt` and
prints the directory containing the generated configuration and results.
