# UM-URANUS

This workspace contains the Uranus driver [`run_uranus.py`](/home/chengcli/scix/workspace/UM-URANUS/run_uranus.py) built on snapy's mesh-level API.

## Requirements

- Python environment with `torch`, `yaml`, and `snapy` available
- `torchrun` on `PATH`
- For GPU runs: at least one CUDA-capable GPU visible to PyTorch

## Main Run

Run the default Uranus case:

```bash
python /home/chengcli/scix/workspace/UM-URANUS/run_uranus.py \
  -c /home/chengcli/scix/workspace/UM-URANUS/uranus.yaml
```

The default output directory is:

```bash
/home/chengcli/data
```

Override it explicitly if needed:

```bash
python /home/chengcli/scix/workspace/UM-URANUS/run_uranus.py \
  -c /home/chengcli/scix/workspace/UM-URANUS/uranus.yaml \
  --output-dir /path/to/output
```

Restart from an existing restart archive:

```bash
python /home/chengcli/scix/workspace/UM-URANUS/run_uranus.py \
  -c /home/chengcli/scix/workspace/UM-URANUS/uranus.yaml \
  --output-dir /home/chengcli/data \
  --restart-name uranus.final.restart
```

## CPU Smoke Test

A reduced CPU smoke case can be run with 6 processes using the temporary smoke config:

```bash
torchrun --nproc-per-node=6 \
  /home/chengcli/scix/workspace/UM-URANUS/run_uranus.py \
  -c /home/chengcli/scix/workspace/UM-URANUS/uranus_smoke.yaml \
  --output-dir /home/chengcli/scix/workspace/UM-URANUS/smoke_output_fix2
```

This exercises:

- mesh initialization
- distributed stepping across 6 cubed-sphere blocks
- neural heating evaluation
- restart and NetCDF output paths

## Single-GPU NCCL RT Test

The RT GPU case can be run on one GPU with 6 cubed-sphere blocks per process:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 \
  /home/chengcli/scix/workspace/UM-URANUS/run_uranus.py \
  -c /home/chengcli/scix/workspace/UM-URANUS/uranus_gpu.yaml \
  --output-dir /home/chengcli/scix/workspace/UM-URANUS/outputsuranusgpu
```

This should place all 6 local blocks on `cuda:0`.

## Validation

Quick syntax check:

```bash
python -m py_compile /home/chengcli/scix/workspace/UM-URANUS/run_uranus.py
```

GPU availability check:

```bash
nvidia-smi -L
```

## Commit Workflow

Review the worktree first:

```bash
git -C /home/chengcli/scix/workspace/UM-URANUS status --short
```

Stage only the files you intend to include:

```bash
git -C /home/chengcli/scix/workspace/UM-URANUS add run_uranus.py README.md
```

Use a detailed commit message with a short subject and multiple body paragraphs. Example:

```bash
git -C /home/chengcli/scix/workspace/UM-URANUS commit \
  -m "Refactor Uranus driver to mesh-based EOS initialization" \
  -m "Replace the single-block MeshBlock entry point in run_uranus.py with the mesh-level snapy Mesh/MeshOptions flow. Initialize through mesh.initialize(...), switch restart handling to mesh.initialize_from_restart(...), and drive time stepping and output at mesh scope." \
  -m "Rework the initial condition path into an explicit isothermal atmosphere setup and remove hard-coded thermodynamic constants. Derive the gas constant from the configured ideal-gas EOS and use EOS conversions to recover temperature for NN preprocessing." \
  -m "Keep the Uranus neural radiative forcing workflow, fix interpolation bugs found by smoke testing, and document the default output location and validation commands."
```

If you created temporary smoke configs or output directories for testing, do not include them in the commit unless you explicitly want them versioned.
