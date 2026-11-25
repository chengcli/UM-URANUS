#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python="$PWD/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Missing project environment. See AGENTS.md for setup." >&2
    exit 1
fi

smoke_dir=$(mktemp -d "$PWD/uranus_smoke_output.XXXXXX")
exec > >(tee "$smoke_dir/run.log") 2>&1
echo "Smoke test directory: $smoke_dir"

"$python" - "$smoke_dir" <<'PY'
import sys
from pathlib import Path

import snapy
import torch
import yaml

if not torch.cuda.is_available():
    sys.exit("No CUDA GPU visible. Run this script on your allocated GPU node.")
print(f"GPU: {torch.cuda.get_device_name(0)}")

config = yaml.safe_load(Path("uranus_gpu.yaml").read_text())
config["geometry"]["cells"].update(nx2=6, nx3=6)
config["integration"].update(nlim=10_000_000, ncycle_out=10)
config["problem"].update(
    log_cycle_interval=10,
    modelfile=str(Path(config["problem"]["modelfile"]).resolve(strict=True)),
)
Path(sys.argv[1], "uranus_smoke_gpu.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY

exec "$python" -m torch.distributed.run --standalone --nproc-per-node=1 \
    run_uranus.py --config "$smoke_dir/uranus_smoke_gpu.yaml" \
    --output-dir "$smoke_dir/results"
