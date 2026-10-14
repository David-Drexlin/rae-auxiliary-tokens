#!/usr/bin/env bash
set -euo pipefail

register_input=native
usage() {
    echo "Usage: bash $(basename -- "$0") CONFIG.json OUTPUT_DIR [CHECKPOINT.pt]"
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi
package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
process_count="$("$python_bin" -c '
import json, sys
with open(sys.argv[1]) as handle:
    count = json.load(handle)["training"]["world_size"]
if type(count) is not int or count < 1:
    raise SystemExit("training.world_size must be a positive integer")
print(count)
' "$1")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--config "$1" --output "$2" --register-input "$register_input")
if [[ $# -eq 3 ]]; then
    args+=(--resume "$3")
fi
exec "$python_bin" -m torch.distributed.run --standalone \
    --nnodes=1 --nproc-per-node="$process_count" \
    "$package_root/code/train.py" "${args[@]}"
