#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash scripts/check_inputs.sh CONFIG.json"
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi
package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
exec "$python_bin" "$package_root/code/train.py" \
    --config "$1" --output . --check-inputs
