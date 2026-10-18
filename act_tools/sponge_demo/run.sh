#!/usr/bin/env bash
set -euo pipefail
demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_path="${demo_dir}/../.venv/bin/python"
if [[ "$#" -eq 0 ]]; then
    set -- --viewer
fi
exec "$python_path" "$demo_dir/sponge_demo.py" --output "$demo_dir/output" "$@"
