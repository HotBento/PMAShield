#!/usr/bin/env bash

set -euo pipefail

# Use uv by default when available, fallback to python.
if command -v uv >/dev/null 2>&1; then
    PY_RUN=(uv run)
elif [[ -n "${PYTHON_BIN:-}" ]]; then
    PY_RUN=("$PYTHON_BIN")
else
    PY_RUN=(python)
fi

run_py() {
    if [[ ${#PY_RUN[@]} -eq 2 ]]; then
        "${PY_RUN[@]}" python "$@"
    else
        "${PY_RUN[@]}" "$@"
    fi
}

ensure_dir() {
    mkdir -p "$1"
}

model_safe() {
    local model="$1"
    printf '%s' "${model//\//_}"
}
