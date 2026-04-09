#!/bin/bash
# Run A/B test: Aer GPU (cuQuantum) vs Blackwell Custom Kernels
# Sets up LD_LIBRARY_PATH for cuQuantum shared libs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
SITE="$VENV/lib/python3.12/site-packages"

source "$VENV/bin/activate"

export LD_LIBRARY_PATH="$SITE/cuquantum/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SCRIPT_DIR/bwk/python:$SCRIPT_DIR:${PYTHONPATH:-}"

exec python3 "$SCRIPT_DIR/tests/ab_test.py" "$@"
