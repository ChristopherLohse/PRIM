#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting CausRCA baseline evaluation..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

echo ""
echo "=== Baselines (sub and full node mode, expert graph) ==="
uv run python prim/evaluation/test_causrca_baselines.py \
    --mode sub full \
    --graph expert

echo ""
echo "CausRCA baseline evaluation completed at: $(date)"
