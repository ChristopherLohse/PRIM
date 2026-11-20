#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting CausRCA digital-twin benchmark evaluation..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

echo "=== Zero-shot evaluation (k_context=0) ==="
uv run python prim/evaluation/test_causrca.py \
    --checkpoint Finalmodels/FinalPRIML.pt --mode full sub\

echo ""
# echo "=== Few-shot evaluation (k_context=4) ==="
# uv run python prim/evaluation/test_causrca.py \
#     --checkpoint context_meta_multigpu_epoch_40.pt \
#     --k_context 4 \
#     --mc_samples 20

# echo ""
echo "CausRCA evaluation completed at: $(date)"
