#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting CausRCA finetuning..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

echo "Running finetuning..."
uv run python prim/finetuning/finetune_causrca.py \
    --resume Finalmodels/FinalPRIML.pt\
    --output finetune_causrca_final.pt \
    --n_graphs 100 \
    --n_settings 4 \
    --max_nodes 100 \
    --lr 5e-6 \
    --n_queries 8 \
    --k_context_min 0 \
    --k_context_max 0 \
    --eval \
    --n_epochs 4 \
    --seed 42
echo ""
echo "Finetuning completed at: $(date)"
