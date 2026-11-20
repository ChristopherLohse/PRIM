#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting finetuning ablation..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

CHECKPOINT="${1:-Finalmodels/FinalPRIMS.pt}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Usage: $0 <checkpoint.pt>"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

echo "=== Ablation 1: Sinusoidal equations (far OOD) ==="
uv run python prim/evaluation/ablation_finetuning.py \
    --checkpoint "$CHECKPOINT" \
    --max_nodes 10 \
    --seed 42 \
    --out_csv results/ablation_finetuning.csv \
    --out_pdf results/ablation_finetuning.pdf

echo ""
echo "=== Ablation 2: NN equations, weak intervention + high noise (near OOD) ==="
uv run python prim/evaluation/ablation_finetuning_weak.py \
    --checkpoint "$CHECKPOINT" \
    --max_nodes 10 \
    --seed 42 \
    --out_csv results/ablation_finetuning_weak.csv \
    --out_pdf results/ablation_finetuning_weak.pdf

echo ""
echo "Ablation completed at: $(date)"
echo "Results:"
echo "  results/ablation_finetuning.csv / .pdf      (sinusoidal)"
echo "  results/ablation_finetuning_weak.csv / .pdf  (weak NN)"
