#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting multi-root-cause MAP@2 evaluation..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

CHECKPOINT="${1:-Finalmodels/FinalPRIMS.pt}"
MAX_NODES="${2:-10}"
FONT_SIZE="${3:-12}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Usage: $0 <checkpoint.pt> [max_nodes] [font_size]"
    exit 1
fi

echo "Checkpoint : $CHECKPOINT"
echo "max_nodes  : $MAX_NODES"
echo "font_size  : $FONT_SIZE"
echo ""

# ── Zero-shot only ────────────────────────────────────────────────────────────
echo "=== Zero-shot evaluation ==="
uv run python prim/evaluation/test_multi_rca.py \
    --checkpoint "$CHECKPOINT" \
    --max_nodes  "$MAX_NODES"  \
    --seed 42 \
    --csv results/multi_rca_results_zeroshot.csv \
    --font_size "$FONT_SIZE" \
    --out_obs results/multi_rca_vary_obs_zeroshot.pdf \
    --out_int results/multi_rca_vary_int_zeroshot.pdf

# ── Finetune + evaluate both ──────────────────────────────────────────────────
echo ""
echo "=== Finetuned evaluation ==="
uv run python prim/evaluation/test_multi_rca.py \
    --checkpoint "$CHECKPOINT"  \
    --max_nodes  "$MAX_NODES"   \
    --seed 42 \
    --finetune \
    --ft_epochs 5 \
    --ft_output Finalmodels/multi_rc_finetuned.pt \
    --csv results/multi_rca_results.csv \
    --font_size "$FONT_SIZE" \
    --out_obs results/multi_rca_vary_obs.pdf \
    --out_int results/multi_rca_vary_int.pdf

echo ""
echo "Done at: $(date)"
echo "Outputs written to results/multi_rca_*.pdf"
echo ""
echo "To replot without re-running experiments (e.g. to change font size):"
echo "  uv run python prim/evaluation/test_multi_rca.py --plot_from_csv --csv results/multi_rca_results.csv --font_size 14"
