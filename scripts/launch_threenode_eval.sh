#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting two-node synthetic benchmark evaluation..."
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
    echo "Usage: $0 <checkpoint.pt> [max_nodes] [n_episodes] [font_size]"
    exit 1
fi

MAX_NODES="${2:-10}"
N_EPISODES="${3:-200}"
FONT_SIZE="${4:-12}"

echo "Using checkpoint: $CHECKPOINT"
echo "Using max_nodes:  $MAX_NODES"
echo "Using n_episodes: $N_EPISODES"
echo "Using font_size:  $FONT_SIZE"
echo ""

# ── 3-node confounder/mediator (GP) — Figure 2 ────────────────────────────────
echo "Running 3-node GP evaluation (confounder/mediator)..."
uv run python prim/evaluation/test_threenode_synthetic.py \
    --checkpoint "$CHECKPOINT" \
    --max_nodes "$MAX_NODES" \
    --n_episodes "$N_EPISODES" \
    --settings confounder mediator \
    --fn_type_3node gp \
    --csv results/threenode_gp_results.csv \
    --font_size "$FONT_SIZE" \
    --vary_int \
    --out_3node results/threenode_eval_gp.pdf \
    --out_3node_vary_int results/threenode_eval_vary_int_gp.pdf

# ── 3-node confounder/mediator (NN) — Figure 2 ────────────────────────────────
echo ""
echo "Running 3-node NN evaluation (confounder/mediator)..."
uv run python prim/evaluation/test_threenode_synthetic.py \
    --checkpoint "$CHECKPOINT" \
    --max_nodes "$MAX_NODES" \
    --n_episodes "$N_EPISODES" \
    --settings confounder mediator \
    --fn_type_3node nn \
    --vary_int \
    --csv results/threenode_nn_results.csv \
    --font_size "$FONT_SIZE" \
    --out_3node results/threenode_eval_nn.pdf \
    --out_3node_vary_int results/threenode_eval_vary_int_nn.pdf

# ── Combine GP + NN into the paper's Figure 2 (obs and int sweeps) ────────────
echo ""
echo "Combining GP + NN results into Figure 2..."
uv run python prim/plots/plot_threenode_combined.py \
    --font_size "$FONT_SIZE" \
    --out_obs results/threenode_combined_obs.pdf \
    --out_int results/threenode_combined_int.pdf

echo ""
echo "Evaluation completed at: $(date)"
echo ""
echo "To replot without re-running experiments (e.g. to change font size):"
echo "  bash scripts/launch_threenode_eval.sh <checkpoint> <max_nodes> <n_episodes> <font_size>"
echo "  or directly:"
echo "  uv run python prim/plots/plot_threenode_combined.py --font_size 14 --out_obs results/threenode_combined_obs.pdf --out_int results/threenode_combined_int.pdf"
