#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting prior synthetic benchmark evaluation..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

CHECKPOINT="${1:-Finalmodels/FinalPRIMS.pt}"
# Trim stray leading/trailing whitespace (e.g. from a misquoted bsub argument).
CHECKPOINT="$(echo "$CHECKPOINT" | xargs)"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: '$CHECKPOINT'"
    echo "  cwd: $(pwd)"
    echo "Usage: $0 <checkpoint.pt> [max_nodes] [n_episodes] [font_size]"
    exit 1
fi

MAX_NODES="${2:-10}"
N_EPISODES="${3:-100}"
FONT_SIZE="${4:-12}"

echo "Using checkpoint: $CHECKPOINT"
echo "Using max_nodes:  $MAX_NODES"
echo "Using n_episodes: $N_EPISODES"
echo "Using font_size:  $FONT_SIZE"
echo ""

# Limit BLAS/OpenMP threads to 1 so numpy Cholesky doesn't spawn extra threads
# that compete with the PC ThreadPoolExecutor and other jobs on shared cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running prior synthetic evaluation (Meta-RCA vs baselines, mlp + gp, TabICL prior)..."
uv run python prim/evaluation/test_prior_eval.py \
    --checkpoint "$CHECKPOINT" \
    --max_nodes "$MAX_NODES" \
    --n_episodes "$N_EPISODES" \
    --fn_types gp mlp\
    --node_sizes 4 5 6 7 8 9 10 \
    --int_sizes 1 2 5 10 20 50 \
    --n_obs 100 \
    --compare_traversal \
    --no_counterfactual \
    --out_dir results

# uv run python prim/evaluation/test_prior_eval.py \
#     --checkpoint "$CHECKPOINT" \
#     --max_nodes "$MAX_NODES" \
#     --n_episodes "$N_EPISODES" \
#     --fn_types mlp gp \
#     --node_sizes 20 30 50 70 100 \
#     --int_sizes 1 2 5 10 20 50 \
#     --n_obs 100 \
#     --out_dir results

# echo ""
# echo "Generating plots..."
# uv run python prim/plots/plot_prior_results.py \
#     --csv results/prior_eval_results_large.csv \
#     --font_size "$FONT_SIZE"

echo ""
echo "Evaluation completed at: $(date)"
echo ""
echo "To replot without re-running experiments (e.g. to change font size):"
echo "  uv run python prim/plots/plot_prior_results.py --font_size 14"
