#!/bin/bash
#SBATCH --job-name=timing_benchmark
# NOTE: cd must come after SBATCH directives but before any commands
#SBATCH --output=logs/timing_benchmark_%j.out
#SBATCH --error=logs/timing_benchmark_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd "$(dirname "$0")/.."

echo "Starting inference timing benchmark..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

mkdir -p logs

CHECKPOINT="${1:-Finalmodels/FinalPRIML.pt}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Usage: sbatch launch_timing_benchmark.sh <checkpoint.pt>"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

echo "Running inference timing benchmark..."
uv run python prim/plots/benchmark_inference_time.py \
    --checkpoint "$CHECKPOINT" \
    --max_nodes 100 \
    --out_dir results \
    --no_counterfactual

echo ""
echo "Benchmark completed at: $(date)"
