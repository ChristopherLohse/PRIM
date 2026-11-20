#!/bin/bash
# Launch meta-training for PRIM-S (small) or PRIM-L (large), auto-detecting
# single- vs multi-GPU. Node range and epoch count match Appendix D of the
# paper; both configurations use the same multi-GPU DDP script when more
# than one GPU is available (Appendix D trains both on multi-GPU).
#
# --prior tabicl: the prior described in Section 5.1 / Appendix B
# (prim/data/tabicl_rca_prior.py); the single-GPU script's own default
# ("scm", an older/simpler prior) would silently train a model the paper
# doesn't describe.
# --single_rc: per Appendix B.1, PRIM-S's base training uses |T| = 1
# (single-root-cause episodes only); without this flag, the tabicl prior's
# default config mixes in 20%/10% dual-/triple-fault episodes. PRIM-L's base
# training uses that default mixed distribution instead (multi-root-cause
# included), so --single_rc is only passed for the small model below.
cd "$(dirname "$0")/.."

MODEL="${1:?Usage: $0 <small|large> [num_epochs]}"

case "$MODEL" in
    small)
        MIN_NODES=2
        MAX_NODES=10
        DEFAULT_EPOCHS=40
        ;;
    large)
        MIN_NODES=17
        MAX_NODES=100
        DEFAULT_EPOCHS=160
        ;;
    *)
        echo "ERROR: unknown model '$MODEL' (expected 'small' or 'large')"
        echo "Usage: $0 <small|large> [num_epochs]"
        exit 1
        ;;
esac

NUM_EPOCHS="${2:-$DEFAULT_EPOCHS}"
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# PRIM-S only: see --single_rc comment above.
SINGLE_RC_FLAGS=()
if [ "$MODEL" = "small" ]; then
    SINGLE_RC_FLAGS=(--single_rc)
fi

echo "Model:      PRIM-$([ "$MODEL" = small ] && echo S || echo L) ($MODEL)"
echo "Nodes:      $MIN_NODES-$MAX_NODES"
echo "Epochs:     $NUM_EPOCHS"
echo "single_rc:  $([ "$MODEL" = small ] && echo yes || echo no)"
echo ""

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ No GPUs detected!"
    exit 1
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "✅ 1 GPU detected. Using single-GPU meta-training..."
    uv run python prim/training/train_meta_rca.py \
        --prior tabicl \
        --alarm_penalty 0.15 \
        "${SINGLE_RC_FLAGS[@]}" \
        --min_nodes $MIN_NODES --max_nodes $MAX_NODES --num_epochs $NUM_EPOCHS
else
    echo "✅ $NUM_GPUS GPUs detected. Using multi-GPU meta-training..."
    nvidia-smi --list-gpus
    echo ""
    uv run python -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS \
      prim/training/train_meta_rca_multigpu.py \
      --prior tabicl \
      --alarm_penalty 0.15 \
      "${SINGLE_RC_FLAGS[@]}" \
      --min_nodes $MIN_NODES --max_nodes $MAX_NODES --num_epochs $NUM_EPOCHS
fi

echo ""
echo "✓ Meta-training complete! ($MODEL)"
