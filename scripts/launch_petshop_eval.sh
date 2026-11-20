#!/bin/bash
cd "$(dirname "$0")/.."

echo "Starting PetShop benchmark evaluation..."
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

echo "Running PetShop evaluation..."
uv run prim/evaluation/test_petshop.py --checkpoint Finalmodels/FinalPRIML.pt

echo ""
echo "PetShop evaluation completed at: $(date)"
 