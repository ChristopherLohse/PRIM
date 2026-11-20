#!/bin/bash
# Installs everything needed for baseline comparisons (PetShop + RCAEval).
#
# `uv sync` only ever reconciles the environment to pyproject.toml + the
# lockfile, which already pins setuptools<81 (needed for `pkg_resources`,
# which setuptools>=81 dropped). sfr-pyrca can't be declared there too: its
# metadata claims scikit-learn>=0.24,<1.2, which conflicts with our own
# scikit-learn>=1.8.0 and makes uv's resolver reject the whole environment
# (confirmed by trying it), even though the actual code works fine with
# modern scikit-learn. So it's installed here with --no-deps instead, and
# --no-deps means uv never tracks it: any later `uv sync` -- or a *synced*
# `uv run` (its default behavior) -- silently uninstalls it again, breaking
# every pyrca-based baseline (hierarchical_rcd/RCD, CIRCA, epsilon-diagnosis).
# Run this script (not a bare `uv sync --extra baselines`) after pulling
# changes or otherwise re-syncing, and use `uv run --no-sync` (not a bare
# `uv run`) for anything that needs these baselines afterwards -- see
# launch_prior_eval.sh and launch_causrca_baselines.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --extra baselines
uv pip install --no-deps "sfr-pyrca @ git+https://github.com/salesforce/PyRCA.git"

echo "Baselines installed."
