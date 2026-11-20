"""
Plot threenode (confounder + mediator) GP and NN results on the same axes.

Produces two plots:
  1. Recall@1 vs n_obs  (threenode_combined_obs.pdf)
  2. Recall@1 vs n_int  (threenode_combined_int.pdf)

Usage
─────
  uv run python prim/plots/plot_threenode_combined.py

  # Custom paths or font size:
  uv run python prim/plots/plot_threenode_combined.py \
      --csv_gp results/threenode_gp_results.csv \
      --csv_nn results/threenode_nn_results.csv \
      --csv_gp_int results/threenode_gp_results_vary_int.csv \
      --csv_nn_int results/threenode_nn_results_vary_int.csv \
      --font_size 15
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"

# ── colours: confounder = green family, mediator = purple family ──────────────
# GP: solid lines,  NN: dashed lines
_STYLES = {
    ("confounder", "gp"):
    dict(color="#1a9850", linestyle="-", marker="o", label="Confounder (GP)"),
    ("confounder", "nn"):
    dict(color="#1a9850", linestyle="--", marker="s", label="Confounder (NN)"),
    ("mediator", "gp"):
    dict(color="#9970ab", linestyle="-", marker="o", label="Mediator (GP)"),
    ("mediator", "nn"):
    dict(color="#9970ab", linestyle="--", marker="s", label="Mediator (NN)"),
}


def _load(csv_path: str, sweep: str) -> dict:
    """Return {series_key: {n, mean, lo, hi}} for the given sweep."""
    df = pd.read_csv(csv_path)
    sub = df[df["sweep"] == sweep]
    out = {}
    for key, grp in sub.groupby("series"):
        grp = grp.sort_values("n")
        out[key] = {
            "n": grp["n"].tolist(),
            "mean": grp["mean"].tolist(),
            "lo": grp["lo"].tolist(),
            "hi": grp["hi"].tolist(),
        }
    return out


def _draw(ax, data_gp: dict, data_nn: dict):
    chance_drawn = False
    for fn, data in [("gp", data_gp), ("nn", data_nn)]:
        for series, res in data.items():
            style = _STYLES.get((series, fn))
            if style is None or res is None:
                continue
            ns = np.array(res["n"])
            means = np.array(res["mean"])
            lo = np.array(res["lo"])
            hi = np.array(res["hi"])
            ax.errorbar(
                ns,
                means,
                yerr=[means - lo, hi - means],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                label=style["label"],
                linewidth=2.0,
                markersize=5,
                capsize=3,
                elinewidth=0.9,
                zorder=3,
            )
        if not chance_drawn:
            ax.axhline(1 / 3,
                       ls=":",
                       color="gray",
                       lw=0.9,
                       label="Chance (33%)")
            chance_drawn = True


def _finalise(ax, fig, xlabel, out_path, font_size):
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel("Recall@1", fontsize=font_size)
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log")
    ax.tick_params(labelsize=font_size)
    ax.legend(fontsize=font_size - 1, frameon=False, loc="lower right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Combined GP+NN threenode plot")
    parser.add_argument("--csv_gp",
                        default=str(RESULTS_DIR / "threenode_gp_results.csv"))
    parser.add_argument("--csv_nn",
                        default=str(RESULTS_DIR / "threenode_nn_results.csv"))
    parser.add_argument("--csv_gp_int",
                        default=str(RESULTS_DIR /
                                    "threenode_gp_results_vary_int.csv"))
    parser.add_argument("--csv_nn_int",
                        default=str(RESULTS_DIR /
                                    "threenode_nn_results_vary_int.csv"))
    parser.add_argument("--out_obs",
                        default=str(RESULTS_DIR /
                                    "threenode_combined_obs.pdf"))
    parser.add_argument("--out_int",
                        default=str(RESULTS_DIR /
                                    "threenode_combined_int.pdf"))
    parser.add_argument("--font_size", type=int, default=15)
    args = parser.parse_args()

    # ── vary n_obs ────────────────────────────────────────────────────────────
    gp_obs = _load(args.csv_gp, "obs")
    nn_obs = _load(args.csv_nn, "obs")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw(ax, gp_obs, nn_obs)
    _finalise(ax, fig, r"$n_{\mathrm{obs}}$", args.out_obs, args.font_size)

    # ── vary n_int ────────────────────────────────────────────────────────────
    gp_int = _load(args.csv_gp_int, "int")
    nn_int = _load(args.csv_nn_int, "int")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw(ax, gp_int, nn_int)
    _finalise(ax, fig, r"$n_{\mathrm{int}}$", args.out_int, args.font_size)


if __name__ == "__main__":
    main()
