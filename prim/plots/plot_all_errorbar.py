"""
Generate all error-bar figures with consistent styling.
One PDF/PNG per sweep — no combined panels.

Outputs
───────
  threenode_combined_obs / _int                 – Confounder/Mediator (PRIM-S)
  multi_rca_vary_obs / _int                     – Multi-RCA           (PRIM-S / Finetuning PRIM-S)
  errorbar_plot_all                             – Node sizes grid     (PRIM-L)
  errorbar_n<val>_<fn_type>                     – Node sizes, one representative
                                                   sweep value (PRIM-L). Sweep is
                                                   n_int (int-size sweep) when the
                                                   results CSV has an n_int column,
                                                   else n_obs (legacy CSVs).
  ablation_finetuning / ablation_finetuning_weak – Finetuning ablation (PRIM-S)

Colour convention (consistent, colorblind-safe, no default blue/orange):
  GP data  → GP_COLOR  (bluish-green, Okabe-Ito)
  NN data  → NN_COLOR  (reddish-purple, Okabe-Ito)
  Finetuned variants → same base colour, dashed line
  All other series   → Paul Tol "muted" palette (no blue/orange)

Usage
─────
  uv run python prim/plots/plot_all_errorbar.py
  uv run python prim/plots/plot_all_errorbar.py --font_size 14
"""

import argparse
import numpy as np
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"

# ── CLI ────────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--font_size", type=int, default=15)
_args = _p.parse_args()
FS = _args.font_size

plt.rcParams.update({
    "font.size": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS - 1,
})

# ── Colorblind-safe palette (no default matplotlib blue / orange) ─────────────
# GP / NN — used consistently across every figure
GP_COLOR = "#009E73"  # Okabe-Ito bluish-green
NN_COLOR = "#CC79A7"  # Okabe-Ito reddish-purple

# Paul Tol "muted" palette — well-separated in greyscale, colourblind-safe
_M = {
    "teal": "#44AA99",  # mid brightness
    "wine": "#882255",  # dark
    "rose": "#CC6677",  # medium
    "sand": "#DDCC77",  # light  (pair with bold marker for B&W)
    "purple": "#AA4499",  # medium-dark
    "olive": "#999933",  # medium
    "green": "#117733",  # dark
    "indigo": "#332288",  # very dark (deep indigo, not typical blue)
    "cyan": "#88CCEE",  # light, high contrast against the rest
    "vermillion":
    "#D55E00",  # Okabe-Ito vermillion, distinct from all of the above
}

ABLATION_COLORS = {
    "zero_shot": _M["teal"],
    "decoder_only": _M["rose"],
    "full_model": _M["wine"],
}
NODE_METHOD_COLORS = {
    "PRIM-L": _M["teal"],
    "Ranked Corr": _M["wine"],
    "Traversal (true)": _M["rose"],
    "Traversal (PC)": _M["olive"],
    "CIRCA (true)": _M["green"],
    "CIRCA (PC)": _M["sand"],
    r"$\varepsilon$-Diag": _M["purple"],
    "RCD": _M["indigo"],
    "Percentile95": _M["cyan"],
    "BARO": _M["vermillion"],
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _eb(ax, x, mean, lo, hi, **kw):
    mean, lo, hi = np.array(mean), np.array(lo), np.array(hi)
    yerr = [np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)]
    ax.errorbar(x, mean, yerr=yerr, **kw)


def _finish_ax(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, stem, tight=True):
    for ext in ("pdf", "png"):
        p = RESULTS_DIR / f"{stem}.{ext}"
        kw = dict(bbox_inches="tight") if tight else {}
        fig.savefig(p, dpi=200, **kw)
        print(f"Saved: {p}")
    plt.close(fig)


def _single_fig():
    return plt.subplots(figsize=(5.5, 4))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Confounder / Mediator  (PRIM-S)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confmed():
    # colour = data type (GP/NN),  linestyle = structure type
    STYLES = {
        ("confounder", "gp"):
        dict(color=GP_COLOR,
             linestyle="-",
             marker="o",
             label="PRIM-S Confounder (GP)"),
        ("confounder", "nn"):
        dict(color=NN_COLOR,
             linestyle="-",
             marker="o",
             label="PRIM-S Confounder (NN)"),
        ("mediator", "gp"):
        dict(color=GP_COLOR,
             linestyle="-.",
             marker="s",
             label="PRIM-S Mediator (GP)"),
        ("mediator", "nn"):
        dict(color=NN_COLOR,
             linestyle="-.",
             marker="s",
             label="PRIM-S Mediator (NN)"),
    }

    sweeps = [
        (
            "obs",
            r"$n_{\mathrm{obs}}$",
            RESULTS_DIR / "threenode_gp_results.csv",
            RESULTS_DIR / "threenode_nn_results.csv",
            "threenode_combined_obs",
        ),
        (
            "int",
            r"$n_{\mathrm{int}}$",
            RESULTS_DIR / "threenode_gp_results_vary_int.csv",
            RESULTS_DIR / "threenode_nn_results_vary_int.csv",
            "threenode_combined_int",
        ),
    ]

    for sweep, xlabel, csv_gp, csv_nn, stem in sweeps:
        fig, ax = _single_fig()
        df_gp, df_nn = pd.read_csv(csv_gp), pd.read_csv(csv_nn)
        x_ticks = sorted(
            set(df_gp[df_gp["sweep"] == sweep]["n"])
            | set(df_nn[df_nn["sweep"] == sweep]["n"]))
        chance_drawn = False
        for fn, df in [("gp", df_gp), ("nn", df_nn)]:
            sub = df[df["sweep"] == sweep]
            for series, grp in sub.groupby("series"):
                style = STYLES.get((series, fn))
                if style is None:
                    continue
                grp = grp.sort_values("n")
                _eb(
                    ax,
                    grp["n"],
                    grp["mean"],
                    grp["lo"],
                    grp["hi"],
                    linewidth=2,
                    markersize=5,
                    capsize=3,
                    elinewidth=1.0,
                    zorder=3,
                    **style,
                )
            if not chance_drawn:
                ax.axhline(1 / 3,
                           ls=":",
                           color="gray",
                           lw=0.9,
                           label="Chance (33%)")
                chance_drawn = True
        _finish_ax(ax, xlabel)
        ax.set_xscale("log")
        # Label every actual data point with its real value (20, 50, 100, ...)
        # instead of log-scale's default sparse powers-of-ten ticks (10^0,
        # 10^1, 10^2), and turn off the automatic minor ticks log-scale would
        # otherwise add between them.
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(v) for v in x_ticks])
        ax.minorticks_off()
        ax.legend(frameon=False, loc="lower right", fontsize=FS - 2)
        fig.tight_layout()
        _save(fig, stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Multi-RCA  (PRIM-S / Finetuning PRIM-S)
# ══════════════════════════════════════════════════════════════════════════════
# MAP@2 random baseline: E[AP@2] = (1/2)(2/n + 1/(n-1)), n=6 nodes, 2 RCs → 4/15
_MULTIRCA_CHANCE = 0.5 * (2 / 6 + 1 / 5)


def plot_multirca():
    df = pd.read_csv(RESULTS_DIR / "multi_rca_results.csv")

    # colour = data type (GP/NN),  linestyle = zero-shot vs finetuned
    STYLES = {
        "gp_zero":
        dict(color=GP_COLOR, linestyle="-", marker="o", label="PRIM-S (GP)"),
        "nn_zero":
        dict(color=NN_COLOR, linestyle="-", marker="o", label="PRIM-S (NN)"),
        "gp_ft":
        dict(color=GP_COLOR,
             linestyle="--",
             marker="s",
             label="Finetuning PRIM-S (GP)"),
        "nn_ft":
        dict(color=NN_COLOR,
             linestyle="--",
             marker="s",
             label="Finetuning PRIM-S (NN)"),
    }

    for sweep, xlabel, stem in [
        ("obs", r"$n_{\mathrm{obs}}$", "multi_rca_vary_obs"),
        ("int", r"$n_{\mathrm{int}}$", "multi_rca_vary_int"),
    ]:
        fig, ax = _single_fig()
        sub = df[df["sweep"] == sweep]
        for series, style in STYLES.items():
            grp = sub[sub["series"] == series].sort_values("n")
            if grp.empty:
                continue
            _eb(
                ax,
                grp["n"],
                grp["mean"],
                grp["lo"],
                grp["hi"],
                linewidth=2,
                markersize=5,
                capsize=3,
                elinewidth=1.0,
                zorder=3,
                **style,
            )
        ax.axhline(
            _MULTIRCA_CHANCE,
            ls=":",
            color="gray",
            lw=0.9,
            label=f"Chance ({_MULTIRCA_CHANCE:.0%})",
        )
        _finish_ax(ax, xlabel)
        ax.set_ylabel("MAP@2")
        ax.set_xscale("log")
        # Label every actual data point with its real value instead of
        # log-scale's default sparse powers-of-ten ticks.
        x_ticks = sorted(sub["n"].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(v) for v in x_ticks])
        ax.minorticks_off()
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        _save(fig, stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figures 4 & 5 – Different node sizes  (PRIM-L)
# ══════════════════════════════════════════════════════════════════════════════
_NODE_METHODS = {
    "PRIM-L": ("meta_rca", "meta_rca_lo", "meta_rca_hi"),
    "Ranked Corr": ("ranked_corr", "ranked_corr_lo", "ranked_corr_hi"),
    "Traversal (true)": (
        "anomaly_trav_true",
        "anomaly_trav_true_lo",
        "anomaly_trav_true_hi",
    ),
    "Traversal (PC)":
    ("anomaly_trav_pc", "anomaly_trav_pc_lo", "anomaly_trav_pc_hi"),
    "CIRCA (true)": ("circa_true", "circa_true_lo", "circa_true_hi"),
    "CIRCA (PC)": ("circa_pc", "circa_pc_lo", "circa_pc_hi"),
    r"$\varepsilon$-Diag":
    ("epsilon_diag", "epsilon_diag_lo", "epsilon_diag_hi"),
    "RCD": ("hier_rcd", "hier_rcd_lo", "hier_rcd_hi"),
    "Percentile95": ("percentile95", "percentile95_lo", "percentile95_hi"),
    "BARO": ("baro", "baro_lo", "baro_hi"),
}
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", ">", "h"]


def _draw_node_panel(ax, sub, jitter):
    for j, (name, (mc, lc, hc)) in enumerate(_NODE_METHODS.items()):
        if mc not in sub.columns:  # absent from older/legacy results CSVs
            continue
        xs = sub["n_nodes"].to_numpy() + jitter[j]
        means = sub[mc].to_numpy()
        lo = sub[lc].to_numpy()
        hi = sub[hc].to_numpy()
        _eb(
            ax,
            xs,
            means,
            lo,
            hi,
            label=name,
            color=NODE_METHOD_COLORS[name],
            marker=_MARKERS[j % len(_MARKERS)],
            linestyle="-",
            linewidth=2.2 if name == "PRIM-L" else 1.4,
            markersize=5,
            capsize=3,
            elinewidth=1.0,
            zorder=3,
        )


_FN_LABELS = {
    "gp": "Gaussian Process",
    "nn": "Neural Network",
    "mlp": "MLP",
    "tree": "Tree",
    "linear": "Linear",
}


def _sweep_col(df: pd.DataFrame) -> str:
    """n_int (current int-size sweep) if present, else n_obs (legacy CSVs)."""
    return "n_int" if "n_int" in df.columns else "n_obs"


def plot_nodesizes(csv_name="prior_eval_results_large.csv",
                   stem="errorbar_plot_all"):
    df = pd.read_csv(RESULTS_DIR / csv_name)
    sweep_col = _sweep_col(df)
    sweep_name = "int" if sweep_col == "n_int" else "obs"
    fn_types = sorted(df["fn_type"].unique())
    sweep_vals = sorted(df[sweep_col].unique())
    nodes = sorted(df["n_nodes"].unique())
    jitter = np.linspace(-0.45, 0.45, len(_NODE_METHODS))

    n_rows, n_cols = len(fn_types), len(sweep_vals)
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(4.5 * n_cols, 4.5 * n_rows),
                             sharey=False,
                             sharex=False)

    for r, fn in enumerate(fn_types):
        for c, val in enumerate(sweep_vals):
            ax = axes[r, c]
            sub = df[(df["fn_type"] == fn)
                     & (df[sweep_col] == val)].sort_values("n_nodes")
            _draw_node_panel(ax, sub, jitter)
            ax.set_xticks(nodes)
            ax.set_xticklabels([str(n) for n in nodes],
                               rotation=45,
                               ha="right")
            ax.set_xlim(nodes[0] - 1, nodes[-1] + 1)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(f"{_FN_LABELS.get(fn, fn)}\nRecall@1" if c ==
                          0 else "")
            if r == 0:
                ax.set_title(rf"$n_{{\mathrm{{{sweep_name}}}}}={val}$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 5 if len(handles) > 8 else 4
    fig.tight_layout()
    # tight_layout() already reserved margin_for_xlabels for the rotated
    # x-tick labels + "Number of nodes" xlabel — keep that, and ADD room for
    # the legend on top of it (its row count varies with len(_NODE_METHODS)),
    # rather than replacing it outright, which squeezed the bottom row's tick
    # labels into the legend once more methods made the legend taller.
    margin_for_xlabels = fig.subplotpars.bottom
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.canvas.draw()
    legend_height_frac = (legend.get_window_extent().transformed(
        fig.transFigure.inverted()).height)
    fig.subplots_adjust(bottom=margin_for_xlabels + legend_height_frac + 0.02)
    _save(fig, stem)


def plot_n100():
    """Single representative sweep-value panel per fn_type (wide, with legend
    on the first fn_type only). Sweep value defaults to n_int=20 for the
    current int-size sweep, or n_obs=100 for legacy CSVs."""
    _fs = FS + 3
    plt.rcParams.update({
        "font.size": _fs,
        "axes.labelsize": _fs,
        "xtick.labelsize": _fs,
        "ytick.labelsize": _fs,
        "legend.fontsize": _fs - 1,
    })
    df = pd.read_csv(RESULTS_DIR / "prior_eval_results_large.csv")
    sweep_col = _sweep_col(df)
    sweep_vals = sorted(df[sweep_col].unique())
    rep_val = 20 if sweep_col == "n_int" else 100
    if rep_val not in sweep_vals:
        rep_val = sweep_vals[len(sweep_vals) // 2]
    fn_types = sorted(df["fn_type"].unique())
    nodes = sorted(df["n_nodes"].unique())
    jitter = np.linspace(-0.45, 0.45, len(_NODE_METHODS))

    # First fn_type (with legend): wide figure, axes occupies left portion
    # Rest (no legend):            narrower figure, axes fills it — same
    # figure HEIGHT so axes heights are identical across PDFs; use height= in LaTeX
    adjust = dict(left=0.10, right=0.63, top=0.97, bottom=0.24)
    for i, fn in enumerate(fn_types):
        sub = df[(df["fn_type"] == fn)
                 & (df[sweep_col] == rep_val)].sort_values("n_nodes")
        fig, ax = plt.subplots(figsize=(9, 4))
        _draw_node_panel(ax, sub, jitter)
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes], rotation=45, ha="right")
        ax.set_xlim(nodes[0] - 1, nodes[-1] + 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Recall@1")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.subplots_adjust(**adjust)
        if i == 0:
            ax.legend(loc="center left",
                      bbox_to_anchor=(1.03, 0.5),
                      frameon=False,
                      ncol=1)
        _save(fig, f"errorbar_n{rep_val}_{fn}")
    plt.rcParams.update({
        "font.size": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS - 1,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Finetuning ablation  (PRIM-S)
# ══════════════════════════════════════════════════════════════════════════════
# Chance = mean(1/n) for n in 3..10
_ABLATION_CHANCE = float(np.mean([1 / n for n in range(3, 11)]))

_ABLATION_STYLES = {
    "zero_shot":
    dict(
        color=ABLATION_COLORS["zero_shot"],
        linestyle="-",
        marker="o",
        label="PRIM-S (zero-shot)",
    ),
    "decoder_only":
    dict(
        color=ABLATION_COLORS["decoder_only"],
        linestyle="--",
        marker="s",
        label="Finetuning PRIM-S (decoder only)",
    ),
    "full_model":
    dict(
        color=ABLATION_COLORS["full_model"],
        linestyle=":",
        marker="^",
        label="Finetuning PRIM-S (full model)",
    ),
}


def _plot_ablation(csv_path: Path, stem: str):
    df = pd.read_csv(csv_path)
    fig, ax = _single_fig()
    for condition, style in _ABLATION_STYLES.items():
        sub = df[df["condition"] == condition].sort_values("n_int")
        if sub.empty:
            continue
        _eb(
            ax,
            sub["n_int"],
            sub["mean"],
            sub["lo"],
            sub["hi"],
            linewidth=2,
            markersize=5,
            capsize=3,
            elinewidth=1.0,
            zorder=3,
            **style,
        )
    ax.axhline(
        _ABLATION_CHANCE,
        ls=":",
        color="gray",
        lw=0.9,
        label=f"Chance ({_ABLATION_CHANCE:.0%})",
    )
    _finish_ax(ax, r"$n_{\mathrm{int}}$")
    ax.set_xscale("log")
    # Label every actual data point with its real value instead of
    # log-scale's default sparse powers-of-ten ticks.
    x_ticks = sorted(df["n_int"].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="lower right", fontsize=FS - 2)
    fig.tight_layout()
    _save(fig, stem)


def plot_ablation():
    _plot_ablation(RESULTS_DIR / "ablation_finetuning.csv",
                   "ablation_finetuning")
    _plot_ablation(RESULTS_DIR / "ablation_finetuning_weak.csv",
                   "ablation_finetuning_weak")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_confmed()
    plot_multirca()
    plot_nodesizes()
    plot_nodesizes("prior_eval_results_10nodes.csv", "errorbar_plot_all_10")
    plot_n100()
    plot_ablation()
    print("All done.")
