import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_parser = argparse.ArgumentParser(
    description="Plot prior eval results from CSV")
_parser.add_argument(
    "--csv",
    type=str,
    default=str(RESULTS_DIR / "prior_eval_results_large.csv"),
    help="Input CSV file",
)
_parser.add_argument(
    "--font_size",
    type=int,
    default=12,
    help="Font size for axis labels, ticks, and legend (default: 12)",
)
_args = _parser.parse_args()

FONT_SIZE = _args.font_size

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(_args.csv)

# ── method definitions ─────────────────────────────────────────────────────
methods = {
    "PRIM": ("meta_rca", "meta_rca_lo", "meta_rca_hi"),
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
    "rcd": ("hier_rcd", "hier_rcd_lo", "hier_rcd_hi"),
}

int_sizes = sorted(df["n_int"].unique())
fn_types = ["mlp", "gp"]
fn_labels = {"mlp": "MLP", "gp": "Gaussian Process"}
nodes = sorted(df["n_nodes"].unique())
n_methods = len(methods)

# ── one colour + marker + linestyle per method ────────────────────────────
cmap = cm.get_cmap("tab10", n_methods)
colors = {name: cmap(i) for i, name in enumerate(methods)}
markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
method_style = {
    name: (markers[i % len(markers)], linestyles[i % len(linestyles)])
    for i, name in enumerate(methods)
}

jitter = np.linspace(-0.18, 0.18, n_methods)


def get_data(fn, n_int):
    return (df[(df["fn_type"] == fn)
               & (df["n_int"] == n_int)].sort_values("n_nodes").reset_index(
                   drop=True))


def plot_panel(ax, fn, n_int):
    data = get_data(fn, n_int)
    for j, (name, (mc, lc, hc)) in enumerate(methods.items()):
        xs = data["n_nodes"].to_numpy() + jitter[j]
        means = data[mc].to_numpy()
        lo_err = np.clip(means - data[lc].to_numpy(), 0, None)
        hi_err = np.clip(data[hc].to_numpy() - means, 0, None)
        mk, ls = method_style[name]
        ax.errorbar(
            xs,
            means,
            yerr=np.array([lo_err, hi_err]),
            label=name,
            color=colors[name],
            marker=mk,
            linestyle=ls,
            linewidth=1.4,
            markersize=5,
            capsize=3,
            capthick=1.2,
            elinewidth=1.0,
            zorder=3,
        )
    ax.set_xticks(nodes)
    ax.set_xticklabels(nodes)
    ax.set_xlim(nodes[0] - 0.5, nodes[-1] + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of nodes", fontsize=FONT_SIZE)
    ax.set_ylabel("Recall@1", fontsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── 1. FULL GRID: rows = fn_type, cols = n_int ───────────────────────────
n_rows = len(fn_types)
n_cols = len(int_sizes)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4.5 * n_cols, 4.5 * n_rows),
    sharey=False,
    sharex=False,
)

for r, fn in enumerate(fn_types):
    for c, n_int in enumerate(int_sizes):
        ax = axes[r, c]
        plot_panel(ax, fn, n_int)
        if c == 0:
            ax.set_ylabel(f"{fn_labels[fn]}\nRecall@1", fontsize=FONT_SIZE)
        else:
            ax.set_ylabel("")
        if r == 0:
            ax.set_title(f"$n_{{\\mathrm{{int}}}}={n_int}$",
                         fontsize=FONT_SIZE)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize=FONT_SIZE - 1,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05),
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "errorbar_plot_all.pdf", bbox_inches="tight")
plt.savefig(RESULTS_DIR / "errorbar_plot_all.png",
            bbox_inches="tight",
            dpi=150)
print(f"Saved {RESULTS_DIR}/errorbar_plot_all.pdf / .png")
plt.close()

# ── 2. INDIVIDUAL n_int=20 PLOTS ─────────────────────────────────────────
_individual_n_int = 20 if 20 in int_sizes else int_sizes[-1]
for fn in fn_types:
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_panel(ax, fn, n_int=_individual_n_int)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=FONT_SIZE - 1,
        frameon=False,
    )
    plt.tight_layout()
    fname = RESULTS_DIR / f"errorbar_n{_individual_n_int}_{fn}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()

print("Done.")
