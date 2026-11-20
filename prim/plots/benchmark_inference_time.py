"""
Benchmark inference time for Meta-RCA vs baselines across graph sizes.

Node sizes: 20, 30, 50, 80, 100
n_obs=100, n_int=20, 100 episodes per node size.
Uses random data — only timing matters.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import torch
from prim.model.prim_neural_process import PRIMNeuralProcess as PRIM

# ── Parameters ───────────────────────────────────────────────────────────────

NODE_SIZES = [20, 30, 50, 80, 100]
N_OBS = 100
N_INT = 20
N_EPISODES = 100
MAX_NODES = 100
_CLIP = 10.0

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PETSHOP_CODE = str(_PROJECT_ROOT / "petshop-root-cause-analysis-main" /
                    "code")

# ── Data helpers ──────────────────────────────────────────────────────────────


def _random_obs_int(n_nodes, max_nodes, rng):
    """Random normalized+padded arrays in model format."""
    obs_raw = rng.standard_normal((N_OBS, n_nodes)).astype(np.float32)
    int_raw = rng.standard_normal((N_INT, n_nodes)).astype(np.float32)
    obs_mean = obs_raw.mean(axis=0, keepdims=True)
    obs_std = obs_raw.std(axis=0, keepdims=True) + 1e-8
    scale = max_nodes / n_nodes
    obs_n = np.clip((obs_raw - obs_mean) / obs_std, -_CLIP, _CLIP) * scale
    int_n = np.clip((int_raw - obs_mean) / obs_std, -_CLIP, _CLIP) * scale
    obs_pad = np.zeros((N_OBS, max_nodes), dtype=np.float32)
    int_pad = np.zeros((N_INT, max_nodes), dtype=np.float32)
    obs_pad[:, :n_nodes] = obs_n
    int_pad[:, :n_nodes] = int_n
    return obs_pad, int_pad


def _to_baseline_df(data, node_names):
    import pandas as pd
    cols = pd.MultiIndex.from_tuples(
        [(name, "value", "raw") for name in node_names],
        names=["node", "metric", "statistic"],
    )
    return pd.DataFrame(data.astype(np.float64), columns=cols)


def _random_dag(n_nodes, node_names, rng):
    """Sparse random DAG used as proxy for the true graph."""
    g = nx.DiGraph()
    g.add_nodes_from(node_names)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.15:
                g.add_edge(node_names[i], node_names[j])
    return g


def _pc_dag(obs_norm, node_names, rng_seed):
    """PC causal discovery → DAG (mirrors test_prior_eval.py)."""
    from causallearn.search.ConstraintBased.PC import pc as pc_alg
    n = obs_norm.shape[1]
    rng = np.random.default_rng(rng_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc_alg(obs_norm,
                    alpha=0.05,
                    indep_test="fisherz",
                    show_progress=False,
                    verbose=False)
    G = cg.G.graph
    edges, undirected = [], []
    for i in range(n):
        for j in range(n):
            if G[i, j] == 1 and G[j, i] == -1:
                edges.append((node_names[i], node_names[j]))
    for i in range(n):
        for j in range(i + 1, n):
            if (G[i, j] == -1 and G[j, i] == -1) or (G[i, j] == 1
                                                     and G[j, i] == 1):
                undirected.append((i, j))
    for i, j in undirected:
        edges.append((node_names[i],
                      node_names[j]) if rng.random() < 0.5 else (
                          node_names[j], node_names[i]))
    if not edges:
        order = list(range(n))
        rng.shuffle(order)
        for a in range(n):
            for b in range(a + 1, n):
                edges.append((node_names[order[a]], node_names[order[b]]))
    g = nx.DiGraph(edges)
    for name in node_names:
        if name not in g:
            g.add_node(name)
    if not nx.is_directed_acyclic_graph(g):
        g = nx.DiGraph([(node_names[i], node_names[i + 1])
                        for i in range(n - 1)])
    return g


# ── CI helper ─────────────────────────────────────────────────────────────────


def _bootstrap_ci(values, n_boot=500, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)),
                      replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * alpha)), float(
        np.percentile(boot, 100 * (1 - alpha)))


# ── Timing routines ───────────────────────────────────────────────────────────


def _warmup_model(model, device, max_nodes):
    """Run a few dummy forward passes to trigger CUDA JIT / kernel compilation."""
    rng = np.random.default_rng(0)
    for n_nodes in [max_nodes // 2, max_nodes]:
        obs_pad, int_pad = _random_obs_int(n_nodes, max_nodes, rng)
        target_mask = np.zeros(max_nodes, dtype=np.float32)
        target_mask[0] = 1.0
        obs_t = torch.from_numpy(obs_pad).unsqueeze(0).to(device)
        int_t = torch.from_numpy(int_pad).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(obs_t, int_t, target_mask=mask_t, n_nodes=n_nodes)
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_model(model, device, max_nodes, n_nodes, n_episodes, rng):
    """Time Meta-RCA forward pass (normalize + inference + softmax)."""
    times = []
    for _ in range(n_episodes):
        obs_pad, int_pad = _random_obs_int(n_nodes, max_nodes, rng)
        symptom = int(rng.integers(0, n_nodes))
        target_mask = np.zeros(max_nodes, dtype=np.float32)
        target_mask[symptom] = 1.0

        obs_t = torch.from_numpy(obs_pad).unsqueeze(0).to(device)
        int_t = torch.from_numpy(int_pad).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(obs_t, int_t, target_mask=mask_t, n_nodes=n_nodes)
            pad_mask = torch.zeros_like(logits)
            if n_nodes < max_nodes:
                pad_mask[:, n_nodes:] = float("-inf")
            _ = torch.softmax(logits + pad_mask, dim=-1).cpu().numpy()
        times.append(time.perf_counter() - t0)
    return np.array(times)


def time_baseline(method_fn,
                  n_nodes,
                  n_episodes,
                  rng,
                  use_pc=False,
                  needs_graph=True):
    """Time one baseline method per episode.
    For PC-graph methods, includes PC discovery in the measured time.
    """
    node_names = [f"X{i + 1}" for i in range(n_nodes)]
    times = []

    for _ in range(n_episodes):
        obs_raw = rng.standard_normal((N_OBS, n_nodes))
        int_raw = rng.standard_normal((N_INT, n_nodes))
        obs_mean = obs_raw.mean(axis=0, keepdims=True)
        obs_std = obs_raw.std(axis=0, keepdims=True) + 1e-8
        obs_norm = (obs_raw - obs_mean) / obs_std
        int_norm = (int_raw - obs_mean) / obs_std

        normal_df = _to_baseline_df(obs_norm, node_names)
        abnormal_df = _to_baseline_df(int_norm, node_names)
        symptom_name = node_names[int(rng.integers(0, n_nodes))]

        t0 = time.perf_counter()

        if not needs_graph:
            baseline_graph = nx.DiGraph([(node_names[i], node_names[i + 1])
                                         for i in range(n_nodes - 1)])
        elif use_pc:
            baseline_graph = _pc_dag(obs_norm, node_names,
                                     int(rng.integers(0, 2**31))).reverse()
        else:
            baseline_graph = _random_dag(n_nodes, node_names, rng).reverse()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                method_fn(baseline_graph, symptom_name, "value", "raw",
                          normal_df, abnormal_df)
        except Exception:
            pass

        times.append(time.perf_counter() - t0)

    return np.array(times)


# ── Load baselines ─────────────────────────────────────────────────────────────


def _load_baselines():
    if _PETSHOP_CODE not in sys.path:
        sys.path.insert(0, _PETSHOP_CODE)

    methods = {}  # name -> (callable, use_pc, needs_graph)

    try:
        from ranked_correlation import make_ranked_correlation
        methods["ranked_corr"] = (
            make_ranked_correlation(filter_for_anomalous_metrics=False,
                                    root_cause_top_k=20),
            False,
            False,
        )
    except Exception as e:
        print(f"[warn] ranked_correlation: {e}")

    try:
        from counterfactual_attribution import make_counterfactual_attribution_method
        _cfn = make_counterfactual_attribution_method(n_jobs=-1)
        methods["counterfactual_true"] = (_cfn, False, True)
        methods["counterfactual_pc"] = (_cfn, True, True)
    except Exception as e:
        print(f"[warn] counterfactual_attribution: {e}")

    try:
        from baseline_anomaly_traversal import make_baseline_analyze_root_causes
        _anom = make_baseline_analyze_root_causes()
        methods["anomaly_trav_true"] = (_anom, False, True)
        methods["anomaly_trav_pc"] = (_anom, True, True)
    except Exception as e:
        print(f"[warn] baseline_anomaly_traversal: {e}")

    try:
        from circa import make_circa
        _circa = make_circa(root_cause_top_k=20)
        methods["circa_true"] = (_circa, False, True)
        methods["circa_pc"] = (_circa, True, True)
    except Exception as e:
        print(f"[warn] circa: {e}")

    try:
        from epsilon_diagnosis import make_epsilon_diagnosis
        methods["epsilon_diag"] = (make_epsilon_diagnosis(root_cause_top_k=20),
                                   False, False)
    except Exception:
        pass

    # try:
    #     from random_walk import make_random_walk
    #     methods["rnd_walk"] = (make_random_walk(root_cause_top_k=20), False, False)
    # except Exception:
    #     pass

    try:
        from hierarchical_rcd import make_hierarchical_rcd
        methods["hier_rcd"] = (make_hierarchical_rcd(root_cause_top_k=20),
                               False, False)
    except Exception:
        pass

    return methods


# ── Plot ──────────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

_METHOD_LABELS = {
    "Meta-RCA": "PRIM",
    "ranked_corr": "Corr",
    "counterfactual_true": "Counterfact (true)",
    "counterfactual_pc": "Counterfact (PC)",
    "anomaly_trav_true": "Traversal (true)",
    "anomaly_trav_pc": "Traversal (PC)",
    "circa_true": "CIRCA (true)",
    "circa_pc": "CIRCA (PC)",
    "epsilon_diag": r"$\varepsilon$-Diag",
    "hier_rcd": "rcd",
}


def plot_timing(timing_results, node_sizes, out_dir):
    plt.rcParams.update(_RCPARAMS)

    method_names = list(timing_results.keys())
    n_methods = len(method_names)
    cmap = cm.get_cmap("tab10", n_methods)
    colors = {name: cmap(i) for i, name in enumerate(method_names)}
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p", "<"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-."]
    style = {
        name: (markers[i % len(markers)], linestyles[i % len(linestyles)])
        for i, name in enumerate(method_names)
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns_arr = np.array(node_sizes, dtype=float)

    for name in method_names:
        rows = timing_results[name]
        means = np.array([r[0] for r in rows])
        lo = np.array([r[1] for r in rows])
        hi = np.array([r[2] for r in rows])
        mk, ls = style[name]
        ax.errorbar(
            ns_arr,
            means,
            yerr=[means - lo, hi - means],
            label=_METHOD_LABELS.get(name, name),
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

    ax.set_xticks(node_sizes)
    ax.set_xticklabels(node_sizes)
    ax.set_xlabel("Number of nodes", fontsize=10)
    ax.set_ylabel("Inference time (s)", fontsize=10)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.18),
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"timing_benchmark.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")

    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference time vs node size")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_nodes", type=int, default=MAX_NODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--no_counterfactual",
                        action="store_true",
                        help="Skip counterfactual methods (very slow)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print(f"Device: {device}  |  node sizes: {NODE_SIZES}  |  "
          f"n_obs={N_OBS}  n_int={N_INT}  episodes={N_EPISODES}")

    timing_results = {}  # method_name -> list of (mean, lo, hi)

    # ── Meta-RCA ──────────────────────────────────────────────────────────────
    if not args.skip_model:
        model = PRIM(
            embedding_size=160,
            num_attention_heads=8,
            mlp_hidden_size=512,
            num_layers=6,
            max_nodes=args.max_nodes,
            dropout=0.0,
        ).to(device)

        ckpt = args.checkpoint
        if ckpt is None:
            candidates = ["Finalmodels/FinalPRIML.pt"]
            for p in candidates:
                if os.path.exists(p):
                    ckpt = p
                    break

        if ckpt is None or not os.path.exists(ckpt):
            print(
                "No checkpoint found — pass --checkpoint <path>. Skipping model."
            )
        else:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            model.eval()
            _warmup_model(model, device, args.max_nodes)
            print(f"\nLoaded: {ckpt}")

            print("\n--- Meta-RCA ---")
            rows = []
            for n_nodes in NODE_SIZES:
                t = time_model(model, device, args.max_nodes, n_nodes,
                               N_EPISODES, rng)
                mean = float(t.mean())
                lo, hi = _bootstrap_ci(t)
                rows.append((mean, lo, hi))
                print(f"  n={n_nodes:>3}  mean={mean*1000:.2f} ms  "
                      f"[{lo*1000:.2f}, {hi*1000:.2f}]")
            timing_results["Meta-RCA"] = rows

    # ── Baselines ─────────────────────────────────────────────────────────────
    if not args.skip_baselines:
        baselines = _load_baselines()
        if args.no_counterfactual:
            for k in ["counterfactual_true", "counterfactual_pc"]:
                baselines.pop(k, None)
        print(f"\nLoaded baselines: {list(baselines.keys())}")

        for bl_name, (bl_fn, use_pc, needs_g) in baselines.items():
            print(f"\n--- {bl_name} ---")
            rows = []
            for n_nodes in NODE_SIZES:
                print(f"  n={n_nodes:>3} ({N_EPISODES} eps) ...", flush=True)
                t = time_baseline(bl_fn,
                                  n_nodes,
                                  N_EPISODES,
                                  rng,
                                  use_pc=use_pc,
                                  needs_graph=needs_g)
                mean = float(t.mean())
                lo, hi = _bootstrap_ci(t)
                rows.append((mean, lo, hi))
                print(
                    f"        mean={mean*1000:.1f} ms  [{lo*1000:.1f}, {hi*1000:.1f}]"
                )
            timing_results[bl_name] = rows

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not timing_results:
        print("No results to plot.")
        return

    plot_timing(timing_results, NODE_SIZES, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
