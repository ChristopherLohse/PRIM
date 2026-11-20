"""
Evaluate Meta-RCA vs baseline methods on synthetic prior data.

Uses the TabICL-style prior (prim.data.tabicl_rca_prior), the same generator
available to training via ``prior=tabicl``:
  - generate_observational_and_interventional_data from tabicl_rca_prior
  - equation_mix forced to a single structural-equation class per sweep cell
    (see ``_EQUATION_MIX``) so each ``fn_type`` is a clean, isolated condition
  - Interventions (weight/shift/do/noise) are equation-agnostic (TabICL's
    gain-override mechanism), so they apply correctly regardless of equation type
  - Symptom = drawn by the prior's own aggregator/leaf-biased descendant logic

Baselines (from petshop-root-cause-analysis-main):
  - counterfactual_true/pc  : DoWhy Shapley (true / PC-discovered graph)
  - anomaly_trav_true/pc    : Anomaly traversal (true / PC-discovered graph)
  - ranked_corr             : Pearson correlation ranking (no graph)
  - epsilon_diag / circa / rnd_walk / hier_rcd  : PyRCA-based (if sfr-pyrca installed)

Sweeps:
  - n_nodes: 20, 30, 50, 70, 100
  - n_int (intervention sample size): 1, 2, 5, 10, 20, 50  (n_obs fixed at 100)
  - fn_type: mlp, gp
"""

import argparse
import os
import signal
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from prim.data.tabicl_rca_prior import (
    TabICLPriorConfig,
    generate_observational_and_interventional_data,
)
from prim.model.prim_neural_process import PRIMNeuralProcess
from causallearn.search.ConstraintBased.PC import pc as _pc_alg

# ─── Sweep parameters ────────────────────────────────────────────────────────

NODE_SIZES = [20, 30, 50, 70, 100]
INT_SIZES = [1, 2, 5, 10, 20, 50]
FN_TYPES = ["mlp", "gp"]
N_OBS = 100  # fixed observational sample size across the int-size sweep
N_EPISODES = 100  # episodes per (n_nodes, n_int) cell

# TabICLPriorConfig.equation_mix order is (mlp, tree, linear, gp); force a
# one-hot mixture so each fn_type sweep cell is a single, isolated equation
# class instead of TabICL's usual random mixture.
_EQUATION_MIX = {
    "mlp": (1.0, 0.0, 0.0, 0.0),
    "tree": (0.0, 1.0, 0.0, 0.0),
    "linear": (0.0, 0.0, 1.0, 0.0),
    "gp": (0.0, 0.0, 0.0, 1.0),
}

_CLIP = 10.0

# ─── Shared helpers ──────────────────────────────────────────────────────────


def _normalize_pad(obs, int_data, max_nodes, n_nodes):
    """z-score with obs stats, clip ±10, scale by max_nodes/n_nodes, pad.

    Clip is applied to z-scores BEFORE scaling — matches training exactly.
    (Scaling before clip would reduce the effective clip to ±10/scale.)
    """
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-8
    scale = max_nodes / n_nodes
    obs_n = np.clip((obs - obs_mean) / obs_std, -_CLIP, _CLIP) * scale
    int_n = np.clip((int_data - obs_mean) / obs_std, -_CLIP, _CLIP) * scale
    obs_pad = np.zeros((obs_n.shape[0], max_nodes), dtype=np.float32)
    int_pad = np.zeros((int_n.shape[0], max_nodes), dtype=np.float32)
    obs_pad[:, :n_nodes] = obs_n
    int_pad[:, :n_nodes] = int_n
    return obs_pad, int_pad


def _bootstrap_ci(correct, n_boot=500, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boot = rng.choice(correct, size=(n_boot, len(correct)),
                      replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * alpha)), float(
        np.percentile(boot, 100 * (1 - alpha)))


def _gen_episode_data(n_nodes, fn_type, n_obs, n_int, rng, max_retries=10):
    """Generate one episode via the TabICL-style prior.

    ``equation_mix`` is forced to a one-hot on ``fn_type`` (see
    ``_EQUATION_MIX``) so every episode in a sweep cell uses the same
    structural-equation class; graph sampling, noise, and interventions
    (weight/shift/do/noise, equation-agnostic via TabICL's gain-override) are
    otherwise TabICLPriorConfig's defaults, MMD-calibrated in
    prior_mmd_calibration.py / notebooks/prior_mmd_calibration.ipynb.
    """
    cfg = TabICLPriorConfig(equation_mix=_EQUATION_MIX[fn_type])
    for _ in range(max_retries):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            obs, int_data, dag, int_node, extra = (
                generate_observational_and_interventional_data(
                    n_nodes=n_nodes,
                    n_obs_samples=n_obs,
                    n_int_samples=n_int,
                    config=cfg,
                    seed=seed,
                    return_extra=True,
                ))
        except Exception:
            continue

        symptom = extra["symptom_node"]
        return obs, int_data, dag, int_node, symptom
    raise RuntimeError(
        f"Failed to generate episode after {max_retries} retries")


# ─── Episode cache ───────────────────────────────────────────────────────────


def _prepare_episodes(n_nodes, fn_type, n_obs, n_int, n_episodes, seed,
                      max_nodes):
    """Generate n_episodes and precompute all derived data for every method.

    Returns a list of dicts, each containing:
      obs_pad / int_pad / target_mask  — model inputs (padded, scaled)
      obs_norm / int_norm              — z-score only (baseline inputs)
      normal_df / abnormal_df          — DataFrame wrappers for baselines
      true_graph_rev                   — reversed true causal graph (string nodes)
      int_node / symptom / symptom_name / node_names
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    episodes = []
    for ep_i in range(n_episodes):
        if ep_i % 10 == 0:
            print(f"    ep {ep_i}/{n_episodes}...", flush=True)
        try:
            obs, int_data, dag, int_node, symptom = _gen_episode_data(
                n_nodes, fn_type, n_obs, n_int, rng)
        except Exception:
            continue

        node_names = [f"X{i + 1}" for i in range(n_nodes)]
        symptom_name = node_names[symptom]

        # Baseline: plain z-score (no scale factor)
        obs_mean = obs.mean(axis=0, keepdims=True)
        obs_std = obs.std(axis=0, keepdims=True) + 1e-8
        obs_norm = (obs - obs_mean) / obs_std
        int_norm = (int_data - obs_mean) / obs_std

        cols = pd.MultiIndex.from_tuples(
            [(n, "value", "raw") for n in node_names],
            names=["node", "metric", "statistic"],
        )
        normal_df = pd.DataFrame(obs_norm.astype(np.float64), columns=cols)
        abnormal_df = pd.DataFrame(int_norm.astype(np.float64), columns=cols)

        # True graph reversed (string node labels)
        causal_graph = nx.relabel_nodes(
            dag, {i: f"X{i + 1}"
                  for i in range(n_nodes)})
        if not nx.is_directed_acyclic_graph(causal_graph):
            causal_graph = nx.DiGraph([(node_names[i], node_names[i + 1])
                                       for i in range(n_nodes - 1)])
        true_graph_rev = causal_graph.reverse()

        # Model: padded + scaled
        obs_pad, int_pad = _normalize_pad(obs, int_data, max_nodes, n_nodes)
        target_mask = np.zeros(max_nodes, dtype=np.float32)
        target_mask[symptom] = 1.0

        episodes.append(
            dict(
                obs_pad=obs_pad,
                int_pad=int_pad,
                target_mask=target_mask,
                obs_norm=obs_norm,
                int_norm=int_norm,
                normal_df=normal_df,
                abnormal_df=abnormal_df,
                true_graph_rev=true_graph_rev,
                int_node=int_node,
                symptom=symptom,
                symptom_name=symptom_name,
                node_names=node_names,
            ))

    return episodes


def _parse_baseline_result(results, ep, n_nodes, rng_fallback):
    """Extract predicted node index from baseline results list."""
    if not results:
        candidates = [i for i in range(n_nodes) if i != ep["symptom"]]
        return int(rng_fallback.choice(candidates)) if candidates else 0
    results_sorted = sorted(results, key=lambda x: x.score, reverse=True)
    pred_name = results_sorted[0].node
    if pred_name in ep["node_names"]:
        return ep["node_names"].index(pred_name)
    candidates = [i for i in range(n_nodes) if i != ep["symptom"]]
    return int(rng_fallback.choice(candidates)) if candidates else 0


# ─── Unified sweep ────────────────────────────────────────────────────────────


def run_all_sweeps(
    model,
    device,
    max_nodes,
    fn_type,
    node_sizes,
    int_sizes,
    n_obs,
    n_episodes,
    base_seed,
    baseline_methods,
):
    """Evaluate model + all baselines on shared episode data (fully sequential)."""
    if model is not None:
        model.eval()

    graphfree_methods = {n: v for n, v in baseline_methods.items() if not v[2]}
    true_graph_methods = {
        n: v
        for n, v in baseline_methods.items() if v[2] and v[1]
    }
    pc_methods = {
        n: v
        for n, v in baseline_methods.items() if v[2] and not v[1]
    }

    model_results = {}
    bl_results = {name: {} for name in baseline_methods}

    for n_nodes in node_sizes:
        seed = base_seed + n_nodes * 1000
        model_rows = []
        bl_rows = {name: [] for name in baseline_methods}

        for n_int in int_sizes:
            print(
                f"\n  [generate] {fn_type} n={n_nodes:>3} n_int={n_int:>4} "
                f"({n_episodes} eps)...",
                flush=True,
            )
            episodes = _prepare_episodes(n_nodes, fn_type, n_obs, n_int,
                                         n_episodes, seed + n_int, max_nodes)
            rng_fb = np.random.default_rng(seed + n_int + 99)

            def dummy_chain(ep, n_nodes=n_nodes):
                return nx.DiGraph([(ep["node_names"][i],
                                    ep["node_names"][i + 1])
                                   for i in range(n_nodes - 1)])

            # ── Model ─────────────────────────────────────────────────────────
            if model is not None:
                correct = []
                for ep in episodes:
                    obs_t = torch.from_numpy(
                        ep["obs_pad"]).unsqueeze(0).to(device)
                    int_t = torch.from_numpy(
                        ep["int_pad"]).unsqueeze(0).to(device)
                    mask_t = torch.from_numpy(
                        ep["target_mask"]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(obs_t,
                                       int_t,
                                       target_mask=mask_t,
                                       n_nodes=n_nodes)
                        pad_mask = torch.zeros_like(logits)
                        if n_nodes < max_nodes:
                            pad_mask[:, n_nodes:] = float("-inf")
                        probs = (torch.softmax(logits + pad_mask,
                                               dim=-1).cpu().numpy()[0])
                    correct.append(
                        int(np.argmax(probs[:n_nodes])) == ep["int_node"])
                correct = np.array(correct, dtype=bool)
                mean = float(correct.mean()) if len(correct) else 0.0
                lo, hi = _bootstrap_ci(correct) if len(correct) else (0.0, 0.0)
                model_rows.append((mean, lo, hi))
                print(
                    f"  [Meta-RCA] {fn_type} n={n_nodes:>3} n_int={n_int:>4}  "
                    f"acc={mean:.3f}  [{lo:.3f}, {hi:.3f}]",
                    flush=True,
                )

            # ── Graph-free baselines ──────────────────────────────────────────
            for bl_name, (bl_fn, _, _) in graphfree_methods.items():
                correct = []
                for ep in episodes:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = bl_fn(
                                dummy_chain(ep),
                                ep["symptom_name"],
                                "value",
                                "raw",
                                ep["normal_df"],
                                ep["abnormal_df"],
                            )
                    except Exception:
                        res = []
                    correct.append(
                        _parse_baseline_result(res, ep, n_nodes, rng_fb) ==
                        ep["int_node"])
                correct = np.array(correct, dtype=bool)
                mean = float(correct.mean()) if len(correct) else 0.0
                lo, hi = _bootstrap_ci(correct) if len(correct) else (0.0, 0.0)
                bl_rows[bl_name].append((mean, lo, hi))
                print(
                    f"  [{bl_name}] {fn_type} n={n_nodes:>3} n_int={n_int:>4}  "
                    f"acc={mean:.3f}  valid={len(correct)}/{n_episodes}",
                    flush=True,
                )

            # ── True-graph baselines ──────────────────────────────────────────
            for bl_name, (bl_fn, _, _) in true_graph_methods.items():
                correct = []
                for ep in episodes:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = bl_fn(
                                ep["true_graph_rev"],
                                ep["symptom_name"],
                                "value",
                                "raw",
                                ep["normal_df"],
                                ep["abnormal_df"],
                            )
                    except Exception:
                        res = []
                    correct.append(
                        _parse_baseline_result(res, ep, n_nodes, rng_fb) ==
                        ep["int_node"])
                correct = np.array(correct, dtype=bool)
                mean = float(correct.mean()) if len(correct) else 0.0
                lo, hi = _bootstrap_ci(correct) if len(correct) else (0.0, 0.0)
                bl_rows[bl_name].append((mean, lo, hi))
                print(
                    f"  [{bl_name}] {fn_type} n={n_nodes:>3} n_int={n_int:>4}  "
                    f"acc={mean:.3f}  valid={len(correct)}/{n_episodes}",
                    flush=True,
                )

            # ── PC-based baselines (sequential) ───────────────────────────────
            if pc_methods and episodes:
                print(
                    f"  [PC] running discovery for {len(episodes)} episodes...",
                    flush=True,
                )
                pc_rng = np.random.default_rng(seed + n_int + 7)
                pc_graphs = []
                for i, ep in enumerate(episodes):
                    if i % 10 == 0:
                        print(f"    [PC] ep {i}/{len(episodes)}...",
                              flush=True)
                    pc_rng_ep = np.random.default_rng(
                        int(pc_rng.integers(0, 2**31)))
                    pc_graphs.append(
                        _pc_dag_timed(ep["obs_norm"], ep["node_names"],
                                      pc_rng_ep))

                for bl_name, (bl_fn, _, _) in pc_methods.items():
                    correct = []
                    for i, ep in enumerate(episodes):
                        if pc_graphs[i] is None:
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                res = bl_fn(
                                    pc_graphs[i].reverse(),
                                    ep["symptom_name"],
                                    "value",
                                    "raw",
                                    ep["normal_df"],
                                    ep["abnormal_df"],
                                )
                        except Exception:
                            res = []
                        correct.append(
                            _parse_baseline_result(res, ep, n_nodes, rng_fb) ==
                            ep["int_node"])
                    correct = np.array(correct, dtype=bool)
                    mean = float(correct.mean()) if len(correct) else 0.0
                    lo, hi = _bootstrap_ci(correct) if len(correct) else (0.0,
                                                                          0.0)
                    bl_rows[bl_name].append((mean, lo, hi))
                    print(
                        f"  [{bl_name}] {fn_type} n={n_nodes:>3} n_int={n_int:>4}  "
                        f"acc={mean:.3f}  valid={len(correct)}/{n_episodes}",
                        flush=True,
                    )

        model_results[n_nodes] = model_rows
        for name in baseline_methods:
            bl_results[name][n_nodes] = bl_rows[name]

    return model_results, bl_results


# ─── Baseline methods ────────────────────────────────────────────────────────

_PETSHOP_CODE = str(
    Path(__file__).parent.parent.parent / "petshop-root-cause-analysis-main" /
    "code")


def _pc_discover_edges(obs_norm, node_names, rng_seed):
    """Run PC, return list of oriented (src, dst) tuples."""
    pc_alg = _pc_alg
    n = obs_norm.shape[1]
    rng = np.random.default_rng(rng_seed)

    # Limit BLAS/OpenMP threads spawned inside causallearn to 1 so PC workers
    # don't saturate the CPU and starve the main thread.
    from threadpoolctl import threadpool_limits

    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore")
        # Scale depth with graph size: O(n^2 * C(n, depth)) must stay tractable
        n_nodes = obs_norm.shape[1]
        depth = 1 if n_nodes >= 50 else (2 if n_nodes >= 20 else 3)
        cg = pc_alg(
            obs_norm,
            alpha=0.05,
            indep_test="fisherz",
            depth=depth,
            show_progress=False,
            verbose=False,
        )

    G = cg.G.graph
    edges = []
    undirected = []

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
        if rng.random() < 0.5:
            edges.append((node_names[i], node_names[j]))
        else:
            edges.append((node_names[j], node_names[i]))

    # Fallback: fully connected DAG in random topological order
    if not edges:
        order = list(range(n))
        rng.shuffle(order)
        for a in range(n):
            for b in range(a + 1, n):
                edges.append((node_names[order[a]], node_names[order[b]]))

    return edges


def _pc_dag(obs_norm, node_names, rng):
    """PC causal discovery → acyclic nx.DiGraph. Falls back to random chain."""
    rng_seed = int(rng.integers(0, 2**31))
    try:
        edges = _pc_discover_edges(obs_norm, node_names, rng_seed)
    except Exception:
        edges = [(node_names[i], node_names[i + 1])
                 for i in range(len(node_names) - 1)]
    g = nx.DiGraph(edges)
    for name in node_names:
        if name not in g:
            g.add_node(name)
    if not nx.is_directed_acyclic_graph(g):
        g = nx.DiGraph([(node_names[i], node_names[i + 1])
                        for i in range(len(node_names) - 1)])
    return g


def _pc_dag_timed(obs_norm, node_names, rng, timeout=60):
    """Run _pc_dag with a SIGALRM timeout; returns None if it hangs."""

    def _handler(signum, frame):
        raise TimeoutError("PC timed out")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        result = _pc_dag(obs_norm, node_names, rng)
    except (TimeoutError, Exception):
        result = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return result


def _build_baseline_methods():
    """Import baseline methods; return dict name → (callable, needs_graph)."""
    if _PETSHOP_CODE not in sys.path:
        sys.path.insert(0, _PETSHOP_CODE)

    methods = {}

    # Tuples: (callable, use_true_graph, needs_graph)
    # needs_graph=False → skip PC discovery entirely (method ignores the graph)

    try:
        from ranked_correlation import make_ranked_correlation

        methods["ranked_corr"] = (
            make_ranked_correlation(filter_for_anomalous_metrics=False,
                                    root_cause_top_k=20),
            False,
            False,  # graph-free
        )
    except Exception as e:
        print(f"[warn] ranked_correlation not available: {e}")

    try:
        from counterfactual_attribution import make_counterfactual_attribution_method

        _cfn = make_counterfactual_attribution_method(n_jobs=-1)
        methods["counterfactual_true"] = (_cfn, True, True)
        methods["counterfactual_pc"] = (_cfn, False, True)
    except Exception as e:
        print(f"[warn] counterfactual_attribution not available: {e}")

    try:
        from baseline_anomaly_traversal import make_baseline_analyze_root_causes

        _anom = make_baseline_analyze_root_causes()
        methods["anomaly_trav_true"] = (_anom, True, True)
        methods["anomaly_trav_pc"] = (_anom, False, True)
    except Exception as e:
        print(f"[warn] baseline_anomaly_traversal not available: {e}")

    try:
        from circa import make_circa

        _circa = make_circa(root_cause_top_k=20)
        methods["circa_true"] = (_circa, True, True)
        methods["circa_pc"] = (_circa, False, True)
    except Exception as e:
        print(f"[warn] circa not available: {e}")

    # PyRCA-based graph-free methods (require sfr-pyrca)
    try:
        from epsilon_diagnosis import make_epsilon_diagnosis

        methods["epsilon_diag"] = (
            make_epsilon_diagnosis(root_cause_top_k=20),
            False,
            False,
        )
    except Exception:
        pass

    try:
        from random_walk import make_random_walk

        methods["rnd_walk"] = (make_random_walk(root_cause_top_k=20), False,
                               False)
    except Exception:
        pass

    try:
        from hierarchical_rcd import make_hierarchical_rcd

        methods["hier_rcd"] = (make_hierarchical_rcd(root_cause_top_k=20),
                               False, False)
    except Exception:
        pass

    # BARO (RCAEval's own e2e implementation): wrapped in the same
    # data,inject_time -> petshop-contract adapter as
    # prim/evaluation/rcaeval_baro.py's _rcaeval_as_petshop_method, since
    # episodes here already build normal_df/abnormal_df in the exact same
    # (node, "value", "raw") MultiIndex convention that adapter expects.
    try:
        from prim.evaluation.rcaeval_baro import make_baro_method

        baro_method = make_baro_method()
        if baro_method is not None:
            methods["baro"] = (baro_method, False, False)  # graph-free
    except Exception as e:
        print(f"[warn] baro not available: {e}")

    # Simple reviewer-requested baselines (graph-free, numpy/pandas only, no
    # pyrca/dowhy dependency — always available).
    from prim.evaluation.simple_baselines import (
        make_gaussian_divergence_baseline,
        make_percentile_baseline,
    )

    methods["percentile95"] = (make_percentile_baseline(), False, False)
    methods["gaussian_kl"] = (make_gaussian_divergence_baseline(), False,
                              False)

    return methods


# ─── Plotting ─────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

# Baseline display names and styles
_BASELINE_STYLE = {
    "ranked_corr": ("Ranked corr", "^:", 0.65),
    "counterfactual_true": ("Counterfact(true)", "s-", 0.80),
    "counterfactual_pc": ("Counterfact(PC)", "s--", 0.65),
    "anomaly_trav_true": ("Traversal(true)", "D-", 0.80),
    "anomaly_trav_pc": ("Traversal(PC)", "D--", 0.65),
    "epsilon_diag": ("EpsilonDiag", "v:", 0.65),
    "circa_true": ("CIRCA(true)", "p-", 0.80),
    "circa_pc": ("CIRCA(PC)", "p--", 0.65),
    "rnd_walk": ("RandomWalk", "h:", 0.65),
    "hier_rcd": ("HierarchicalRCD", "*:", 0.65),
    "baro": ("BARO", "o:", 0.65),
    "percentile95": ("Percentile95", "x:", 0.65),
    "gaussian_kl": ("GaussianKL", "+:", 0.65),
}


def _node_colors(node_sizes):
    cmap = plt.cm.plasma
    n = max(len(node_sizes) - 1, 1)
    return {k: cmap(i / n) for i, k in enumerate(node_sizes)}


def plot_fn_type(fn_type, model_results, all_baseline_res, node_sizes,
                 int_sizes, out_path):
    """Plot Recall@1 vs n_int (intervention sample size).
    Model: one line per n_nodes (solid). Baselines: averaged across n_nodes (dashed).
    """
    plt.rcParams.update(_RCPARAMS)
    ns_arr = np.array(int_sizes)
    colors = _node_colors(node_sizes)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Meta-RCA: lines per n_nodes
    for n_nodes in node_sizes:
        rows_m = model_results.get(n_nodes)
        if not rows_m:
            continue
        means = np.array([r[0] for r in rows_m])
        lo = np.array([r[1] for r in rows_m])
        hi = np.array([r[2] for r in rows_m])
        ax.errorbar(
            ns_arr,
            means,
            yerr=[means - lo, hi - means],
            fmt="o-",
            color=colors[n_nodes],
            lw=1.8,
            markersize=3,
            capsize=2,
            elinewidth=0.7,
            label=f"Meta-RCA n={n_nodes}",
        )

    # Baselines: average across n_nodes for readability
    bl_colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_baseline_res), 1)))
    for bl_idx, (bl_name, bl_res) in enumerate(all_baseline_res.items()):
        label, fmt, alpha = _BASELINE_STYLE.get(bl_name, (bl_name, "x--", 0.6))
        all_means = []
        for n_nodes in node_sizes:
            rows = bl_res.get(n_nodes)
            if rows:
                all_means.append([r[0] for r in rows])
        if not all_means:
            continue
        avg = np.mean(all_means, axis=0)
        ax.plot(
            ns_arr,
            avg,
            fmt,
            color=bl_colors[bl_idx],
            lw=1.2,
            markersize=3,
            alpha=alpha,
            label=f"{label} (avg n)",
        )

    ax.set_xlabel("$n_{int}$")
    ax.set_ylabel("Recall@1")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Prior evaluation — {fn_type}")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=6,
              frameon=False,
              loc="lower right",
              ncol=2,
              columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def plot_node_size_sweep(fn_type, model_results, all_baseline_res, node_sizes,
                         int_sizes, out_path):
    """x=n_nodes, y=Recall@1 averaged across selected n_int values."""
    plt.rcParams.update(_RCPARAMS)
    cmap = plt.cm.viridis
    int_subset = [s
                  for s in [1, 5, 20, 50] if s in int_sizes] or list(int_sizes)
    colors = {
        s: cmap(i / max(len(int_subset) - 1, 1))
        for i, s in enumerate(int_subset)
    }
    ns_arr = np.array(node_sizes)

    fig, ax = plt.subplots(figsize=(6, 4))

    for n_int in int_subset:
        idx = int_sizes.index(n_int)
        color = colors[n_int]
        m_means = np.array([
            model_results[k][idx][0] for k in node_sizes
            if k in model_results and model_results[k]
        ])
        if len(m_means):
            ax.plot(
                ns_arr[:len(m_means)],
                m_means,
                "o-",
                color=color,
                lw=1.8,
                markersize=3,
                label=f"Meta-RCA n_int={n_int}",
            )

    bl_colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_baseline_res), 1)))
    for bl_idx, (bl_name, bl_res) in enumerate(all_baseline_res.items()):
        label, fmt, alpha = _BASELINE_STYLE.get(bl_name, (bl_name, "x--", 0.6))
        b_means = np.array([
            bl_res[k][idx][0] for k in node_sizes if k in bl_res and bl_res[k]
        ])
        if len(b_means):
            ax.plot(
                ns_arr[:len(b_means)],
                b_means,
                fmt,
                color=bl_colors[bl_idx],
                lw=1.2,
                markersize=3,
                alpha=alpha,
                label=f"{label} n_int={int_subset[0]}",
            )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Scaling with graph size — {fn_type}")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=6, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


# ─── Tables ──────────────────────────────────────────────────────────────────


def _save_tables(fn_types, node_sizes, int_sizes, all_model_res,
                 all_baseline_res, out_dir):
    """Print a summary table and save a CSV with all results."""
    import csv

    nan3 = (float("nan"), ) * 3
    baseline_names = list(all_baseline_res.keys())
    rows = []
    for fn_type in fn_types:
        for n_nodes in node_sizes:
            for s_idx, n_int in enumerate(int_sizes):
                m_rows = all_model_res.get(fn_type, {}).get(n_nodes)
                m_mean, m_lo, m_hi = m_rows[s_idx] if m_rows else nan3
                row = dict(
                    fn_type=fn_type,
                    n_nodes=n_nodes,
                    n_int=n_int,
                    meta_rca=m_mean,
                    meta_rca_lo=m_lo,
                    meta_rca_hi=m_hi,
                )
                for bl in baseline_names:
                    bl_rows = all_baseline_res[bl].get(fn_type,
                                                       {}).get(n_nodes)
                    bl_mean, bl_lo, bl_hi = bl_rows[s_idx] if bl_rows else nan3
                    row[bl] = bl_mean
                    row[f"{bl}_lo"] = bl_lo
                    row[f"{bl}_hi"] = bl_hi
                rows.append(row)

    # ── Print to stdout ───────────────────────────────────────────────────────
    def _fmt(v):
        return f"{v:.3f}" if v == v else "   —"

    col_w = 8
    header = f"{'fn_type':<10} {'n_nodes':>7} {'n_int':>6}  {'Meta-RCA':>{col_w}}"
    for bl in baseline_names:
        lbl = _BASELINE_STYLE.get(bl, (bl, ))[0][:col_w]
        header += f"  {lbl:>{col_w}}"
    print(f"\n{'='*len(header)}")
    print("Summary table: Recall@1")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    prev_fn = None
    for r in rows:
        if r["fn_type"] != prev_fn:
            if prev_fn is not None:
                print()
            prev_fn = r["fn_type"]
        line = (f"{r['fn_type']:<10} {r['n_nodes']:>7} {r['n_int']:>6}  "
                f"{_fmt(r['meta_rca']):>{col_w}}")
        for bl in baseline_names:
            line += f"  {_fmt(r[bl]):>{col_w}}"
        print(line)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "prior_eval_results.csv")
    fieldnames = [
        "fn_type",
        "n_nodes",
        "n_int",
        "meta_rca",
        "meta_rca_lo",
        "meta_rca_hi",
    ]
    for bl in baseline_names:
        fieldnames += [bl, f"{bl}_lo", f"{bl}_hi"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=
        "Evaluate Meta-RCA vs baseline methods on prior synthetic data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fn_types",
        nargs="+",
        default=FN_TYPES,
        choices=list(_EQUATION_MIX),
    )
    parser.add_argument("--node_sizes",
                        nargs="+",
                        type=int,
                        default=NODE_SIZES)
    parser.add_argument("--int_sizes", nargs="+", type=int, default=INT_SIZES)
    parser.add_argument(
        "--n_obs",
        type=int,
        default=N_OBS,
        help="Fixed observational sample size across the int-size sweep",
    )
    parser.add_argument(
        "--compare_traversal",
        action="store_true",
        help="Run baseline methods alongside Meta-RCA",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Restrict --compare_traversal to these method names "
        "(e.g. --methods percentile95 gaussian_kl).",
    )
    parser.add_argument(
        "--no_counterfactual",
        action="store_true",
        help="Skip counterfactual attribution (slow DoWhy Shapley)",
    )
    parser.add_argument("--out_dir",
                        type=str,
                        default=".",
                        help="Directory for output plots/CSV")
    parser.add_argument(
        "--skip_model",
        action="store_true",
        help="Skip model evaluation (only run baselines)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  max_nodes={args.max_nodes}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = None
    if not args.skip_model:
        model = PRIMNeuralProcess(
            embedding_size=160,
            num_attention_heads=8,
            mlp_hidden_size=512,
            num_layers=6,
            max_nodes=args.max_nodes,
            dropout=0.0,
        ).to(device)

        ckpt = args.checkpoint
        if ckpt is None:
            candidates = ["context_meta_multigpu_final.pt"]
            candidates += [
                f"context_meta_multigpu_epoch_{e}.pt"
                for e in range(59, 0, -1)
            ]
            for p in candidates:
                if os.path.exists(p):
                    ckpt = p
                    break

        if ckpt is None or not os.path.exists(ckpt):
            print("No checkpoint found — pass --checkpoint <path>.")
            return

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        print(
            f"Loaded: {ckpt}  params={sum(p.numel() for p in model.parameters()):,}"
        )
        model.eval()

    # ── Load baseline methods ─────────────────────────────────────────────
    baseline_methods = {}
    if args.compare_traversal:
        print("\nLoading baseline methods...")
        baseline_methods = _build_baseline_methods()
        if args.no_counterfactual:
            for k in ["counterfactual_true", "counterfactual_pc"]:
                baseline_methods.pop(k, None)
        if args.methods:
            baseline_methods = {
                k: v
                for k, v in baseline_methods.items() if k in args.methods
            }
        print(f"  Available: {list(baseline_methods.keys())}")

    all_model_res = {}  # fn_type -> node_size -> rows
    # all_baseline_res[method_name][fn_type][node_size] = rows
    all_baseline_res = {name: {} for name in baseline_methods}

    # ── Evaluate per fn_type ─────────────────────────────────────────────────
    for fn_type in args.fn_types:
        print(f"\n{'='*65}")
        print(f"fn_type = {fn_type}")
        print(f"{'='*65}")

        model_res, bl_res_by_method = run_all_sweeps(
            model,
            device,
            args.max_nodes,
            fn_type,
            args.node_sizes,
            args.int_sizes,
            args.n_obs,
            args.n_episodes,
            args.seed,
            baseline_methods,
        )
        all_model_res[fn_type] = model_res
        for bl_name, bl_res in bl_res_by_method.items():
            all_baseline_res[bl_name][fn_type] = bl_res

        # Plots: aggregate baselines for this fn_type
        fn_bl = {
            name: all_baseline_res[name].get(fn_type, {})
            for name in baseline_methods
        }

        plot_fn_type(
            fn_type,
            model_res,
            fn_bl,
            args.node_sizes,
            args.int_sizes,
            out_path=os.path.join(args.out_dir, f"prior_eval_{fn_type}.pdf"),
        )
        if model_res or fn_bl:
            plot_node_size_sweep(
                fn_type,
                model_res,
                fn_bl,
                args.node_sizes,
                args.int_sizes,
                out_path=os.path.join(args.out_dir,
                                      f"prior_eval_{fn_type}_vs_nodes.pdf"),
            )

    # Flatten all_baseline_res for _save_tables:
    # _save_tables expects all_baseline_res[method][fn_type][n_nodes]
    _save_tables(
        args.fn_types,
        args.node_sizes,
        args.int_sizes,
        all_model_res,
        all_baseline_res,
        args.out_dir,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
