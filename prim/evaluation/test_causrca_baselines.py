"""
Evaluate baseline RCA methods on the CausRCA digital-twin benchmark.

Baselines: epsilon-diagnosis (graph-free), RCD (graph-free), anomaly traversal
(uses PC-discovered or expert graph), and CIRCA/HT (uses expert graph).

Data loading and preprocessing are shared with test_causrca.py.

Multi-root-cause handling: all methods are configured with root_cause_top_k=3
and evaluated with MAP@3, consistent with the CausRCA benchmark protocol.
Graph-requiring methods (traversal, CIRCA) use either the expert GML graph
("expert" mode) or a PC-discovered graph ("pc" mode).
"""

import importlib
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
CAUSRCA_ROOT = PROJECT_ROOT / "causRCA-main"
PETSHOP_CODE = PROJECT_ROOT / "petshop-root-cause-analysis-main" / "code"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CAUSRCA_ROOT / "src"))
sys.path.insert(0, str(PETSHOP_CODE))

from test_causrca import (
    DATASETS,
    TIME_STEP_MS,
    MIN_NORMAL_STEPS,
    collect_all_runs,
    load_relevant_nodes,
    load_full_nodes,
    calculate_apk,
    _bootstrap_ci,
)

EXPERT_GRAPH_DIR = CAUSRCA_ROOT / "data" / "expert_graph"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_NODES = 100
TOP_K = 3

# PC algorithm depth limit (same as prior-eval and petshop baseline scripts)
PC_ALPHA = 0.05
PC_MAX_COND_VARS = 3

# ─────────────────────────────────────────────────────────────────────────────
# Expert graph loading
# ─────────────────────────────────────────────────────────────────────────────


def load_expert_graph(dataset_name: str, node_names: list) -> nx.DiGraph:
    """Load the expert GML graph for a dataset and restrict to node_names."""
    gml_candidates = list((CAUSRCA_ROOT / "data" / "dig_twin" /
                           f"exp_{dataset_name}").glob("*_expert_graph.gml"))
    if not gml_candidates:
        # Fallback: try expert_graph dir
        gml_candidates = list(EXPERT_GRAPH_DIR.glob(f"*{dataset_name}*.gml"))
    if not gml_candidates:
        return None
    G = nx.read_gml(str(gml_candidates[0]))
    # Restrict to node_names subset (marginalize missing nodes)
    nodes_to_remove = [n for n in list(G.nodes()) if n not in node_names]
    for node in nodes_to_remove:
        # rewire children to grandparents before removing
        for parent in list(G.predecessors(node)):
            for child in list(G.successors(node)):
                G.add_edge(parent, child)
        G.remove_node(node)
    # Add any node_names not in the graph as isolated nodes
    for n in node_names:
        if n not in G:
            G.add_node(n)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# PC graph discovery
# ─────────────────────────────────────────────────────────────────────────────


def _pc_discover(obs_array: np.ndarray, node_names: list, rng) -> nx.DiGraph:
    """Run PC algorithm on obs data and return a DiGraph over node_names."""
    from causallearn.search.ConstraintBased.PC import pc as pc_alg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc_alg(
            obs_array,
            alpha=PC_ALPHA,
            indep_test="fisherz",
            depth=PC_MAX_COND_VARS,
            show_progress=False,
            verbose=False,
        )

    n = obs_array.shape[1]
    G_mat = cg.G.graph
    edges = []

    for i in range(n):
        for j in range(n):
            if G_mat[i, j] == 1 and G_mat[j, i] == -1:
                edges.append((node_names[i], node_names[j]))

    # Randomly orient undirected edges
    for i in range(n):
        for j in range(i + 1, n):
            if (G_mat[i, j] == -1
                    and G_mat[j, i] == -1) or (G_mat[i, j] == 1
                                               and G_mat[j, i] == 1):
                if rng.random() < 0.5:
                    edges.append((node_names[i], node_names[j]))
                else:
                    edges.append((node_names[j], node_names[i]))

    if not edges:
        for i in range(n - 1):
            edges.append((node_names[i], node_names[i + 1]))

    G = nx.DiGraph(edges)
    for nm in node_names:
        if nm not in G:
            G.add_node(nm)

    if not nx.is_directed_acyclic_graph(G):
        G = nx.DiGraph([(node_names[i], node_names[i + 1])
                        for i in range(n - 1)])
        for nm in node_names:
            if nm not in G:
                G.add_node(nm)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame format expected by petshop baseline methods
# ─────────────────────────────────────────────────────────────────────────────


def _to_baseline_df(array: np.ndarray, node_names: list) -> pd.DataFrame:
    """Convert (n_samples, n_nodes) array to MultiIndex DataFrame."""
    cols = pd.MultiIndex.from_tuples(
        [(name, "value", "raw") for name in node_names],
        names=["node", "metric", "statistic"],
    )
    return pd.DataFrame(array.astype(np.float64), columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Build baseline methods
# ─────────────────────────────────────────────────────────────────────────────


def build_methods() -> dict:
    """
    Import and instantiate baseline methods.

    Returns dict: name -> (callable, needs_graph)
      needs_graph=True  -> method requires a causal graph (traversal, circa)
      needs_graph=False -> graph-free (epsilon_diag, rcd)
    """
    methods = {}

    for bl_name, mod_str, fn_str in [
        ("epsilon_diag", "epsilon_diagnosis", "make_epsilon_diagnosis"),
        ("rcd", "hierarchical_rcd", "make_hierarchical_rcd"),
    ]:
        try:
            mod = importlib.import_module(mod_str)
            methods[bl_name] = (
                getattr(mod, fn_str)(root_cause_top_k=TOP_K),
                False,
            )
            print(f"  [ok] {bl_name}")
        except Exception as e:
            print(f"  [warn] {bl_name} unavailable: {e}")

    try:
        from baseline_anomaly_traversal import make_baseline_analyze_root_causes  # type: ignore[import]
        methods["traversal"] = (make_baseline_analyze_root_causes(), True)
        print("  [ok] traversal")
    except Exception as e:
        print(f"  [warn] traversal unavailable: {e}")

    try:
        from circa import make_circa  # type: ignore[import]
        methods["circa"] = (make_circa(root_cause_top_k=TOP_K), True)
        print("  [ok] circa")
    except Exception as e:
        print(f"  [warn] circa unavailable: {e}")

    return methods


# ─────────────────────────────────────────────────────────────────────────────
# Run one baseline on one run
# ─────────────────────────────────────────────────────────────────────────────


def _run_baseline(
    method_fn,
    needs_graph: bool,
    obs_array: np.ndarray,
    int_array: np.ndarray,
    node_names: list,
    alarm_nodes: list,
    causal_graph: nx.DiGraph | None,
    rng,
) -> list:
    """
    Call a single baseline method and return a ranked list of predicted node names.

    The petshop methods expect a 'call graph' (edges reversed from causal direction),
    a target_node (symptom/alarm node), and MultiIndex DataFrames.
    """
    normal_df = _to_baseline_df(obs_array, node_names)
    abnormal_df = _to_baseline_df(int_array, node_names)

    # Use first alarm node as the target; if none available, use first node
    target_node = alarm_nodes[0] if alarm_nodes else node_names[0]
    if target_node not in node_names:
        target_node = node_names[0]

    if needs_graph:
        if causal_graph is None:
            # Fallback to random chain if no graph available
            causal_graph = nx.DiGraph([(node_names[i], node_names[i + 1])
                                       for i in range(len(node_names) - 1)])
            for nm in node_names:
                if nm not in causal_graph:
                    causal_graph.add_node(nm)
        # Petshop methods expect the "call graph" (reversed causal direction)
        call_graph = causal_graph.reverse()
    else:
        call_graph = nx.DiGraph([(node_names[i], node_names[i + 1])
                                 for i in range(len(node_names) - 1)])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = method_fn(call_graph, target_node, "value", "raw",
                                normal_df, abnormal_df)
    except Exception:
        return []

    if not results:
        return []

    results_sorted = sorted(results, key=lambda x: x.score, reverse=True)
    seen = set()
    ranked = []
    for rc in results_sorted:
        if rc.node not in seen and rc.node in node_names:
            ranked.append(rc.node)
            seen.add(rc.node)
    return ranked[:TOP_K]


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_causrca_baselines(methods: dict,
                               mode: str = "sub",
                               graph_source: str = "expert"):
    """
    Evaluate all baseline methods on CausRCA digital-twin datasets.

    Args:
        mode:         "sub"  – dataset-specific relevant nodes
                      "full" – all 92 nodes from expert knowledge graph
        graph_source: "expert" – use provided GML expert graph
                      "pc"     – discover graph from observational data with PC

    Returns:
        dict: method_name -> DataFrame with per-run results
    """
    assert mode in ("sub", "full")
    assert graph_source in ("expert", "pc")

    rng = np.random.default_rng(42)
    full_node_names = load_full_nodes() if mode == "full" else None

    # Accumulate rows per method
    method_rows = {name: [] for name in methods}

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  [mode={mode}, graph={graph_source}]")
        print(f"{'='*60}")

        node_names = full_node_names if mode == "full" else load_relevant_nodes(
            ds_path)
        if not node_names:
            print(
                f"  WARNING: no relevant nodes found for {ds_name}, skipping.")
            continue

        n_nodes = len(node_names)
        print(
            f"  Nodes ({n_nodes}): {node_names[:5]}{'...' if n_nodes > 5 else ''}"
        )

        # Load expert graph once per dataset (for "expert" graph_source)
        expert_graph = None
        if graph_source == "expert":
            expert_graph = load_expert_graph(ds_name, node_names)
            if expert_graph is None:
                print(
                    f"  WARNING: expert graph not found for {ds_name}. Graph-requiring methods will use PC fallback."
                )

        all_runs = collect_all_runs(ds_path)
        print(f"  Total runs: {len(all_runs)}")

        for run in all_runs:
            # Get raw obs/int arrays (not padded – we need raw for baselines)
            from test_causrca import _discretize

            df = run["csv_df"]
            t_fault = run["cause_start_at"]
            t_diag = run["diagnosis_at"]

            df_normal = df[df["time_s"] <= t_fault].copy()
            df_abnormal = df[(df["time_s"] > t_fault)
                             & (df["time_s"] <= t_diag)].copy()

            if df_normal.empty or df_abnormal.empty:
                continue

            t_offset = df_abnormal["time_s"].min()
            df_abnormal = df_abnormal.copy()
            df_abnormal["time_s"] -= t_offset

            duration_normal = df_normal["time_s"].max()
            duration_abnormal = df_abnormal["time_s"].max()

            if duration_normal < (TIME_STEP_MS / 1000.0) * MIN_NORMAL_STEPS:
                continue

            try:
                wide_normal = _discretize(df_normal, duration_normal)
                wide_abnormal = _discretize(df_abnormal, duration_abnormal)
            except Exception:
                continue

            wide_normal = wide_normal.reindex(columns=node_names,
                                              fill_value=np.nan)
            wide_abnormal = wide_abnormal.reindex(columns=node_names,
                                                  fill_value=np.nan)

            normal_arr = wide_normal.values.astype(np.float64)
            abnormal_arr = wide_abnormal.values.astype(np.float64)

            # Impute NaNs
            with np.errstate(all="ignore"):
                col_means = np.nanmean(normal_arr, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            for c in range(normal_arr.shape[1]):
                normal_arr[np.isnan(normal_arr[:, c]), c] = col_means[c]
                abnormal_arr[np.isnan(abnormal_arr[:, c]), c] = col_means[c]

            # Standardise with normal statistics
            obs_mean = normal_arr.mean(axis=0, keepdims=True)
            obs_std = normal_arr.std(axis=0, keepdims=True) + 1e-8
            obs_norm = (normal_arr - obs_mean) / obs_std
            int_norm = (abnormal_arr - obs_mean) / obs_std

            # Graph: expert or PC-discovered
            if graph_source == "expert":
                causal_graph = expert_graph  # may be None — handled in _run_baseline
            else:
                try:
                    causal_graph = _pc_discover(obs_norm, node_names, rng)
                except Exception:
                    causal_graph = None

            gt = run["manipulated_vars"]
            alarm_nodes = [
                a for a in run.get("alarm_nodes", []) if a in node_names
            ]

            for method_name, (method_fn, needs_graph) in methods.items():
                ranked = _run_baseline(
                    method_fn,
                    needs_graph,
                    obs_norm,
                    int_norm,
                    node_names,
                    alarm_nodes,
                    causal_graph,
                    rng,
                )

                def in_topk(ranked, gt_list, k):
                    return int(any(g in ranked[:k] for g in gt_list))

                method_rows[method_name].append({
                    "dataset":
                    ds_name,
                    "mode":
                    mode,
                    "graph":
                    graph_source,
                    "exp":
                    run["exp_name"],
                    "run":
                    run["run_name"],
                    "manipulated":
                    "|".join(gt),
                    "top1_pred":
                    ranked[0] if ranked else "",
                    "rec@1":
                    in_topk(ranked, gt, 1),
                    "rec@3":
                    in_topk(ranked, gt, 3),
                    "rec@5":
                    in_topk(ranked, gt, 5),
                    "_ranked":
                    ranked,
                    "_gt":
                    gt,
                    "n_nodes":
                    n_nodes,
                })

            print(
                f"  {run['exp_name']}/{run['run_name']:6s}  gt={gt}  " +
                "  ".join(
                    f"{m}={method_rows[m][-1]['top1_pred'] if method_rows[m] else '?'}"
                    for m in methods))

    return {name: pd.DataFrame(rows) for name, rows in method_rows.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Results printing
# ─────────────────────────────────────────────────────────────────────────────


def print_results(results: dict, title_suffix: str = ""):
    """Print MAP@3 and Recall@k table for each method and dataset."""
    for method_name, df in results.items():
        if df.empty:
            print(f"\n[{method_name}] No results.")
            continue

        print(f"\n{'='*68}")
        print(f"Method: {method_name.upper()}{title_suffix}")
        print(f"{'='*68}")
        print(
            f"{'dataset':<14} {'map@3':>8}  {'rec@1':>6}  {'rec@3':>6}  {'n':>5}"
        )
        print("-" * 48)

        for ds in list(DATASETS.keys()) + ["OVERALL"]:
            sub = df if ds == "OVERALL" else df[df["dataset"] == ds]
            if len(sub) == 0:
                continue
            ap_per_run = [
                calculate_apk(gt, pred, k=3) for gt, pred in zip(
                    sub["_gt"].tolist(), sub["_ranked"].tolist())
            ]
            map3 = float(np.mean(ap_per_run))
            r1 = sub["rec@1"].mean()
            r3 = sub["rec@3"].mean()
            lo, hi = _bootstrap_ci(ap_per_run)
            print(
                f"{ds:<14} {map3:>8.3f}  {r1:>6.3f}  {r3:>6.3f}  {len(sub):>5}"
            )
            if ds == "OVERALL":
                print(f"{'':14} {'(90% CI: '+f'{lo:.3f}–{hi:.3f})':>8}")


def print_comparison_table(results: dict):
    """Single-table overview: MAP@3 per method × dataset."""
    print("\n" + "=" * 80)
    print("CausRCA Baselines — MAP@3 Comparison")
    print("=" * 80)

    method_names = list(results.keys())
    header = f"{'dataset':<14}" + "".join(f"{m:>14}" for m in method_names)
    print(header)
    print("-" * (14 + 14 * len(method_names)))

    for ds in list(DATASETS.keys()) + ["OVERALL"]:
        row = f"{ds:<14}"
        for method_name in method_names:
            df = results[method_name]
            if df.empty:
                row += f"{'—':>14}"
                continue
            sub = df if ds == "OVERALL" else df[df["dataset"] == ds]
            if len(sub) == 0:
                row += f"{'—':>14}"
                continue
            ap_per_run = [
                calculate_apk(gt, pred, k=3) for gt, pred in zip(
                    sub["_gt"].tolist(), sub["_ranked"].tolist())
            ]
            map3 = float(np.mean(ap_per_run))
            row += f"{map3:>14.3f}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse
    global DATASETS

    parser = argparse.ArgumentParser(
        description=
        "Evaluate baseline RCA methods on CausRCA digital-twin benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        choices=["sub", "full"],
        default=["sub"],
        help="Node set: sub=relevant nodes per subsystem, full=all 92 nodes",
    )
    parser.add_argument(
        "--graph",
        type=str,
        nargs="+",
        choices=["expert", "pc"],
        default=["expert"],
        help=
        "Graph source for graph-requiring methods: expert GML or PC discovery",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["epsilon_diag", "rcd", "traversal", "circa"],
        default=None,
        help="Methods to run (default: all available)",
    )
    args = parser.parse_args()

    DATASETS = {k: v for k, v in DATASETS.items() if k in args.datasets}

    print("Loading baseline methods...")
    all_methods = build_methods()

    if args.methods:
        all_methods = {
            k: v
            for k, v in all_methods.items() if k in args.methods
        }

    if not all_methods:
        print("No baseline methods available. Install pyrca and causallearn.")
        return

    for mode in args.mode:
        for graph_source in args.graph:
            tag = f"  [mode={mode}, graph={graph_source}]"
            print(f"\n{'='*70}")
            print(f"BASELINES{tag}")
            print(f"{'='*70}")

            results = evaluate_causrca_baselines(all_methods,
                                                 mode=mode,
                                                 graph_source=graph_source)

            print_comparison_table(results)
            print_results(results, title_suffix=tag)

            # Save CSVs
            for method_name, df in results.items():
                if df.empty:
                    continue
                out_cols = [c for c in df.columns if not c.startswith("_")]
                out_file = str(
                    RESULTS_DIR /
                    f"causrca_baselines_{method_name}_{mode}_{graph_source}.csv"
                )
                df[out_cols].to_csv(out_file, index=False)
                print(f"  Saved: {out_file}")


if __name__ == "__main__":
    main()
