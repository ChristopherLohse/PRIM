"""
Evaluate our meta-learning model on the PetShop RCA benchmark.

Computes top-1 and top-3 recall per (scenario, target_metric) to match
Table 3 from the petshop paper. Zero-shot, single eval-mode forward pass,
no context, no MC dropout.

Uses latency/p95 for latency issues and availability/Average for
availability issues. Model trained with max_nodes=50.
"""

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from prim.model.prim_neural_process import PRIMNeuralProcess

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = str(_PROJECT_ROOT / "petshop-root-cause-analysis-main" /
                  "dataset")
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    "low_traffic",
    "high_traffic",
    "temporal_traffic1",
    "temporal_traffic2",
]

MAX_NODES = 100

METRIC_CONFIG = {
    "latency": ("latency", "p95"),
    "availability": ("availability", "Average"),
}


def predict(model, query_obs, query_int, target_mask, n_nodes=None):
    """
    Single eval-mode forward pass. No context, no MC dropout.

    Args:
        query_obs:   (n_queries, n_obs, max_nodes)
        query_int:   (n_queries, n_int, max_nodes)
        target_mask: (n_queries, max_nodes)
        n_nodes:     actual number of nodes (pads beyond this are masked)

    Returns:
        probs: (n_queries, max_nodes) - softmax probabilities
    """
    model.eval()
    max_nodes = query_obs.shape[-1]
    with torch.no_grad():
        logits = model(query_obs,
                       query_int,
                       target_mask=target_mask,
                       n_nodes=n_nodes)
        if n_nodes is not None and n_nodes < max_nodes:
            pad_mask = torch.zeros_like(logits)
            pad_mask[:, n_nodes:] = float("-inf")
            logits = logits + pad_mask
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def load_scenario(path):
    """Load graph, normal metrics, and train/test issues."""
    graph = nx.from_pandas_adjacency(
        pd.read_csv(os.path.join(path, "graph.csv"), index_col=0),
        create_using=nx.DiGraph,
    )
    normal_metrics = pd.read_csv(
        os.path.join(path, "noissue", "metrics.csv"),
        header=[0, 1, 2],
        index_col=0,
    )
    issues = {"train": [], "test": []}
    for split in issues:
        split_dir = os.path.join(path, split)
        if not os.path.exists(split_dir):
            continue
        for issue_name in sorted(os.listdir(split_dir)):
            if issue_name.startswith("."):
                continue
            issue_path = os.path.join(split_dir, issue_name)
            metrics = pd.read_csv(
                os.path.join(issue_path, "metrics.csv"),
                header=[0, 1, 2],
                index_col=0,
            )
            with open(os.path.join(issue_path, "target.json")) as f:
                target = json.load(f)
            issues[split].append((metrics, target))
    return graph, normal_metrics, issues


def reduce_df(df, metric, statistic):
    """Extract columns for a given metric and statistic."""
    data = df.loc[:, (slice(None), [metric], [statistic])]
    data.columns = [c[0] for c in data.columns]
    return data


def get_node_names(normal_metrics, metric, statistic):
    """Get node names that have data for the given metric/statistic."""
    df = reduce_df(normal_metrics, metric, statistic)
    df = df.dropna(axis=1, how="all")
    return list(df.columns)


def petshop_to_model_input(normal_metrics, abnormal_metrics, node_names,
                           metric, statistic):
    """
    Convert petshop normal/abnormal metrics to model's (obs, int) format.

    Returns:
        obs: (n_obs, MAX_NODES) float32
        int_data: (n_int, MAX_NODES) float32
    """
    normal_df = reduce_df(normal_metrics, metric, statistic)
    abnormal_df = reduce_df(abnormal_metrics, metric, statistic)

    normal_df = normal_df.reindex(columns=node_names, fill_value=np.nan)
    abnormal_df = abnormal_df.reindex(columns=node_names, fill_value=np.nan)

    normal_arr = normal_df.values.astype(np.float64)
    abnormal_arr = abnormal_df.values.astype(np.float64)

    # Impute NaNs with column mean
    col_means = np.nanmean(normal_arr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for c in range(normal_arr.shape[1]):
        normal_arr[np.isnan(normal_arr[:, c]), c] = col_means[c]
        abnormal_arr[np.isnan(abnormal_arr[:, c]), c] = col_means[c]

    n_nodes = len(node_names)

    # Normalize using normal data statistics; clip at ±10 to match training
    # preprocessing and prevent exploding values for near-zero-std nodes.
    _CLIP = 10.0
    obs_mean = normal_arr.mean(axis=0, keepdims=True)
    obs_std = normal_arr.std(axis=0, keepdims=True) + 1e-8
    obs_normalized = np.clip((normal_arr - obs_mean) / obs_std, -_CLIP, _CLIP)
    int_normalized = np.clip((abnormal_arr - obs_mean) / obs_std, -_CLIP,
                             _CLIP)

    # Pad to MAX_NODES (same scaling as training)
    scale_factor = MAX_NODES / n_nodes
    n_obs = obs_normalized.shape[0]
    n_int = int_normalized.shape[0]

    obs_padded = np.zeros((n_obs, MAX_NODES), dtype=np.float32)
    int_padded = np.zeros((n_int, MAX_NODES), dtype=np.float32)
    obs_padded[:, :n_nodes] = obs_normalized * scale_factor
    int_padded[:, :n_nodes] = int_normalized * scale_factor

    return obs_padded, int_padded


def evaluate_petshop(model, device, split="test"):
    """
    Evaluate model on PetShop benchmark.

    Zero-shot, single eval-mode forward pass. No context, no MC dropout.
    """
    rows = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario} ---")
        scenario_path = os.path.join(DATASET_DIR, scenario)
        graph, normal_metrics, issues = load_scenario(scenario_path)

        for idx, (abnormal_metrics, target) in enumerate(issues[split]):
            target_node = target["target"]["node"]
            target_metric = target["target"]["metric"]
            root_cause_node = target["root_cause"]["node"]

            metric, statistic = METRIC_CONFIG.get(target_metric,
                                                  ("latency", "p95"))
            node_names = get_node_names(normal_metrics, metric, statistic)
            n_nodes = len(node_names)

            query_obs, query_int = petshop_to_model_input(
                normal_metrics, abnormal_metrics, node_names, metric,
                statistic)

            target_mask = np.zeros(MAX_NODES, dtype=np.float32)
            if target_node in node_names:
                target_mask[node_names.index(target_node)] = 1.0

            query_obs_t = torch.from_numpy(query_obs).unsqueeze(0).to(device)
            query_int_t = torch.from_numpy(query_int).unsqueeze(0).to(device)
            target_mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(
                device)

            probs = predict(model,
                            query_obs_t,
                            query_int_t,
                            target_mask_t,
                            n_nodes=n_nodes)
            probs = probs[0]  # (MAX_NODES,)

            ranked_indices = np.argsort(-probs)
            ranked_nodes = [
                node_names[i] for i in ranked_indices if i < n_nodes
            ]

            def topk_hits(ranked, rc, ks=(1, 3, 5)):
                return {k: int(rc in ranked[:k]) for k in ks}

            hits = topk_hits(ranked_nodes, root_cause_node)

            if root_cause_node in node_names:
                rc_idx = node_names.index(root_cause_node)
                rc_prob = probs[rc_idx]
                rc_rank = int((probs[:n_nodes] > rc_prob).sum()) + 1
            else:
                rc_prob = 0.0
                rc_rank = n_nodes + 1

            rows.append({
                "scenario": scenario,
                "target_metric": target_metric,
                "issue_idx": idx,
                "root_cause": root_cause_node,
                "top1_pred": ranked_nodes[0] if ranked_nodes else "",
                "top1_recall": hits[1],
                "top3_recall": hits[3],
                "top5_recall": hits[5],
                "rc_rank": rc_rank,
                "rc_prob": float(rc_prob),
                "n_nodes": n_nodes,
            })

            print(f"  issue_{idx}: "
                  f"metric={target_metric} "
                  f"rc={root_cause_node} "
                  f"pred={ranked_nodes[0] if ranked_nodes else '?'} "
                  f"rank={rc_rank} "
                  f"top3={'Y' if hits[3] else 'N'}")

    return pd.DataFrame(rows)


def _bootstrap_ci(values, n_boot=500, alpha=0.05, seed=0):
    """Bootstrap 90% CI (5th–95th percentile) for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.percentile(boot, 100 * alpha)), float(
        np.percentile(boot, 100 * (1 - alpha)))


def print_table(df):
    """Print results matching the paper's Table 3 format."""
    scenario_map = {
        "low_traffic": "low",
        "high_traffic": "high",
        "temporal_traffic1": "temporal",
        "temporal_traffic2": "temporal",
    }
    df = df.copy()
    df["scenario_label"] = df["scenario"].map(scenario_map)

    print("\n" + "=" * 76)
    print("PetShop Benchmark Results - Top-k Recall")
    print("=" * 76)

    for k in [1, 3, 5]:
        col = f"top{k}_recall"
        print(f"\n--- Top-{k} Recall ---")
        print(
            f"{'scenario':<12} {'metric':<15} {'recall':>8}  {'ci_lo':>8}  {'ci_hi':>8}  {'n':>5}"
        )
        print("-" * 62)

        for scenario_label in ["low", "high", "temporal"]:
            for metric in ["latency", "availability"]:
                mask = (df["scenario_label"]
                        == scenario_label) & (df["target_metric"] == metric)
                subset = df[mask]
                if len(subset) == 0:
                    continue
                recall = subset[col].mean()
                lo, hi = _bootstrap_ci(subset[col].values)
                print(
                    f"{scenario_label:<12} {metric:<15} {recall:>8.3f}  {lo:>8.3f}  {hi:>8.3f}  {len(subset):>5}"
                )

        print("-" * 62)
        lo, hi = _bootstrap_ci(df[col].values)
        print(
            f"{'Overall':<12} {'':15} {df[col].mean():>8.3f}  {lo:>8.3f}  {hi:>8.3f}  {len(df):>5}"
        )


def print_detailed_results(df):
    """Print additional analysis."""
    print("\n" + "=" * 70)
    print("Detailed Analysis")
    print("=" * 70)

    print("\n--- Per Root-Cause Node ---")
    for rc in sorted(df["root_cause"].unique()):
        subset = df[df["root_cause"] == rc]
        print(f"  {rc}: top1={subset['top1_recall'].mean():.2f}  "
              f"top3={subset['top3_recall'].mean():.2f}  "
              f"avg_rank={subset['rc_rank'].mean():.1f}  n={len(subset)}")

    print(
        f"\nMean rank: {df['rc_rank'].mean():.2f}  Median: {df['rc_rank'].median():.1f}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate on PetShop benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model - must match training architecture
    model = PRIMNeuralProcess(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=100,
        dropout=0.0,
    ).to(device)
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = ["context_meta_multigpu_final.pt"]
        candidates += [
            f"context_meta_multigpu_epoch_{e}.pt" for e in range(50, 0, -1)
        ]
        for path in candidates:
            if os.path.exists(path):
                checkpoint_path = path
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(
            "No checkpoint found! Train the model first or pass --checkpoint.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 70)
    print("ZERO-SHOT EVALUATION")
    print("=" * 70)
    df = evaluate_petshop(model, device)
    print_table(df)
    print_detailed_results(df)

    out_csv = str(RESULTS_DIR / "petshop_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
