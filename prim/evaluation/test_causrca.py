"""
Evaluate our meta-learning model on the CausRCA digital twin benchmark.

Datasets: exp_probe, exp_coolant, exp_hydraulics
Ground truth: manipulatedVars from experiment description JSON.
Target (symptom): alarm nodes present in the run data.

For each run:
  - "Normal" data = time series before cause_start_at (obs)
  - "Fault" data  = time series from cause_start_at to diagnosis_at (int)
Both periods are discretized to a fixed time-step grid, encoded to float,
normalised with pre-fault statistics, and padded to MAX_NODES.

Primary metric: MAP@3 (mean average precision at k=3), matching the causRCA
codebase convention. Also reports Recall@1/3/5.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is on sys.path so causRCA helpers are importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
CAUSRCA_ROOT = PROJECT_ROOT / "causRCA-main"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CAUSRCA_ROOT / "src"))

from causrca.utils.discretize import __discretize_dataset_helper  # internal helper
from causrca.utils.utils import get_encoding_dict
from prim.model.prim_neural_process import PRIMNeuralProcess

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DIG_TWIN_DIR = CAUSRCA_ROOT / "data" / "dig_twin"
EXPERT_GRAPH_DIR = CAUSRCA_ROOT / "data" / "expert_graph"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "probe": DIG_TWIN_DIR / "exp_probe",
    "coolant": DIG_TWIN_DIR / "exp_coolant",
    "hydraulics": DIG_TWIN_DIR / "exp_hydraulics",
}

# MAX_NODES is set in main() from --max_nodes (default 100 for new training).
# Module-level default kept for backward-compat imports.
MAX_NODES = 100
TIME_STEP_MS = 500  # discretisation step (ms)
MIN_NORMAL_STEPS = 5  # skip runs with fewer normal time steps

# --drop_silent: strip constant-and-unchanged columns (idle subsystems in
# full-mode) so the model sees an in-distribution, sub-mode-sized problem.
DROP_SILENT = False

CATEGORICAL_ENCODING_DICT = get_encoding_dict()

# ─────────────────────────────────────────────────────────────────────────────
# MAP@K helpers  (mirrors causRCA eval helpers)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_apk(y_true, y_pred, k=3):
    if not y_true:
        return 0.0
    y_pred = y_pred[:k]
    score, hits, used = 0.0, 0, set()
    for i, p in enumerate(y_pred):
        if p in y_true and p not in used:
            hits += 1
            score += hits / (i + 1)
            used.add(p)
    return score / min(len(y_true), k)


def calculate_mapk(y_true_list, y_pred_list, k=3):
    return float(
        np.mean([
            calculate_apk(yt, yp, k)
            for yt, yp in zip(y_true_list, y_pred_list)
        ]))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_relevant_nodes(dataset_path: Path) -> list:
    """Return dataset-specific relevant node labels from *_nodes.csv (Sub mode)."""
    csv_paths = list(dataset_path.glob("*_nodes.csv"))
    if not csv_paths:
        return []
    df = pd.read_csv(csv_paths[0])
    return df["label"].tolist() if "label" in df.columns else []


def load_full_nodes() -> list:
    """Return all 92 node labels from the expert graph (Full mode)."""
    path = EXPERT_GRAPH_DIR / "all_nodes.csv"
    df = pd.read_csv(path)
    return df["label"].tolist() if "label" in df.columns else []


def load_run(run_dir: Path, exp_desc: dict) -> dict | None:
    """
    Load a single run.

    Returns a dict with:
        csv_df         - raw long-format DataFrame
        cause_start_at - float
        diagnosis_at   - float
        manipulated_vars - list[str]
        alarm_nodes    - list[str]
    Returns None if causes.json is missing or the run is unusable.
    """
    causes_path = run_dir / "causes.json"
    if not causes_path.exists():
        return None

    csv_files = list(run_dir.glob("*.csv"))
    if not csv_files:
        return None

    with open(causes_path) as f:
        causes = json.load(f)

    df = pd.read_csv(csv_files[0])

    return {
        "csv_df": df,
        "cause_start_at": float(causes["cause_start_at"]),
        "diagnosis_at": float(causes["diagnosis_at"]),
        "manipulated_vars": exp_desc.get("manipulatedVars", []),
        "alarm_nodes": exp_desc.get("alarms", []),
        "run_name": run_dir.name,
        "exp_name": exp_desc.get("exp_id", run_dir.parent.name),
    }


def collect_all_runs(dataset_path: Path) -> list[dict]:
    """Collect all runs for a dataset directory."""
    runs = []
    for exp_dir in sorted(dataset_path.glob("exp_*")):
        if not exp_dir.is_dir():
            continue
        desc_files = list(exp_dir.glob("*_description.json"))
        if not desc_files:
            continue
        with open(desc_files[0]) as f:
            exp_desc = json.load(f)
        for run_dir in sorted(exp_dir.glob("run_*")):
            if not run_dir.is_dir():
                continue
            run = load_run(run_dir, exp_desc)
            if run is not None:
                runs.append(run)
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Data preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def _discretize(df: pd.DataFrame, max_time_s: float) -> pd.DataFrame:
    """Discretise a long-format DataFrame to a (time_steps × nodes) wide frame."""
    # Cast 'value' to object to avoid pandas 2.0+ StringDtype rejecting float assignments
    df = df.copy()
    df["value"] = df["value"].astype(object)
    return __discretize_dataset_helper(
        data=df,
        time_step_ms=TIME_STEP_MS,
        max_time_s=max_time_s,
        convert_to_float=True,
        categorical_encoding_dict=CATEGORICAL_ENCODING_DICT,
        remove_constant_columns=False,
    )


def run_to_model_input(run: dict,
                       node_names: list,
                       drop_silent: bool | None = None
                       ) -> tuple[np.ndarray, np.ndarray, list] | None:
    """
    Convert a run to (obs, int_data, kept_names).

    obs         = pre-fault period  (normal)   -> (T_obs, MAX_NODES)
    int_data    = fault period      (abnormal) -> (T_int, MAX_NODES)
    kept_names  = node_names, minus the constant-and-unchanged columns when
                 ``drop_silent`` (defaults to the module DROP_SILENT flag).

    Returns None if the normal window is too short.
    """
    if drop_silent is None:
        drop_silent = DROP_SILENT
    df = run["csv_df"]
    t_fault = run["cause_start_at"]
    t_diag = run["diagnosis_at"]

    # Split raw data
    df_normal = df[df["time_s"] <= t_fault].copy()
    df_abnormal = df[(df["time_s"] > t_fault)
                     & (df["time_s"] <= t_diag)].copy()

    if df_normal.empty or df_abnormal.empty:
        return None

    # Offset abnormal time so it starts at 0 for discretisation
    t_offset = df_abnormal["time_s"].min()
    df_abnormal = df_abnormal.copy()
    df_abnormal["time_s"] -= t_offset

    duration_normal = df_normal["time_s"].max()
    duration_abnormal = df_abnormal["time_s"].max()

    if duration_normal < (TIME_STEP_MS / 1000.0) * MIN_NORMAL_STEPS:
        return None

    wide_normal = _discretize(df_normal, duration_normal)
    wide_abnormal = _discretize(df_abnormal, duration_abnormal)

    # Align to common node_names, fill missing with NaN
    wide_normal = wide_normal.reindex(columns=node_names, fill_value=np.nan)
    wide_abnormal = wide_abnormal.reindex(columns=node_names,
                                          fill_value=np.nan)

    normal_arr = wide_normal.values.astype(np.float64)
    abnormal_arr = wide_abnormal.values.astype(np.float64)

    # Impute NaNs with column mean from normal data (all-NaN columns fall back to 0)
    with np.errstate(all="ignore"):
        col_means = np.nanmean(normal_arr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for c in range(normal_arr.shape[1]):
        normal_arr[np.isnan(normal_arr[:, c]), c] = col_means[c]
        abnormal_arr[np.isnan(abnormal_arr[:, c]), c] = col_means[c]

    kept_names = list(node_names)
    if drop_silent:
        from prim.evaluation.node_filtering import active_columns

        alarm_idx = [
            node_names.index(a) for a in run.get("alarm_nodes", [])
            if a in node_names
        ]
        keep = active_columns(normal_arr, abnormal_arr, force_keep=alarm_idx)
        if keep.sum() < len(node_names):
            normal_arr = normal_arr[:, keep]
            abnormal_arr = abnormal_arr[:, keep]
            kept_names = [n for n, k in zip(node_names, keep) if k]

    n_nodes = len(kept_names)

    # Standardise using normal statistics; clip at ±10 to match training
    # preprocessing and prevent exploding values for near-zero-std nodes.
    _CLIP = 10.0
    obs_mean = normal_arr.mean(axis=0, keepdims=True)
    obs_std = normal_arr.std(axis=0, keepdims=True) + 1e-8
    obs_norm = np.clip((normal_arr - obs_mean) / obs_std, -_CLIP, _CLIP)
    int_norm = np.clip((abnormal_arr - obs_mean) / obs_std, -_CLIP, _CLIP)

    # Pad to MAX_NODES with scale factor (same as petshop)
    scale = MAX_NODES / n_nodes
    n_obs = obs_norm.shape[0]
    n_int = int_norm.shape[0]

    obs_padded = np.zeros((n_obs, MAX_NODES), dtype=np.float32)
    int_padded = np.zeros((n_int, MAX_NODES), dtype=np.float32)
    obs_padded[:, :n_nodes] = (obs_norm * scale).astype(np.float32)
    int_padded[:, :n_nodes] = (int_norm * scale).astype(np.float32)

    return obs_padded, int_padded, kept_names


# ─────────────────────────────────────────────────────────────────────────────
# Inference — single eval-mode forward pass, no context, no MC dropout
# ─────────────────────────────────────────────────────────────────────────────


def predict(model, query_obs, query_int, target_mask, n_nodes=None):
    """
    Single eval-mode forward pass.

    Args:
        query_obs:   (n_queries, n_obs, max_nodes)
        query_int:   (n_queries, n_int, max_nodes)
        target_mask: (n_queries, max_nodes)
        n_nodes:     actual number of nodes (pads beyond this are masked)

    Returns:
        probs: (n_queries, max_nodes) numpy array
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


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_causrca(model, device, mode="sub"):
    """
    Evaluate model on all CausRCA digital-twin datasets.

    Zero-shot, single eval-mode forward pass. No context, no MC dropout.

    Args:
        mode: "sub"  – dataset-specific relevant nodes (11-17 per subsystem)
              "full" – all 92 nodes from the expert knowledge graph

    Returns a DataFrame with per-run results.
    """
    assert mode in ("sub",
                    "full"), f"mode must be 'sub' or 'full', got {mode!r}"

    full_node_names = load_full_nodes() if mode == "full" else None
    rows = []

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  [mode={mode}]")
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

        all_runs = collect_all_runs(ds_path)
        print(f"  Total runs loaded: {len(all_runs)}")

        for run in all_runs:
            result = run_to_model_input(run, node_names)
            if result is None:
                print(f"  [{run['run_name']}] SKIP: insufficient data")
                continue
            query_obs, query_int, run_names = result  # run_names == node_names unless --drop_silent
            run_n = len(run_names)

            target_mask = np.zeros(MAX_NODES, dtype=np.float32)
            for alarm in run["alarm_nodes"]:
                if alarm in run_names:
                    target_mask[run_names.index(alarm)] = 1.0

            q_obs_t = torch.from_numpy(query_obs).unsqueeze(0).to(device)
            q_int_t = torch.from_numpy(query_int).unsqueeze(0).to(device)
            q_mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(device)

            probs = predict(model, q_obs_t, q_int_t, q_mask_t, n_nodes=run_n)
            probs = probs[0]  # (MAX_NODES,)

            ranked_order = np.argsort(-probs[:run_n])
            ranked = [run_names[i] for i in ranked_order]

            gt = run["manipulated_vars"]
            gt_dropped = [g for g in gt
                          if g not in run_names]  # filtered-out RCs

            def in_topk(ranked, gt_list, k):
                return int(any(g in ranked[:k] for g in gt_list))

            rows.append({
                "dataset": ds_name,
                "mode": mode,
                "exp": run["exp_name"],
                "run": run["run_name"],
                "manipulated": "|".join(gt),
                "top1_pred": ranked[0] if ranked else "",
                "rec@1": in_topk(ranked, gt, 1),
                "rec@3": in_topk(ranked, gt, 3),
                "rec@5": in_topk(ranked, gt, 5),
                "_ranked": ranked,
                "_gt": gt,
                "n_nodes": run_n,
                "gt_dropped": len(gt_dropped),
            })

            print(
                f"  {run['exp_name']}/{run['run_name']:6s}  "
                f"n={run_n:>3}  gt={gt}  pred={ranked[0] if ranked else '?'}  "
                f"rec@3={'Y' if rows[-1]['rec@3'] else 'N'}"
                f"{'  [GT DROPPED: ' + ','.join(gt_dropped) + ']' if gt_dropped else ''}"
            )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Results printing
# ─────────────────────────────────────────────────────────────────────────────


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


def print_table(df: pd.DataFrame, title_suffix: str = ""):
    """Print MAP@3 and Recall@k table per dataset and overall."""
    print("\n" + "=" * 68)
    print(
        f"CausRCA Digital Twin Benchmark  –  MAP@3 and Recall@k{title_suffix}")
    print("=" * 68)

    for k in [1, 3, 5]:
        print(f"\n--- Recall@{k} ---")
        print(
            f"{'dataset':<14} {'recall':>8}  {'ci_lo':>8}  {'ci_hi':>8}  {'n':>5}"
        )
        print("-" * 48)
        for ds in list(DATASETS.keys()) + ["OVERALL"]:
            sub = df if ds == "OVERALL" else df[df["dataset"] == ds]
            if len(sub) == 0:
                continue
            mean = sub[f"rec@{k}"].mean()
            lo, hi = _bootstrap_ci(sub[f"rec@{k}"].values)
            print(
                f"{ds:<14} {mean:>8.3f}  {lo:>8.3f}  {hi:>8.3f}  {len(sub):>5}"
            )

    print("\n--- MAP@3 ---")
    print(
        f"{'dataset':<14} {'map@3':>8}  {'ci_lo':>8}  {'ci_hi':>8}  {'n':>5}")
    print("-" * 48)
    for ds in list(DATASETS.keys()) + ["OVERALL"]:
        sub = df if ds == "OVERALL" else df[df["dataset"] == ds]
        if len(sub) == 0:
            continue
        ap_per_run = [
            calculate_apk(gt, pred, k=3)
            for gt, pred in zip(sub["_gt"].tolist(), sub["_ranked"].tolist())
        ]
        map3 = float(np.mean(ap_per_run))
        lo, hi = _bootstrap_ci(ap_per_run)
        print(f"{ds:<14} {map3:>8.3f}  {lo:>8.3f}  {hi:>8.3f}  {len(sub):>5}")


def print_comparison_table(df_sub: pd.DataFrame, df_full: pd.DataFrame):
    """Side-by-side MAP@3 comparison: Sub vs Full mode."""
    print("\n" + "=" * 70)
    print("Sub vs Full Mode Comparison  –  MAP@3")
    print("=" * 70)
    print(
        f"{'dataset':<14} {'sub MAP@3':>10} {'full MAP@3':>11} {'delta':>8}  {'n':>5}"
    )
    print("-" * 55)
    for ds in list(DATASETS.keys()) + ["OVERALL"]:
        s = df_sub if ds == "OVERALL" else df_sub[df_sub["dataset"] == ds]
        f = df_full if ds == "OVERALL" else df_full[df_full["dataset"] == ds]
        if len(s) == 0 and len(f) == 0:
            continue
        map_s = calculate_mapk(s["_gt"].tolist(), s["_ranked"].tolist(),
                               k=3) if len(s) else float("nan")
        map_f = calculate_mapk(f["_gt"].tolist(), f["_ranked"].tolist(),
                               k=3) if len(f) else float("nan")
        delta = map_f - map_s if (len(s) and len(f)) else float("nan")
        sign = "+" if delta > 0 else ""
        n = max(len(s), len(f))
        print(
            f"{ds:<14} {map_s:>10.3f} {map_f:>11.3f} {sign}{delta:>7.3f}  {n:>5}"
        )


def print_detailed(df: pd.DataFrame):
    """Per-manipulated-variable breakdown."""
    print("\n" + "=" * 60)
    print("Per Manipulated Variable")
    print("=" * 60)
    for var in sorted(df["manipulated"].unique()):
        sub = df[df["manipulated"] == var]
        map3 = calculate_mapk(sub["_gt"].tolist(),
                              sub["_ranked"].tolist(),
                              k=3)
        r1 = sub["rec@1"].mean()
        r3 = sub["rec@3"].mean()
        print(
            f"  {var:<30}  rec@1={r1:.2f}  rec@3={r3:.2f}  MAP@3={map3:.3f}  n={len(sub)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse
    global DATASETS, MAX_NODES, DROP_SILENT

    parser = argparse.ArgumentParser(
        description="Evaluate on CausRCA digital-twin benchmark")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=100,
        help="Max nodes the model was trained with (default 100)")
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        choices=["sub", "full"],
        default=["sub", "full"],
        help="Node sets to evaluate: sub=relevant nodes, full=all 92 nodes")
    parser.add_argument("--datasets",
                        type=str,
                        nargs="+",
                        choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Datasets to evaluate on")
    parser.add_argument("--drop_silent",
                        action="store_true",
                        help="Strip constant-and-unchanged columns before the "
                        "model call (idle subsystems in full-mode) so the "
                        "model sees an in-distribution, sub-mode-sized graph.")
    args = parser.parse_args()

    # Apply global overrides
    DATASETS = {k: v for k, v in DATASETS.items() if k in args.datasets}
    MAX_NODES = args.max_nodes
    DROP_SILENT = args.drop_silent
    if DROP_SILENT:
        print("drop_silent ON: constant-and-unchanged columns removed per run")

    # Sanity-check: full graph has 92 nodes, must fit in MAX_NODES
    if "full" in args.mode and MAX_NODES < 92:
        print(f"WARNING: --max_nodes {MAX_NODES} < 92 (full graph size). "
              f"Full-mode evaluation may be inaccurate. Use --max_nodes 100.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  max_nodes={MAX_NODES}")

    model = PRIMNeuralProcess(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=100,
        dropout=0.1,
    ).to(device)

    # Auto-detect checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = ["context_meta_multigpu_final.pt"]
        candidates += [
            f"context_meta_multigpu_epoch_{e}.pt" for e in range(50, 0, -1)
        ]
        for p in candidates:
            if os.path.exists(p):
                checkpoint_path = p
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No checkpoint found! Train first or pass --checkpoint <path>.")
        return

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    all_dfs = {}  # mode -> DataFrame

    for mode in args.mode:
        print(f"\n{'='*70}")
        print(f"ZERO-SHOT  |  mode={mode.upper()}")
        print(f"{'='*70}")

        df = evaluate_causrca(model, device, mode=mode)

        if df.empty:
            print("No results.")
            continue

        all_dfs[mode] = df
        print_table(df, title_suffix=f"  [ZERO-SHOT | {mode.upper()}]")
        print_detailed(df)

        out_cols = [c for c in df.columns if not c.startswith("_")]
        out_file = str(RESULTS_DIR / f"causrca_results_{mode}.csv")
        df[out_cols].to_csv(out_file, index=False)
        print(f"\nResults saved to {out_file}")

    # Side-by-side comparison when both modes were run
    if "sub" in all_dfs and "full" in all_dfs:
        print(f"\n{'='*70}")
        print("SUB vs FULL COMPARISON  |  ZERO-SHOT")
        print(f"{'='*70}")
        print_comparison_table(all_dfs["sub"], all_dfs["full"])


if __name__ == "__main__":
    main()
