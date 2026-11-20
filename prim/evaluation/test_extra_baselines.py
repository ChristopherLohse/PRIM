"""
Run the graph-free RCA baselines not already reported for PetShop / CausRCA in
the paper table (methods already covered there: epsilon-diagnosis, RCD, Baro,
ranked-correlation -- see table caption): RCAEval's EasyRCA, MicroCause,
CausalRCA, MSCRED, RUN. CIRCA (RCAEval's own and PetShop's HT-based one) and
PetShop's anomaly-traversal are deliberately excluded -- all three either take
or discover a causal graph, and only graph-free methods are wanted here.

TimeRank (Mehling et al., 2026) is PetShop's Table 1 "TimeRank" column. It is
NOT one of RCAEval's e2e baselines -- earlier code mistakenly wired PetShop's
TimeRank number to RCAEval's unrelated "cloudranger" baseline, which just
happens to share a rough "recency-ish" flavour but implements a different
algorithm (CloudRanger, Meng et al.) entirely. `_timerank_as_petshop_method`
instead calls the original causRCA-main implementation directly
(causrca.rca_models.unsupervised_rca_models.TimeRecency_BaselineUnsupervisedRCA),
via an adapter that converts PetShop's continuous per-timestep metrics into
the discrete (time_s, node, type) event log that class expects: for each
node, the first timestep in the abnormal window where |value - obs_mean| >
TIMERANK_Z_THRESHOLD * obs_std is treated as its "change" event, and the
known symptom/target node is given a synthetic "Alarm" event at the end of
the abnormal window (the diagnosis time). Nodes that never cross the
threshold have no change event and are therefore not ranked, same convention
as the RCAEval adapter below for methods that return fewer ranks than nodes.

CausRCA: reuses test_causrca_baselines.py's evaluate_causrca_baselines(),
_run_baseline() and MAP@3 scoring completely unchanged -- the new methods are
just extra entries in the `methods: {name: (callable, needs_graph)}` dict it
already accepts. RCAEval's e2e methods are wrapped in an adapter
(_rcaeval_as_petshop_method) that flattens the MultiIndex (node, "value",
"raw") frames test_causrca_baselines.py builds into the plain time-indexed
DataFrame RCAEval's contract expects, and wraps the ranked column names back
into PotentialRootCause objects.

PetShop: no existing script runs baselines against the real dataset
(test_petshop.py only evaluates PRIM), so this adds one, following
test_petshop.py's own scenario/issue loop and METRIC_CONFIG.

Metrics: PetShop -> top-1/top-3 recall, matching test_petshop.py. CausRCA ->
MAP@3 via test_causrca_baselines.py, Full and Sub graph modes reported
separately.
"""

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(
    0, str(Path(__file__).parent
           ))  # sibling bare imports (test_causrca_baselines, test_petshop)
sys.path.insert(
    0, str(PROJECT_ROOT / "petshop-root-cause-analysis-main" / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "causRCA-main" /
                       "src"))  # causrca.rca_models.* (TimeRank)

from rca_task import PotentialRootCause

import prim.evaluation.rcaeval_baro as re1mod  # reuses its py313 RCAEval.e2e import patch

from test_causrca_baselines import (
    RESULTS_DIR,
    evaluate_causrca_baselines,
    print_comparison_table,
    print_results,
)
from test_petshop import DATASET_DIR, SCENARIOS, METRIC_CONFIG, load_scenario, get_node_names, _bootstrap_ci

TIMERANK_Z_THRESHOLD = 2.0  # std-devs from the observational baseline to count as a "change"

# Default set: the graph-free methods not yet reported for either benchmark.
RCAEVAL_BASELINE_NAMES = [
    "easyrca", "microcause", "causalrca", "mscred", "run"
]
# baro is already reported for CausRCA in the table but not PetShop; cloudranger
# is a genuine (if unglamorous) extra RCAEval baseline, not the TimeRank stand-in
# it used to be -- see "timerank" for the real TimeRank. All three are
# selectable via --methods, not run by default (would duplicate/mislabel a
# reported row).
RCAEVAL_METHOD_CHOICES = RCAEVAL_BASELINE_NAMES + [
    "baro", "cloudranger", "timerank"
]


def _rcaeval_as_petshop_method(rcaeval_func):
    """Wrap an RCAEval.e2e function (data, inject_time, ...) -> {"ranks": [...]}
    as a PetShop-contract analyze_root_causes(graph, target_node, target_metric,
    target_statistic, normal_metrics, abnormal_metrics) -> List[PotentialRootCause].
    `graph` is ignored; these RCAEval methods discover structure themselves."""

    def analyze_root_causes(graph, target_node, target_metric,
                            target_statistic, normal_metrics,
                            abnormal_metrics):
        # Reduce to one column per node at (target_metric, target_statistic), the same
        # selection every PetShop analyze_root_causes does internally -- needed because
        # the caller may pass either an already-single-metric MultiIndex (CausRCA's
        # dummy ("value","raw") scheme from test_causrca_baselines.py's _to_baseline_df)
        # or PetShop's full, unreduced (node, metric, statistic) frame.
        nm = normal_metrics.loc[:, (slice(None), [target_metric],
                                    [target_statistic])].copy()
        am = abnormal_metrics.loc[:, (slice(None), [target_metric],
                                      [target_statistic])].copy()
        nm.columns = [c[0] for c in nm.columns]
        am.columns = [c[0] for c in am.columns]

        # normal_metrics and abnormal_metrics don't always report the same nodes at this
        # (metric, statistic) -- reindex both onto the normal-period node list (dropped
        # nodes become NaN and get mean-imputed below), matching test_petshop.py's own
        # petshop_to_model_input() convention, so vstack below has matching column counts.
        node_names = list(nm.columns)
        am = am.reindex(columns=node_names, fill_value=np.nan)
        if len(node_names) < 2:
            return []

        # copy=True: to_numpy() can otherwise return a read-only view sharing
        # the DataFrame's internal block, and the in-place NaN fill below
        # (or the wrapped method's own preprocessing) needs to write to it.
        nm_arr = nm.to_numpy(dtype=np.float64, copy=True)
        am_arr = am.to_numpy(dtype=np.float64, copy=True)
        col_means = np.nanmean(nm_arr, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for c in range(nm_arr.shape[1]):
            nm_arr[np.isnan(nm_arr[:, c]), c] = col_means[c]
            am_arr[np.isnan(am_arr[:, c]), c] = col_means[c]

        n_normal = nm_arr.shape[0]
        data = pd.DataFrame(np.vstack([nm_arr, am_arr]), columns=node_names)
        data.insert(0, "time", np.arange(len(data)))
        inject_time = n_normal

        run_args = argparse.Namespace(root_path=str(PROJECT_ROOT),
                                      data_path="")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # dataset must be a non-None string: RCAEval.io.time_series.preprocess()
                # only drops the "time" column (and constant columns) when dataset is
                # not None -- passing None would silently leave "time" as a candidate node.
                out = rcaeval_func(
                    data,
                    inject_time,
                    dataset="generic",
                    anomalies=None,
                    dk_select_useful=False,
                    sli=target_node,
                    verbose=False,
                    n_iter=len(node_names),
                    args=run_args,
                )
            ranks = out.get("ranks") or []
        except Exception:
            ranks = []

        return [
            PotentialRootCause(node=r,
                               metric=target_metric,
                               score=1.0 / (i + 1))
            for i, r in enumerate(ranks) if r in node_names
        ]

    return analyze_root_causes


def _timerank_as_petshop_method():
    """
    PetShop-contract analyze_root_causes(...) wrapping causRCA-main's own
    TimeRecency_BaselineUnsupervisedRCA (the actual TimeRank algorithm from
    Mehling et al., 2026 -- "ranks by recency of value changes"), instead of
    reimplementing the ranking logic here. See module docstring for the
    change-detection adapter this needs (PetShop is continuous per-timestep
    metrics; TimeRecency_BaselineUnsupervisedRCA expects a discrete
    (time_s, node, type) event log).
    """
    from causrca.rca_models.unsupervised_rca_models import TimeRecency_BaselineUnsupervisedRCA

    model = TimeRecency_BaselineUnsupervisedRCA()

    def analyze_root_causes(graph, target_node, target_metric,
                            target_statistic, normal_metrics,
                            abnormal_metrics):
        nm = normal_metrics.loc[:, (slice(None), [target_metric],
                                    [target_statistic])].copy()
        am = abnormal_metrics.loc[:, (slice(None), [target_metric],
                                      [target_statistic])].copy()
        nm.columns = [c[0] for c in nm.columns]
        am.columns = [c[0] for c in am.columns]

        node_names = list(nm.columns)
        am = am.reindex(columns=node_names, fill_value=np.nan)
        if len(node_names) < 2 or target_node not in node_names:
            return []

        obs_mean = nm.mean(axis=0)
        obs_std = nm.std(axis=0).replace(0, np.nan)
        am_arr = am.to_numpy(dtype=np.float64)
        n_steps = am_arr.shape[0]

        # One synthetic "change" event per node at its first threshold-crossing
        # timestep in the abnormal window; the known symptom node additionally
        # gets a synthetic "Alarm" event at the end of that window.
        rows = []
        for j, node in enumerate(node_names):
            std = obs_std.get(node)
            if not std or np.isnan(std):
                continue
            deviation = np.abs(am_arr[:, j] - obs_mean.get(node, 0.0)) / std
            crossings = np.where(deviation > TIMERANK_Z_THRESHOLD)[0]
            if len(crossings) == 0:
                continue
            rows.append({
                "time_s": float(crossings[0]),
                "node": node,
                "type": "change"
            })
        rows.append({
            "time_s": float(n_steps),
            "node": target_node,
            "type": "Alarm"
        })

        if len(rows) < 2:  # only the synthetic alarm, no detected changes
            return []

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with tempfile.NamedTemporaryFile(mode="w",
                                                 suffix=".csv",
                                                 delete=False) as f:
                    pd.DataFrame(rows).to_csv(f.name, index=False)
                    csv_path = f.name
                try:
                    ranks = model.predict(csv_path,
                                          diagnosis_time=float(n_steps),
                                          relevant_nodes=node_names)
                finally:
                    os.unlink(csv_path)
        except Exception:
            ranks = []

        return [
            PotentialRootCause(node=r,
                               metric=target_metric,
                               score=1.0 / (i + 1))
            for i, r in enumerate(ranks) if r in node_names
        ]

    return analyze_root_causes


def build_extra_methods(names=None):
    """{name: (analyze_root_causes callable, needs_graph=False)} for the requested methods."""
    names = names if names is not None else RCAEVAL_BASELINE_NAMES
    methods = {}

    for name in names:
        if name == "timerank":
            try:
                methods[name] = (_timerank_as_petshop_method(), False)
                print(f"  [ok] {name}")
            except Exception as e:
                print(
                    f"  [warn] {name} unavailable ({e!r}) -- is causRCA-main/ cloned? See README."
                )
            continue

        func = getattr(re1mod.e2e, name,
                       None) if re1mod.e2e is not None else None
        if func is None:
            print(
                f"  [warn] {name} unavailable (RCAEval.e2e failed to import: {re1mod._E2E_IMPORT_ERROR!r})"
            )
            continue
        methods[name] = (_rcaeval_as_petshop_method(func), False)
        print(f"  [ok] {name}")

    return methods


# ─────────────────────────────────────────────────────────────────────────────
# PetShop evaluation (no existing baseline runner for the real dataset)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_petshop_baselines(methods, split="test"):
    """Mirrors test_petshop.py's evaluate_petshop() loop, one row per (method, issue)."""
    rows = {name: [] for name in methods}

    for scenario in SCENARIOS:
        print(f"\n--- {scenario} ---")
        scenario_path = os.path.join(DATASET_DIR, scenario)
        _graph, normal_metrics, issues = load_scenario(
            scenario_path)  # graph unused: every method here is graph-free

        for idx, (abnormal_metrics, target) in enumerate(issues[split]):
            target_node = target["target"]["node"]
            target_metric = target["target"]["metric"]
            root_cause_node = target["root_cause"]["node"]

            metric, statistic = METRIC_CONFIG.get(target_metric,
                                                  ("latency", "p95"))
            node_names = get_node_names(normal_metrics, metric, statistic)
            if target_node not in node_names or len(node_names) < 2:
                continue

            for method_name, (method_fn, needs_graph) in methods.items():
                # All methods here are graph-free (needs_graph is always False);
                # the `graph` positional arg is unused by _rcaeval_as_petshop_method.
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = method_fn(
                            nx.DiGraph(),
                            target_node,
                            metric,
                            statistic,
                            normal_metrics,
                            abnormal_metrics,
                        )
                except Exception:
                    results = []

                results_sorted = sorted(results,
                                        key=lambda x: x.score,
                                        reverse=True) if results else []
                seen, ranked = set(), []
                for rc in results_sorted:
                    if rc.node not in seen and rc.node in node_names:
                        ranked.append(rc.node)
                        seen.add(rc.node)

                rows[method_name].append({
                    "scenario":
                    scenario,
                    "target_metric":
                    target_metric,
                    "issue_idx":
                    idx,
                    "root_cause":
                    root_cause_node,
                    "top1_recall":
                    int(root_cause_node in ranked[:1]),
                    "top3_recall":
                    int(root_cause_node in ranked[:3]),
                })

    return {name: pd.DataFrame(r) for name, r in rows.items()}


def print_petshop_table(results):
    scenario_map = {
        "low_traffic": "low",
        "high_traffic": "high",
        "temporal_traffic1": "temporal",
        "temporal_traffic2": "temporal",
    }
    print("\n" + "=" * 78)
    print(
        "PetShop -- Top-3 Recall (6 traffic/metric scenarios), new baselines")
    print("=" * 78)
    print(
        f"{'method':<14} {'top1':>8} {'top3':>8}  {'ci_lo':>8}  {'ci_hi':>8}  {'n':>5}"
    )
    print("-" * 62)
    for method_name, df in results.items():
        if df.empty:
            print(f"{method_name:<14} {'—':>8} {'—':>8}")
            continue
        df = df.copy()
        df["scenario_label"] = df["scenario"].map(scenario_map)
        lo, hi = _bootstrap_ci(df["top3_recall"].values)
        print(
            f"{method_name:<14} {df['top1_recall'].mean():>8.3f} {df['top3_recall'].mean():>8.3f}  "
            f"{lo:>8.3f}  {hi:>8.3f}  {len(df):>5}")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Run extra (not-yet-reported) baselines on PetShop and CausRCA")
    parser.add_argument("--benchmarks",
                        type=str,
                        nargs="+",
                        choices=["petshop", "causrca"],
                        default=["petshop", "causrca"])
    parser.add_argument("--methods",
                        type=str,
                        nargs="+",
                        choices=RCAEVAL_METHOD_CHOICES,
                        default=None)
    args = parser.parse_args()

    print("Loading baseline methods...")
    methods = build_extra_methods(args.methods)
    if not methods:
        print("No baseline methods available.")
        return

    if "petshop" in args.benchmarks:
        print(f"\n{'='*70}\nPETSHOP\n{'='*70}")
        results = evaluate_petshop_baselines(methods)
        print_petshop_table(results)
        for name, df in results.items():
            if not df.empty:
                df.to_csv(str(RESULTS_DIR /
                              f"petshop_extra_baseline_{name}.csv"),
                          index=False)

    if "causrca" in args.benchmarks:
        for mode in ["sub", "full"]:
            print(
                f"\n{'='*70}\nCAUSRCA  [mode={mode}, graph=expert]\n{'='*70}")
            results = evaluate_causrca_baselines(methods,
                                                 mode=mode,
                                                 graph_source="expert")
            print_comparison_table(results)
            print_results(results, title_suffix=f"  [mode={mode}]")
            for name, df in results.items():
                if not df.empty:
                    out_cols = [c for c in df.columns if not c.startswith("_")]
                    df[out_cols].to_csv(
                        str(RESULTS_DIR /
                            f"causrca_extra_baseline_{name}_{mode}.csv"),
                        index=False)


if __name__ == "__main__":
    main()
