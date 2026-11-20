"""
Evaluate PRIM on synthetic data with small, fixed causal structures.

Primary use (--settings confounder mediator): the three-node confounder vs.
mediator evaluation from Figure 2 of the paper (Z->X, Z->Y, X->Y confounder;
X->Z, X->Y, Z->Y mediator; X is always the root cause), sweeping n_obs and
n_int with --vary_int, for both GP and NN structural equations
(--fn_type_3node).

Also supports two additional two-node settings not featured in the paper
(--settings identifiable nonidentifiable), kept for completeness:
  - identifiable      (Peters & Bühlmann 2014, equal σ)
  - nonidentifiable    (Geiger & Heckerman 2002, InvGamma, η=2β)
"""

import argparse
import os
import sys
import warnings
from functools import partial
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from prim.model.prim_neural_process import PRIMNeuralProcess

_PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Two-node paper hyperparameters (Appendix C.2.1) ─────────────────────────
SIGMA = 1.0  # noise std (identifiable)
SIGMA_W = 1.0  # edge weight noise std
ALPHA = 3.0  # InvGamma shape (non-identifiable)
BETA = 0.5  # InvGamma scale
ETA = 2 * BETA  # = 1.0 — non-identifiability condition η=2β

WEIGHT_CHANGE_FACTOR = 3.0
N_NODES = 2  # two-node settings
N_NODES_3 = 3  # three-node settings (confounder / mediator, Dhir et al.)
SIGMA_3 = 1.0  # noise std for three-node SCMs

SAMPLE_SIZES = [20, 50, 100, 200, 300]
N_INT_SAMPLES = 10  # interventional samples per episode (default)

# Vary-interventional-samples plot
INT_SIZES = [1, 2, 5, 10, 20, 50]  # x-axis for second plot
N_OBS_FIXED = 100  # fixed obs count for second plot

# Three-node topologies (Dhir et al. Figure 4):
#   Confounder: Z→X, Z→Y, X→Y  (Z=0, X=1, Y=2)
#   Mediator:   X→Z, X→Y, Z→Y  (X=0, Z=1, Y=2)
# Int_node is uniformly random over all nodes (including Y).
# Symptom = last descendant of int_node; for both topologies the only sink
# is Y=2, so symptom is always Y regardless of which node is intervened.
# When int_node=2 the root cause IS the symptom itself.  Chance = 1/3.
MULTI_INT_NODES = [0, 1, 2]  # any node can be the root cause
MULTI_SYMPTOM_NODE = 2  # Y is always the observed symptom (only sink)

# Known true causal graphs (both 3-node topologies have the same edge set)
# Confounder: Z(X1)→X(X2), Z(X1)→Y(X3), X(X2)→Y(X3)
# Mediator:   X(X1)→Z(X2), X(X1)→Y(X3), Z(X2)→Y(X3)
TRUE_GRAPH_3NODE = [("X1", "X2"), ("X1", "X3"), ("X2", "X3")]
# 2-node: true graph is sampled internally → can only use PC
TRUE_GRAPH_2NODE = None

_PETSHOP_CODE = str(
    Path(__file__).parent.parent.parent / "petshop-root-cause-analysis-main" /
    "code")


def _to_baseline_df(data: np.ndarray, node_names):
    """Convert (n_samples, n_nodes) array to baseline MultiIndex DataFrame."""
    import pandas as pd
    cols = pd.MultiIndex.from_tuples(
        [(name, "value", "raw") for name in node_names],
        names=["node", "metric", "statistic"],
    )
    return pd.DataFrame(data.astype(np.float64), columns=cols)


def _build_baseline_methods_twonode():
    """Import baseline methods; return dict name → (callable, needs_graph).
    needs_graph=False → method is graph-free, skip PC discovery.
    """
    import importlib
    if _PETSHOP_CODE not in sys.path:
        sys.path.insert(0, _PETSHOP_CODE)

    methods = {}
    try:
        from ranked_correlation import make_ranked_correlation  # type: ignore[import]
        methods["ranked_corr"] = (
            make_ranked_correlation(filter_for_anomalous_metrics=False,
                                    root_cause_top_k=20),
            False,
        )
    except Exception as e:
        print(f"[warn] ranked_correlation: {e}")
    try:
        from counterfactual_attribution import make_counterfactual_attribution_method  # type: ignore[import]
        methods["counterfactual"] = (make_counterfactual_attribution_method(
            n_jobs=-1), True)
    except Exception as e:
        print(f"[warn] counterfactual_attribution: {e}")
    try:
        from baseline_anomaly_traversal import make_baseline_analyze_root_causes  # type: ignore[import]
        methods["traversal"] = (make_baseline_analyze_root_causes(), True)
    except Exception as e:
        print(f"[warn] baseline_anomaly_traversal: {e}")
    try:
        from circa import make_circa  # type: ignore[import]
        methods["circa"] = (make_circa(root_cause_top_k=20), True)
    except Exception as e:
        print(f"[warn] circa: {e}")
    for bl_name, mod_str, fn_str in [
        ("epsilon_diag", "epsilon_diagnosis", "make_epsilon_diagnosis"),
        ("rnd_walk", "random_walk", "make_random_walk"),
        ("hier_rcd", "hierarchical_rcd", "make_hierarchical_rcd"),
    ]:
        try:
            mod = importlib.import_module(mod_str)
            methods[bl_name] = (getattr(mod,
                                        fn_str)(root_cause_top_k=20), False)
        except Exception:
            pass

    # Simple reviewer-requested baselines (graph-free, numpy/pandas only).
    from prim.evaluation.simple_baselines import (
        make_gaussian_divergence_baseline,
        make_percentile_baseline,
    )
    methods["percentile95"] = (make_percentile_baseline(), False)
    methods["gaussian_kl"] = (make_gaussian_divergence_baseline(), False)
    return methods


# ─── Two-node data generators ────────────────────────────────────────────────


def gen_identifiable(n_obs, int_node, rng, n_int_samples=N_INT_SAMPLES):
    """Two-node identifiable SCM (σ=σ_w=1). true_graph sampled internally."""
    true_graph = int(rng.integers(1, 3))
    w = rng.normal(0, SIGMA_W)

    def sample(n, wc=1.0, int_node_=-1):
        # wc: weight-change factor applied to the intervened node's mechanism
        # Root node → scale its noise std by wc
        # Non-root node → scale the incoming edge weight by wc
        U1 = rng.normal(0, SIGMA, n)
        U2 = rng.normal(0, SIGMA, n)
        if true_graph == 1:  # X1→X2
            s1 = wc if int_node_ == 0 else 1.0
            s2 = wc if int_node_ == 1 else 1.0
            X1 = s1 * U1
            X2 = (wc * w if int_node_ == 1 else w) * X1 + s2 * U2
        else:  # X2→X1
            s1 = wc if int_node_ == 0 else 1.0
            s2 = wc if int_node_ == 1 else 1.0
            X2 = s2 * U2
            X1 = (wc * w if int_node_ == 0 else w) * X2 + s1 * U1
        return np.stack([X1, X2], axis=1)

    obs = sample(n_obs)
    int_data = sample(n_int_samples,
                      wc=WEIGHT_CHANGE_FACTOR,
                      int_node_=int_node)
    return obs, int_data


def gen_nonidentifiable(n_obs, int_node, rng, n_int_samples=N_INT_SAMPLES):
    """Two-node non-identifiable SCM (InvGamma, η=2β). true_graph sampled internally."""
    true_graph = int(rng.integers(1, 3))
    seed = int(rng.integers(0, 2**31))
    tau1_sq = stats.invgamma.rvs(a=ALPHA, scale=BETA, random_state=seed)
    tau2_sq = stats.invgamma.rvs(a=ALPHA, scale=BETA, random_state=seed + 1)
    w = rng.normal(0, sqrt(ETA * tau1_sq))

    def sample(n, wc=1.0, int_node_=-1):
        U1 = rng.normal(0, sqrt(tau1_sq), n)
        U2 = rng.normal(0, sqrt(tau2_sq), n)
        if true_graph == 1:  # G1: X2→X1
            s2 = wc if int_node_ == 1 else 1.0
            s1 = wc if int_node_ == 0 else 1.0
            X2 = s2 * U2
            X1 = (wc * w if int_node_ == 0 else w) * X2 + s1 * U1
        else:  # G2: X1→X2
            s1 = wc if int_node_ == 0 else 1.0
            s2 = wc if int_node_ == 1 else 1.0
            X1 = s1 * U1
            X2 = (wc * w if int_node_ == 1 else w) * X1 + s2 * U2
        return np.stack([X1, X2], axis=1)

    obs = sample(n_obs)
    int_data = sample(n_int_samples,
                      wc=WEIGHT_CHANGE_FACTOR,
                      int_node_=int_node)
    return obs, int_data


# ─── Three-node data generators (Dhir et al. Figure 4) ───────────────────────


def _make_fn(rng, fn_type):
    """Return a (n,)→(n,) function for one parent→child edge.

    fn_type:
      'linear' — weighted linear
      'gp'     — uses training GPEquation (correct kernel, latent var, Gamma noise)
      'nn'     — uses training NeuralNetEquation (same architecture as training)
    """
    if fn_type == "linear":
        w = rng.normal(0, 1)
        return lambda x, _w=w: _w * x

    if fn_type == "gp":
        from prim.data.dag_generator_simplified import GPEquation
        eq = GPEquation()
        eq._init_hyperparams(
        )  # fixes kernel type, lengthscale, output_scale once

        def f_gp(x, _eq=eq, scale=1.0):
            noise = np.random.normal(0, 1, len(x))
            orig = _eq.output_scale
            _eq.output_scale = orig * scale
            out = _eq.forward_batch(x[:, None], noise)
            _eq.output_scale = orig
            return out

        return f_gp

    if fn_type == "nn":
        from prim.data.dag_generator_simplified import NeuralNetEquation
        eq = NeuralNetEquation(n_parents=1)
        eq._ensure_initialized(input_dim=1)  # fixes weights once

        def f_nn(x, _eq=eq):
            xt = torch.from_numpy(x[:, None].astype(np.float32))
            out = _eq._forward_batch(xt).numpy()
            noise = np.random.normal(0, 0.1, len(x))
            return out + noise

        return f_nn

    raise ValueError(f"Unknown fn_type: {fn_type!r}")


def gen_confounder(n_obs,
                   int_node,
                   rng,
                   n_int_samples=N_INT_SAMPLES,
                   fn_type="linear"):
    """
    3-node confounder SCM (Dhir et al. Fig 4): Z→X, Z→Y, X→Y.

    Node layout: X=0, Z=1 (confounder/root), Y=2 (leaf/symptom).
    Always intervene on X (node 0). Z is a parent of X so Z does NOT shift
    in the interventional data — only X and Y change.
    fn_type: 'linear' | 'gp' | 'nn'
    """
    f_ZX = _make_fn(rng, fn_type)  # Z → X
    f_ZY = _make_fn(rng, fn_type)  # Z → Y
    f_XY = _make_fn(rng, fn_type)  # X → Y

    def _sample_obs(n):
        Z = rng.normal(0, SIGMA_3, n)
        X = f_ZX(Z) + rng.normal(0, SIGMA_3, n)
        Y = f_XY(X) + f_ZY(Z) + rng.normal(0, SIGMA_3, n)
        return np.stack([X, Z, Y], axis=1)

    def _sample_int(n):
        Z = rng.normal(0, SIGMA_3, n)
        # Weight-change: for GP scale output_scale internally; for linear/nn scale output.
        if fn_type == "gp":
            X = f_ZX(Z, scale=WEIGHT_CHANGE_FACTOR) + rng.normal(0, SIGMA_3, n)
        else:
            X = WEIGHT_CHANGE_FACTOR * f_ZX(Z) + rng.normal(0, SIGMA_3, n)
        Y = f_XY(X) + f_ZY(Z) + rng.normal(0, SIGMA_3, n)
        return np.stack([X, Z, Y], axis=1)

    obs = _sample_obs(n_obs)
    int_data = _sample_int(n_int_samples)
    return obs, int_data


def gen_mediator(n_obs,
                 int_node,
                 rng,
                 n_int_samples=N_INT_SAMPLES,
                 fn_type="linear"):
    """
    3-node mediator SCM (Dhir et al. Fig 4): X→Z, X→Y, Z→Y.

    Node layout: X=0 (root), Z=1 (mediator), Y=2 (leaf/symptom).
    Always intervene on X (node 0). All 3 nodes shift in int data.
    fn_type: 'linear' | 'gp' | 'nn'
    """
    f_XZ = _make_fn(rng, fn_type)  # X → Z
    f_XY = _make_fn(rng, fn_type)  # X → Y
    f_ZY = _make_fn(rng, fn_type)  # Z → Y

    def _sample_obs(n):
        X = rng.normal(0, SIGMA_3, n)
        Z = f_XZ(X) + rng.normal(0, SIGMA_3, n)
        Y = f_XY(X) + f_ZY(Z) + rng.normal(0, SIGMA_3, n)
        return np.stack([X, Z, Y], axis=1)

    def _sample_int(n):
        # Root node: weight-change = scale noise by WEIGHT_CHANGE_FACTOR,
        # so X's std grows ~WEIGHT_CHANGE_FACTOR×. Downstream propagated.
        X = rng.normal(0, WEIGHT_CHANGE_FACTOR * SIGMA_3, n)
        Z = f_XZ(X) + rng.normal(0, SIGMA_3, n)
        Y = f_XY(X) + f_ZY(Z) + rng.normal(0, SIGMA_3, n)
        return np.stack([X, Z, Y], axis=1)

    obs = _sample_obs(n_obs)
    int_data = _sample_int(n_int_samples)
    return obs, int_data


# ─── Preprocessing ────────────────────────────────────────────────────────────

_CLIP = 10.0


def to_model_input(obs, int_data, max_nodes, n_nodes):
    """Standardise with obs stats, clip, pad to max_nodes."""
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-8
    scale = max_nodes / n_nodes

    obs_norm = np.clip((obs - obs_mean) / obs_std, -_CLIP, _CLIP)
    int_norm = np.clip((int_data - obs_mean) / obs_std, -_CLIP, _CLIP)

    obs_padded = np.zeros((obs_norm.shape[0], max_nodes), dtype=np.float32)
    int_padded = np.zeros((int_norm.shape[0], max_nodes), dtype=np.float32)
    obs_padded[:, :n_nodes] = (obs_norm * scale).astype(np.float32)
    int_padded[:, :n_nodes] = (int_norm * scale).astype(np.float32)
    return obs_padded, int_padded


# ─── Inference ────────────────────────────────────────────────────────────────


def predict(model, query_obs, query_int, target_mask, n_nodes, device):
    model.eval()
    max_nodes = query_obs.shape[-1]
    with torch.no_grad():
        logits = model(
            query_obs.to(device),
            query_int.to(device),
            target_mask=target_mask.to(device),
            n_nodes=n_nodes,
        )
        pad_mask = torch.zeros_like(logits)
        if n_nodes < max_nodes:
            pad_mask[:, n_nodes:] = float("-inf")
        probs = torch.softmax(logits + pad_mask, dim=-1).cpu().numpy()
    return probs[0]


# ─── Neural model episode runner ─────────────────────────────────────────────


def run_episodes(
    model,
    device,
    max_nodes,
    gen_fn,
    n_episodes,
    n_obs,
    rng,
    n_nodes,
    int_nodes,
    symptom_fn,
    n_int_samples=N_INT_SAMPLES,
):
    """
    Generic episode runner for any n_nodes.

    gen_fn signature: (n_obs, int_node, rng, n_int_samples) → (obs, int_data)
    symptom_fn:        int_node → symptom_node index
    """
    correct = np.zeros(n_episodes, dtype=bool)
    confidence = np.zeros(n_episodes, dtype=float)

    for ep in range(n_episodes):
        int_node = int(rng.choice(int_nodes))
        symptom = symptom_fn(int_node)

        obs, int_data = gen_fn(n_obs, int_node, rng, n_int_samples)
        obs_pad, int_pad = to_model_input(obs, int_data, max_nodes, n_nodes)

        target_mask_np = np.zeros(max_nodes, dtype=np.float32)
        target_mask_np[symptom] = 1.0

        obs_t = torch.from_numpy(obs_pad).unsqueeze(0)
        int_t = torch.from_numpy(int_pad).unsqueeze(0)
        mask_t = torch.from_numpy(target_mask_np).unsqueeze(0)

        probs = predict(model, obs_t, int_t, mask_t, n_nodes, device)
        pred = int(np.argmax(probs[:n_nodes]))

        correct[ep] = pred == int_node
        confidence[ep] = float(probs[int_node])

    return correct, confidence


def _pc_discover_edges_n(obs_norm, node_names, rng):
    """Run PC on normalised obs and return list of oriented (src, tgt) edge tuples.

    obs_norm should already be standardised (zero mean, unit std per column).
    Edges PC cannot orient are randomly oriented and logged.
    """
    from causallearn.search.ConstraintBased.PC import pc as pc_alg

    n = obs_norm.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc_alg(obs_norm,
                    alpha=0.05,
                    indep_test="fisherz",
                    show_progress=False,
                    verbose=False)

    G = cg.G.graph  # G[i,j]=1, G[j,i]=-1 → i→j | G[i,j]=-1, G[j,i]=-1 → undirected
    edges = []
    undirected = []

    # Log skeleton (edges present regardless of orientation)
    skeleton_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if G[i, j] != 0 or G[j, i] != 0:
                skeleton_edges.append(f"{node_names[i]}—{node_names[j]}")
    # print(f"    [PC skeleton] {len(skeleton_edges)}/{n*(n-1)//2} edges: {skeleton_edges}", flush=True)

    for i in range(n):
        for j in range(n):
            if G[i, j] == 1 and G[j, i] == -1:
                edges.append((node_names[i], node_names[j]))

    for i in range(n):
        for j in range(i + 1, n):
            # causallearn encodes undirected as (-1,-1), also catch (1,1) for safety
            if (G[i, j] == -1 and G[j, i] == -1) or (G[i, j] == 1
                                                     and G[j, i] == 1):
                undirected.append((i, j))

    # print(f"    [PC orient]   {len(edges)} directed, {len(undirected)} undirected", flush=True)

    # PC pure form: cannot orient → log and randomly orient
    for i, j in undirected:
        if rng.random() < 0.5:
            chosen = (node_names[i], node_names[j])
        else:
            chosen = (node_names[j], node_names[i])
        # print(f"    [PC] undirected edge {node_names[i]}—{node_names[j]}: "
        #       f"randomly oriented as {chosen[0]}→{chosen[1]}", flush=True)
        edges.append(chosen)

    # Fallback: random chain if nothing was discovered
    if not edges:
        for i in range(n - 1):
            if rng.random() < 0.5:
                edges.append((node_names[i], node_names[i + 1]))
            else:
                edges.append((node_names[i + 1], node_names[i]))

    return edges


def run_baseline_episodes(
    method_fn,
    gen_fn,
    n_episodes,
    n_obs,
    rng,
    n_nodes,
    int_nodes,
    symptom_fn,
    n_int_samples=N_INT_SAMPLES,
    true_graph_edges=None,
    needs_graph=True,
):
    """Baseline episode runner.

    Args:
        true_graph_edges: list of (src, tgt) name pairs for the known causal graph,
            or None to discover with PC.
        needs_graph: if False, method is graph-free — skip PC discovery entirely.
    """
    import networkx as nx

    node_names = [f"X{i + 1}" for i in range(n_nodes)]
    correct = np.zeros(n_episodes, dtype=bool)

    for ep in range(n_episodes):
        int_node = int(rng.choice(int_nodes))
        symptom = symptom_fn(int_node)
        symptom_name = node_names[symptom]

        obs, int_data = gen_fn(n_obs, int_node, rng, n_int_samples)

        obs_mean = obs.mean(axis=0, keepdims=True)
        obs_std = obs.std(axis=0, keepdims=True) + 1e-8
        obs_norm = (obs - obs_mean) / obs_std
        int_norm = (int_data - obs_mean) / obs_std

        normal_df = _to_baseline_df(obs_norm, node_names)
        abnormal_df = _to_baseline_df(int_norm, node_names)

        if not needs_graph:
            # Graph-free method — dummy chain, won't be used
            causal_graph = nx.DiGraph([(node_names[i], node_names[i + 1])
                                       for i in range(n_nodes - 1)])
        elif true_graph_edges is not None:
            causal_graph = nx.DiGraph(true_graph_edges)
            for nm in node_names:
                if nm not in causal_graph:
                    causal_graph.add_node(nm)
        else:
            # PC discovery
            try:
                edges = _pc_discover_edges_n(obs_norm, node_names, rng)
            except Exception:
                edges = [(node_names[i], node_names[i + 1])
                         for i in range(n_nodes - 1)]
            causal_graph = nx.DiGraph(edges)
            for nm in node_names:
                if nm not in causal_graph:
                    causal_graph.add_node(nm)
            if not nx.is_directed_acyclic_graph(causal_graph):
                causal_graph = nx.DiGraph([(node_names[i], node_names[i + 1])
                                           for i in range(n_nodes - 1)])

        # Baseline methods expect "call graph" (reversed causal direction)
        baseline_graph = causal_graph.reverse()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = method_fn(baseline_graph, symptom_name, "value",
                                    "raw", normal_df, abnormal_df)
        except Exception:
            results = []

        if not results:
            pred = int(rng.choice(int_nodes))
        else:
            results_sorted = sorted(results,
                                    key=lambda x: x.score,
                                    reverse=True)
            pred_name = results_sorted[0].node
            if pred_name in node_names:
                pred = node_names.index(pred_name)
            else:
                pred = int(rng.choice(int_nodes))

        correct[ep] = (pred == int_node)

    return correct


def evaluate_baseline_setting(
    method_name,
    method_fn,
    gen_fn,
    setting_name,
    n_episodes,
    sample_sizes,
    seed,
    n_nodes,
    int_nodes,
    symptom_fn,
    true_graph_edges=None,
    needs_graph=True,
):
    rng = np.random.default_rng(seed)
    results = {
        "n": sample_sizes,
        "acc_mean": [],
        "acc_lo": [],
        "acc_hi": [],
        "n_nodes": n_nodes,
        "n_candidates": n_nodes,
    }
    if not needs_graph:
        tag = f"{method_name}(no-graph)"
    else:
        tag = f"{method_name}({'true' if true_graph_edges else 'PC'})"

    for n_obs in sample_sizes:
        print(
            f"  [{tag}] {setting_name} n={n_nodes:>2} n_obs={n_obs:>4} "
            f"({n_episodes} eps) ...",
            flush=True)
        correct = run_baseline_episodes(
            method_fn,
            gen_fn,
            n_episodes,
            n_obs,
            rng,
            n_nodes=n_nodes,
            int_nodes=int_nodes,
            symptom_fn=symptom_fn,
            true_graph_edges=true_graph_edges,
            needs_graph=needs_graph,
        )
        acc_mean = correct.mean()
        lo, hi = _bootstrap_ci(correct)
        results["acc_mean"].append(acc_mean)
        results["acc_lo"].append(lo)
        results["acc_hi"].append(hi)
        print(
            f"           acc={acc_mean:.3f}  [{lo:.3f}, {hi:.3f}]  "
            f"valid={int(correct.sum())}/{n_episodes}",
            flush=True)

    return results


# ─── Setting evaluators ───────────────────────────────────────────────────────


def evaluate_setting(
    model,
    device,
    max_nodes,
    gen_fn,
    setting_name,
    n_episodes,
    sample_sizes,
    seed,
    n_nodes,
    int_nodes,
    symptom_fn,
):
    rng = np.random.default_rng(seed)
    results = {
        "n": sample_sizes,
        "acc_mean": [],
        "acc_lo": [],
        "acc_hi": [],
        "n_nodes": n_nodes,
        "n_candidates": n_nodes,
    }  # model predicts over all n_nodes}

    for n_obs in sample_sizes:
        print(
            f"  [PRIM] {setting_name} n={n_nodes:>2} n_obs={n_obs:>4} "
            f"({n_episodes} eps) ...",
            flush=True)
        correct, _ = run_episodes(
            model,
            device,
            max_nodes,
            gen_fn,
            n_episodes,
            n_obs,
            rng,
            n_nodes=n_nodes,
            int_nodes=int_nodes,
            symptom_fn=symptom_fn,
        )
        acc_mean = correct.mean()
        lo, hi = _bootstrap_ci(correct)
        results["acc_mean"].append(acc_mean)
        results["acc_lo"].append(lo)
        results["acc_hi"].append(hi)
        print(
            f"           acc={acc_mean:.3f}  [{lo:.3f}, {hi:.3f}]  "
            f"valid={int(correct.sum())}/{n_episodes}",
            flush=True)

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

_C2 = {"ident_2": "#2166ac", "nonident_2": "#d6604d"}  # 2-node: blue / red
_C3 = {
    "confounder": "#1a9850",
    "mediator": "#9970ab"
}  # 3-node: green / purple


def _bootstrap_ci(correct, n_boot=500, alpha=0.05, seed=0):
    """Return (lo, hi) bootstrap percentile CI for mean accuracy."""
    rng = np.random.default_rng(seed)
    boot = rng.choice(correct, size=(n_boot, len(correct)),
                      replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * alpha)), float(
        np.percentile(boot, 100 * (1 - alpha)))


def _save_plot(ax, fig, xlabel, out_path, x_left, x_sizes):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log")
    ax.set_xlim(left=x_left)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def _draw_series(ax, series_list, color_map):
    chance_lines = set()
    for key, res, marker, ls, lw, lbl in series_list:
        if res is None:
            continue
        ns = np.array(res["n"])
        means = np.array(res["acc_mean"])
        lo = np.array(res["acc_lo"])
        hi = np.array(res["acc_hi"])
        color = color_map[key]
        alpha = 1.0 if ls == "-" else 0.75
        ax.errorbar(
            ns,
            means,
            yerr=[means - lo, hi - means],
            fmt=marker + ls,
            color=color,
            lw=lw,
            alpha=alpha,
            label=lbl,
            zorder=3,
            markersize=4,
            capsize=2,
            elinewidth=0.8,
        )
        n_cand = res.get("n_candidates", 2)
        chance = 1 / n_cand
        if chance not in chance_lines:
            ax.axhline(chance,
                       ls=":",
                       color="gray",
                       lw=0.9,
                       label=f"Chance ({chance:.0%})")
            chance_lines.add(chance)


def plot_results(
    model_ident=None,
    model_nonident=None,
    extra_baselines=None,
    out_path="twonode_eval.png",
):
    """2-node Recall@1 vs n_obs — paper style, separate file.

    extra_baselines: list of (key, res, marker, ls, lw, label) tuples.
    """
    series = [
        ("ident_2", model_ident, "o", "-", 2.0, "PRIM — identifiable"),
        ("nonident_2", model_nonident, "o", "-", 2.0,
         "PRIM— non-identifiable"),
    ]
    if extra_baselines:
        series.extend(extra_baselines)
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw_series(ax, series, _C2)
    _save_plot(ax, fig, "$n_{{obs}}$", out_path, SAMPLE_SIZES[0] * 0.8,
               SAMPLE_SIZES)


def plot_results_3node(
    model_confounder=None,
    model_mediator=None,
    extra_baselines=None,
    out_path="threenode_eval.png",
):
    """3-node confounder/mediator Recall@1 vs n_obs — separate file.

    extra_baselines: list of (key, res, marker, ls, lw, label) tuples.
    """
    series = [
        ("confounder", model_confounder, "o", "-", 2.0, "PRIM — Confounder"),
        ("mediator", model_mediator, "o", "-", 2.0, "PRIM — Mediator"),
    ]
    if extra_baselines:
        series.extend(extra_baselines)
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw_series(ax, series, _C3)
    _save_plot(ax, fig, "$n_{{obs}}$", out_path, SAMPLE_SIZES[0] * 0.8,
               SAMPLE_SIZES)


# ─── Vary-interventional-samples evaluators ───────────────────────────────────


def evaluate_setting_vary_int(
    model,
    device,
    max_nodes,
    gen_fn,
    setting_name,
    n_episodes,
    n_obs_fixed,
    int_sizes,
    seed,
    n_nodes,
    int_nodes,
    symptom_fn,
):
    """Like evaluate_setting but fixes n_obs and sweeps n_int_samples."""
    rng = np.random.default_rng(seed)
    results = {
        "n": int_sizes,
        "acc_mean": [],
        "acc_lo": [],
        "acc_hi": [],
        "n_nodes": n_nodes,
        "n_candidates": n_nodes,
    }  # model predicts over all n_nodes}

    print(f"\n{'='*60}")
    print(f"Model (vary n_int) — {setting_name}")
    print(f"{'='*60}")
    print(f"  n_obs_fixed={n_obs_fixed}  n_episodes={n_episodes}")
    print(f"  {'n_int':>6}  {'acc':>8}  {'lo':>8}  {'hi':>8}")
    print(f"  {'-'*36}")

    for n_int in int_sizes:
        correct, _ = run_episodes(
            model,
            device,
            max_nodes,
            gen_fn,
            n_episodes,
            n_obs_fixed,
            rng,
            n_nodes=n_nodes,
            int_nodes=int_nodes,
            symptom_fn=symptom_fn,
            n_int_samples=n_int,
        )
        acc_mean = correct.mean()
        lo, hi = _bootstrap_ci(correct)
        results["acc_mean"].append(acc_mean)
        results["acc_lo"].append(lo)
        results["acc_hi"].append(hi)
        print(f"  {n_int:>6}  {acc_mean:>8.3f}  {lo:>8.3f}  {hi:>8.3f}")

    return results


def plot_results_vary_int(
    model_ident=None,
    model_nonident=None,
    n_obs_fixed=N_OBS_FIXED,
    out_path="twonode_eval_vary_int.png",
):
    """2-node Recall@1 vs n_int (fixed n_obs) — separate file."""
    series = [
        ("ident_2", model_ident, "o", "-", 2.0, "PRIM — identifiable"),
        ("nonident_2", model_nonident, "o", "-", 2.0,
         "PRIM — non-identifiable"),
    ]
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw_series(ax, series, _C2)
    _save_plot(ax, fig, "$n_{{int}}$", out_path, INT_SIZES[0] * 0.8, INT_SIZES)


def plot_results_3node_vary_int(
    model_confounder=None,
    model_mediator=None,
    n_obs_fixed=N_OBS_FIXED,
    out_path="threenode_eval_vary_int.png",
):
    """3-node Recall@1 vs n_int (fixed n_obs) — separate file."""
    series = [
        ("confounder", model_confounder, "o", "-", 2.0, "PRIM — Confounder"),
        ("mediator", model_mediator, "o", "-", 2.0, "PRIM — Mediator"),
    ]
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw_series(ax, series, _C3)
    _save_plot(ax, fig, "$n_{{int}}$", out_path, INT_SIZES[0] * 0.8, INT_SIZES)


# ─── CSV helpers ─────────────────────────────────────────────────────────────


def _results_to_rows(res_dict: dict, sweep_type: str) -> list:
    """Flatten a {series_key: result_dict} mapping to a list of CSV rows."""
    rows = []
    for key, res in res_dict.items():
        if res is None:
            continue
        mean_col = "acc_mean" if "acc_mean" in res else "map_mean"
        lo_col = "acc_lo" if "acc_lo" in res else "map_lo"
        hi_col = "acc_hi" if "acc_hi" in res else "map_hi"
        for n, m, lo, hi in zip(res["n"], res[mean_col], res[lo_col],
                                res[hi_col]):
            rows.append({
                "series": key,
                "sweep": sweep_type,
                "n": n,
                "mean": m,
                "lo": lo,
                "hi": hi,
            })
    return rows


def save_results_csv(obs_results: dict, int_results: dict | None,
                     csv_path: str):
    """Save obs-sweep and optional int-sweep results to *csv_path*."""
    import pandas as pd
    rows = _results_to_rows(obs_results, "obs")
    if int_results:
        rows += _results_to_rows(int_results, "int")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


def _load_series_from_csv(csv_path: str, sweep: str) -> dict:
    """Load {series_key: result_dict} from a CSV for the given sweep type."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    sub = df[df["sweep"] == sweep]
    results = {}
    for key, grp in sub.groupby("series"):
        grp = grp.sort_values("n")
        results[key] = {
            "n": grp["n"].tolist(),
            "acc_mean": grp["mean"].tolist(),
            "acc_lo": grp["lo"].tolist(),
            "acc_hi": grp["hi"].tolist(),
        }
    return results


def _plot_from_csv(csv_path: str,
                   out_2node: str,
                   out_3node: str,
                   out_2node_int: str,
                   out_3node_int: str,
                   csv_vary_int: str | None = None):
    """Regenerate all plots from a previously saved CSV."""
    import os
    # obs-sweep
    if os.path.exists(csv_path):
        obs = _load_series_from_csv(csv_path, "obs")
    else:
        print(f"CSV not found: {csv_path}")
        return

    # Separate vary-int CSV: use explicit path if given, else derive from main CSV
    vi_csv = csv_vary_int if csv_vary_int else csv_path.replace(
        ".csv", "_vary_int.csv")
    vi_int = {}
    if os.path.exists(vi_csv):
        vi_int = _load_series_from_csv(vi_csv, "int")

    _BL_MARKERS = ["^", "D", "v", "p", "h", "*", "P", "X"]
    _BL_LS = {"_pc": "--", "_true": ":"}

    def _build_extra(results, prim_keys):
        extra = []
        for i, (key, res) in enumerate(results.items()):
            if key in prim_keys:
                continue
            base = key.replace("_pc", "").replace("_true", "")
            tag = "PC" if "_pc" in key else "true"
            ls = "--" if "_pc" in key else ":"
            extra.append(
                (base, res, _BL_MARKERS[i % 8], ls, 1.2, f"{base}({tag})"))
        return extra or None

    if any(k in obs for k in ("ident_2", "nonident_2")):
        plot_results(
            model_ident=obs.get("ident_2"),
            model_nonident=obs.get("nonident_2"),
            extra_baselines=_build_extra(obs, {"ident_2", "nonident_2"}),
            out_path=out_2node,
        )
    if any(k in obs for k in ("confounder", "mediator")):
        plot_results_3node(
            model_confounder=obs.get("confounder"),
            model_mediator=obs.get("mediator"),
            extra_baselines=_build_extra(obs, {"confounder", "mediator"}),
            out_path=out_3node,
        )
    if vi_int:
        if any(k in vi_int for k in ("ident_2", "nonident_2")):
            plot_results_vary_int(
                model_ident=vi_int.get("ident_2"),
                model_nonident=vi_int.get("nonident_2"),
                out_path=out_2node_int,
            )
        if any(k in vi_int for k in ("confounder", "mediator")):
            plot_results_3node_vary_int(
                model_confounder=vi_int.get("confounder"),
                model_mediator=vi_int.get("mediator"),
                out_path=out_3node_int,
            )


# ─── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PRIM on synthetic linear Gaussian data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=20,
        help="Episodes per sample size (default 20)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",
                        type=str,
                        default=str(RESULTS_DIR / "twonode_eval.pdf"),
                        help="Output for 2-node obs plot")
    parser.add_argument(
        "--out_3node",
        type=str,
        default=str(RESULTS_DIR / "threenode_eval.pdf"),
        help="Output for 3-node obs plot",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["identifiable", "nonidentifiable", "confounder", "mediator"],
        default=["identifiable", "nonidentifiable"],
        help="Settings to evaluate",
    )
    parser.add_argument(
        "--compare_baselines",
        action="store_true",
        help="Run petshop baselines (traversal, circa, ranked_corr, etc.)")
    parser.add_argument("--no_counterfactual",
                        action="store_true",
                        help="Skip counterfactual attribution (slow)")
    parser.add_argument(
        "--baselines_only",
        action="store_true",
        help="Skip loading/running the PRIM model entirely — only run "
        "--compare_baselines methods (optionally filtered by --methods) "
        "over --settings. No checkpoint needed. Fast, CPU-only.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Restrict --compare_baselines / --baselines_only to these method "
        "names (e.g. --methods percentile95 gaussian_kl).",
    )
    parser.add_argument("--vary_int",
                        action="store_true",
                        help="Also produce vary-int plots")
    parser.add_argument("--n_obs_fixed", type=int, default=N_OBS_FIXED)
    parser.add_argument("--out_vary_int",
                        type=str,
                        default=str(RESULTS_DIR / "twonode_eval_vary_int.pdf"))
    parser.add_argument("--out_3node_vary_int",
                        type=str,
                        default=str(RESULTS_DIR /
                                    "threenode_eval_vary_int.pdf"))
    parser.add_argument(
        "--fn_type_3node",
        choices=["linear", "gp", "nn"],
        # default gp and nn
        default="gp",
        help="Structural equation type for 3-node generators (linear|gp|nn)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RESULTS_DIR / "twonode_results.csv"),
        help="Path to save (or load, with --plot_from_csv) results CSV",
    )
    parser.add_argument(
        "--csv_vary_int",
        type=str,
        default=None,
        help=
        "Path to vary-int results CSV for --plot_from_csv (default: <csv>_vary_int.csv)",
    )
    parser.add_argument(
        "--plot_from_csv",
        action="store_true",
        help="Skip experiments; regenerate plots from an existing --csv file",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=12,
        help="Font size for axis labels, ticks, and legend (default: 12)",
    )
    args = parser.parse_args()

    # Apply font size before any plotting
    _RCPARAMS["font.size"] = args.font_size
    _RCPARAMS["axes.labelsize"] = args.font_size
    _RCPARAMS["xtick.labelsize"] = args.font_size
    _RCPARAMS["ytick.labelsize"] = args.font_size
    _RCPARAMS["legend.fontsize"] = max(args.font_size - 1, 8)

    if args.plot_from_csv:
        _plot_from_csv(args.csv,
                       args.out,
                       args.out_3node,
                       args.out_vary_int,
                       args.out_3node_vary_int,
                       csv_vary_int=args.csv_vary_int)
        return

    if args.baselines_only:
        # No PRIM model needed — CPU-only, fast. Use this to add/compare new
        # baselines (e.g. percentile95, gaussian_kl) for a subset of --settings
        # (e.g. confounder + mediator) without re-running the neural model,
        # whose numbers are already fixed by --seed.
        print("Baselines-only mode: skipping PRIM model.")
        _gen_confounder = partial(gen_confounder, fn_type=args.fn_type_3node)
        _gen_mediator = partial(gen_mediator, fn_type=args.fn_type_3node)

        two_node_spec = dict(n_nodes=N_NODES,
                             int_nodes=list(range(N_NODES)),
                             symptom_fn=lambda _: 1)
        confounder_spec = dict(n_nodes=N_NODES_3,
                               int_nodes=[0],
                               symptom_fn=lambda _: MULTI_SYMPTOM_NODE)
        mediator_spec = dict(n_nodes=N_NODES_3,
                             int_nodes=[0],
                             symptom_fn=lambda _: MULTI_SYMPTOM_NODE)
        _seeds = {
            "identifiable": 0,
            "nonidentifiable": 1,
            "confounder": 2,
            "mediator": 3
        }

        _run_settings = []
        if "identifiable" in args.settings:
            _run_settings.append(
                ("ident_2", gen_identifiable, two_node_spec, "identifiable"))
        if "nonidentifiable" in args.settings:
            _run_settings.append(("nonident_2", gen_nonidentifiable,
                                  two_node_spec, "nonidentifiable"))
        if "confounder" in args.settings:
            _run_settings.append(
                ("confounder", _gen_confounder, confounder_spec, "confounder"))
        if "mediator" in args.settings:
            _run_settings.append(
                ("mediator", _gen_mediator, mediator_spec, "mediator"))

        baseline_methods = _build_baseline_methods_twonode()
        if args.methods:
            baseline_methods = {
                k: v
                for k, v in baseline_methods.items() if k in args.methods
            }
        print(f"Methods: {list(baseline_methods.keys())}")

        pr = {}
        for bl_name, (bl_fn, needs_g) in baseline_methods.items():
            pr[bl_name] = {}
            for setting, gen_fn, spec, seeds_key in _run_settings:
                pr[bl_name][setting] = evaluate_baseline_setting(
                    bl_name,
                    bl_fn,
                    gen_fn,
                    setting,
                    args.n_episodes,
                    SAMPLE_SIZES,
                    args.seed + _seeds[seeds_key],
                    true_graph_edges=None,
                    needs_graph=needs_g,
                    **spec,
                )

        print("\n" + "=" * 65)
        print("Summary: Recall@1 (baselines only)")
        print("=" * 65)
        for bl_name, d in pr.items():
            for setting, res in d.items():
                print(f"\n  {bl_name}/{setting}")
                for n, m, lo, hi in zip(res["n"], res["acc_mean"],
                                        res["acc_lo"], res["acc_hi"]):
                    print(f"    n={n:>4}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

        all_obs = {f"{m}/{k}": v for m, d in pr.items() for k, v in d.items()}
        save_results_csv(all_obs, None, args.csv)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  max_nodes={args.max_nodes}")

    model = PRIMNeuralProcess(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=10,
        dropout=0.1,
    ).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = ["context_meta_multigpu_final.pt"]
        candidates += [
            f"context_meta_multigpu_epoch_{e}.pt" for e in range(59, 0, -1)
        ]
        for p in candidates:
            if os.path.exists(p):
                checkpoint_path = p
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No checkpoint found — pass --checkpoint <path>.")
        return

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded: {checkpoint_path}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    n_ep = args.n_episodes

    # Wrap 3-node generators with the chosen structural equation type
    _gen_confounder = partial(gen_confounder, fn_type=args.fn_type_3node)
    _gen_mediator = partial(gen_mediator, fn_type=args.fn_type_3node)

    two_node_spec = dict(
        n_nodes=N_NODES,
        int_nodes=list(range(N_NODES)),
        symptom_fn=lambda _: 1
    )  # symptom fixed at node 1; root cause can be 0 or 1 → chance=50%
    # Both 3-node settings use X=0, Z=1, Y=2. Always intervene on X (node 0).
    #   Confounder: Z(1) is root/confounder → Z does NOT shift when X intervened.
    #               Only X and Y change in int data.
    #   Mediator:   X(0) is root → Z and Y both shift downstream.
    #               All 3 columns change in int data.
    # Model predicts over all 3 nodes → chance = 1/3.
    confounder_spec = dict(n_nodes=N_NODES_3,
                           int_nodes=[0],
                           symptom_fn=lambda _: MULTI_SYMPTOM_NODE)
    mediator_spec = dict(n_nodes=N_NODES_3,
                         int_nodes=[0],
                         symptom_fn=lambda _: MULTI_SYMPTOM_NODE)

    mr = {}  # model results
    pr = {}  # baseline results: method_name → setting_key → result_dict

    # ── Neural model ──────────────────────────────────────────────────────────
    # Each setting gets a unique seed offset so identical generators can't
    # produce the same RNG stream (confounder vs mediator bug fix).
    _seeds = {
        "identifiable": 0,
        "nonidentifiable": 1,
        "confounder": 2,
        "mediator": 3
    }

    if "identifiable" in args.settings:
        mr["ident_2"] = evaluate_setting(
            model,
            device,
            args.max_nodes,
            gen_identifiable,
            "2-node identifiable (Peters & Bühlmann 2014)",
            n_ep,
            SAMPLE_SIZES,
            args.seed + _seeds["identifiable"],
            **two_node_spec,
        )
    if "nonidentifiable" in args.settings:
        mr["nonident_2"] = evaluate_setting(
            model,
            device,
            args.max_nodes,
            gen_nonidentifiable,
            f"2-node non-identifiable (α={ALPHA}, β={BETA}, η={ETA})",
            n_ep,
            SAMPLE_SIZES,
            args.seed + _seeds["nonidentifiable"],
            **two_node_spec,
        )
    if "confounder" in args.settings:
        mr["confounder"] = evaluate_setting(
            model,
            device,
            args.max_nodes,
            _gen_confounder,
            "3-node confounder (Z→X, Z→Y, X→Y)",
            n_ep,
            SAMPLE_SIZES,
            args.seed + _seeds["confounder"],
            **confounder_spec,
        )
    if "mediator" in args.settings:
        mr["mediator"] = evaluate_setting(
            model,
            device,
            args.max_nodes,
            _gen_mediator,
            "3-node mediator (X→Z, X→Y, Z→Y)",
            n_ep,
            SAMPLE_SIZES,
            args.seed + _seeds["mediator"],
            **mediator_spec,
        )

    # ── Baseline methods ─────────────────────────────────────────────────────
    if args.compare_baselines:
        baseline_methods = _build_baseline_methods_twonode()
        if args.no_counterfactual:
            baseline_methods.pop("counterfactual", None)
        if args.methods:
            baseline_methods = {
                k: v
                for k, v in baseline_methods.items() if k in args.methods
            }
        print(f"\nBaseline methods available: {list(baseline_methods.keys())}")

        # Markers cycling through non-circle/square shapes for baselines
        _bl_markers = ["^", "D", "v", "p", "h", "*", "P", "X"]
        for bl_idx, (bl_name,
                     (bl_fn, needs_g)) in enumerate(baseline_methods.items()):
            pr[bl_name] = {}
            mk = _bl_markers[bl_idx % len(_bl_markers)]
            seed_offset = 2000 + bl_idx * 100

            _run_settings = []
            if "identifiable" in args.settings:
                _run_settings.append(
                    ("ident_2", gen_identifiable, two_node_spec,
                     TRUE_GRAPH_2NODE, "identifiable"))
            if "nonidentifiable" in args.settings:
                _run_settings.append(
                    ("nonident_2", gen_nonidentifiable, two_node_spec,
                     TRUE_GRAPH_2NODE, "nonidentifiable"))
            if "confounder" in args.settings:
                _run_settings.append(
                    ("confounder", _gen_confounder, confounder_spec,
                     TRUE_GRAPH_3NODE, "confounder"))
            if "mediator" in args.settings:
                _run_settings.append(("mediator", _gen_mediator, mediator_spec,
                                      TRUE_GRAPH_3NODE, "mediator"))

            for setting, gen_fn, spec, true_edges, seeds_key in _run_settings:

                base_seed = args.seed + _seeds[seeds_key] + seed_offset

                # PC variant
                pr[bl_name][f"{setting}_pc"] = evaluate_baseline_setting(
                    bl_name,
                    bl_fn,
                    gen_fn,
                    f"{setting} [{bl_name}/PC]",
                    n_ep,
                    SAMPLE_SIZES,
                    base_seed,
                    true_graph_edges=None,
                    needs_graph=needs_g,
                    **spec,
                )

                # True-graph variant (3-node only; 2-node true graph is unknown;
                # skip for graph-free methods — result is identical to PC variant)
                if true_edges is not None and needs_g:
                    pr[bl_name][f"{setting}_true"] = evaluate_baseline_setting(
                        bl_name,
                        bl_fn,
                        gen_fn,
                        f"{setting} [{bl_name}/true]",
                        n_ep,
                        SAMPLE_SIZES,
                        base_seed + 50,
                        true_graph_edges=true_edges,
                        needs_graph=needs_g,
                        **spec,
                    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Summary: Recall@1")
    print("=" * 65)
    all_named = {
        **mr,
        **{
            f"{m}/{k}": v
            for m, d in pr.items()
            for k, v in d.items()
        },
    }
    for key, res in all_named.items():
        print(f"\n  {key}")
        for n, m, lo, hi in zip(res["n"], res["acc_mean"], res["acc_lo"],
                                res["acc_hi"]):
            print(f"    n={n:>4}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

    # Build extra_baselines series for 2-node and 3-node plots
    _bl_markers = ["^", "D", "v", "p", "h", "*", "P", "X"]
    _extra_2node = []
    _extra_3node = []
    for bl_idx, (bl_name, bl_d) in enumerate(pr.items()):
        mk = _bl_markers[bl_idx % len(_bl_markers)]
        for sk, res in bl_d.items():
            ls = "--" if "_pc" in sk else ":"
            lw = 1.2
            # Determine color key (strip _pc/_true suffix)
            color_key = sk.replace("_pc", "").replace("_true", "")
            tag = "PC" if "_pc" in sk else "true"
            lbl = f"{bl_name}({tag}) — {color_key}"
            entry = (color_key, res, mk, ls, lw, lbl)
            if color_key in ("ident_2", "nonident_2"):
                _extra_2node.append(entry)
            elif color_key in ("confounder", "mediator"):
                _extra_3node.append(entry)

    # ── Save results to CSV ───────────────────────────────────────────────────
    all_obs = {
        **mr,
        **{
            f"{m}/{k}": v
            for m, d in pr.items()
            for k, v in d.items()
        }
    }
    save_results_csv(all_obs, None, args.csv)

    # ── Obs-sweep plots ───────────────────────────────────────────────────────
    if mr.get("ident_2") or mr.get("nonident_2") or _extra_2node:
        plot_results(
            model_ident=mr.get("ident_2"),
            model_nonident=mr.get("nonident_2"),
            extra_baselines=_extra_2node or None,
            out_path=args.out,
        )
    if mr.get("confounder") or mr.get("mediator") or _extra_3node:
        plot_results_3node(
            model_confounder=mr.get("confounder"),
            model_mediator=mr.get("mediator"),
            extra_baselines=_extra_3node or None,
            out_path=args.out_3node,
        )

    # ── Vary-int plots (optional) ─────────────────────────────────────────────
    if args.vary_int:
        vi_mr = {}
        if "identifiable" in args.settings:
            vi_mr["ident_2"] = evaluate_setting_vary_int(
                model,
                device,
                args.max_nodes,
                gen_identifiable,
                "2-node identifiable",
                n_ep,
                args.n_obs_fixed,
                INT_SIZES,
                args.seed + 10 + _seeds["identifiable"],
                **two_node_spec,
            )
        if "nonidentifiable" in args.settings:
            vi_mr["nonident_2"] = evaluate_setting_vary_int(
                model,
                device,
                args.max_nodes,
                gen_nonidentifiable,
                "2-node non-identifiable",
                n_ep,
                args.n_obs_fixed,
                INT_SIZES,
                args.seed + 10 + _seeds["nonidentifiable"],
                **two_node_spec,
            )
        if "confounder" in args.settings:
            vi_mr["confounder"] = evaluate_setting_vary_int(
                model,
                device,
                args.max_nodes,
                _gen_confounder,
                "3-node confounder",
                n_ep,
                args.n_obs_fixed,
                INT_SIZES,
                args.seed + 10 + _seeds["confounder"],
                **confounder_spec,
            )
        if "mediator" in args.settings:
            vi_mr["mediator"] = evaluate_setting_vary_int(
                model,
                device,
                args.max_nodes,
                _gen_mediator,
                "3-node mediator",
                n_ep,
                args.n_obs_fixed,
                INT_SIZES,
                args.seed + 10 + _seeds["mediator"],
                **mediator_spec,
            )

        # Append vary-int results to the same CSV
        save_results_csv({}, vi_mr, args.csv.replace(".csv", "_vary_int.csv"))

        if vi_mr.get("ident_2") or vi_mr.get("nonident_2"):
            plot_results_vary_int(
                model_ident=vi_mr.get("ident_2"),
                model_nonident=vi_mr.get("nonident_2"),
                n_obs_fixed=args.n_obs_fixed,
                out_path=args.out_vary_int,
            )
        if vi_mr.get("confounder") or vi_mr.get("mediator"):
            plot_results_3node_vary_int(
                model_confounder=vi_mr.get("confounder"),
                model_mediator=vi_mr.get("mediator"),
                n_obs_fixed=args.n_obs_fixed,
                out_path=args.out_3node_vary_int,
            )


if __name__ == "__main__":
    main()
