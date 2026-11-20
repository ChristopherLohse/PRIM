"""
test_multi_rca.py  —  Multi-root-cause (MAP@2) evaluation for PRIM
───────────────────────────────────────────────────────────────────
Evaluates whether PRIM can rank two simultaneous root causes in its
top-2 logits (MAP@2) on the fixed DAG from the paper figure:

    X1 ──► X3 ──► X6
    │             ▲
    └────────────►│
    X2 ──► X3    │
    │      ▲     │
    └──► X4 ──►──┘
    X5 ──────────►

  Root causes : X1 (0), X2 (1)   [red]
  Mediators   : X3 (2), X4 (3)   [orange]
  Confounder  : X5 (4)            [white — not intervened]
  Alarm       : X6 (5)

Each realisation draws fresh NeuralNetEquation weights for the SAME
fixed DAG, so results test generalisation over functional forms.

Two conditions:
  zero-shot  — pretrained  Finalmodels/FinalPRIMS.pt
  finetuned  — after decoder-only finetuning on random multi-RC DAGs

Two plots (same style as test_threenode_synthetic.py):
  1. MAP@2 vs n_obs  (fixed n_int = N_INT_FIXED)
  2. MAP@2 vs n_int  (fixed n_obs = N_OBS_FIXED)

Usage
─────
  # Zero-shot only:
  uv run python prim/evaluation/test_multi_rca.py \\
      --checkpoint Finalmodels/FinalPRIMS.pt

  # Finetune then evaluate both:
  uv run python prim/evaluation/test_multi_rca.py \\
      --checkpoint Finalmodels/FinalPRIMS.pt --finetune
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from prim.data.dag_generator_simplified import (
    StructuralCausalModel,
    apply_weight_intervention,
    generate_dag,
)

from prim.model.prim_neural_process import PRIMNeuralProcess

_PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_PETSHOP_CODE = str(_PROJECT_ROOT / "petshop-root-cause-analysis-main" /
                    "code")

# ── Fixed evaluation DAG (picture graph) ──────────────────────────────────────
#   Nodes: X1=0, X2=1, X3=2, X4=3, X5=4, X6=5
#   Edges match picture exactly
_EVAL_EDGES = [(0, 2), (0, 5), (1, 2), (1, 3), (2, 5), (3, 5), (4, 5)]
EVAL_N_NODES = 6
EVAL_RC_NODES = [0, 1]  # X1, X2
EVAL_ALARM = 5  # X6


def _build_fixed_dag() -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(EVAL_N_NODES))
    dag.add_edges_from(_EVAL_EDGES)
    assert nx.is_directed_acyclic_graph(dag)
    return dag


FIXED_DAG = _build_fixed_dag()

# ── Evaluation constants ───────────────────────────────────────────────────────
N_REALISATIONS = 200  # realisations per (n_obs, n_int) point
SAMPLE_SIZES = [20, 50, 100, 200, 300]  # n_obs sweep
N_INT_FIXED = 10  # fixed n_int for obs-sweep plot
N_OBS_FIXED = 100  # fixed n_obs for int-sweep plot
INT_SIZES = [1, 2, 5, 10, 20, 50]  # n_int sweep
CLIP = 10.0

# ── Finetuning constants ───────────────────────────────────────────────────────
FT_N_ENVS = 200
FT_N_SCENARIOS = 8
FT_NODE_RANGE = (5, 10)
FT_N_OBS_RANGE = (40, 150)
FT_N_INT_RANGE = (5, 20)
FT_LR = 3e-5
FT_EPOCHS = 5
FT_K_CTX = (0, 3)
FT_N_QUERIES = 4

# ── MAP@2 ──────────────────────────────────────────────────────────────────────


def map_at_2(probs: np.ndarray, rc_nodes: list[int], n_nodes: int) -> float:
    """Average precision at 2 for one prediction.

    AP@2 = (1/|R|) Σ_{k=1}^{2} rel(k) · hits(k)/k
    where hits(k) = #{true RCs in top-k}.
    """
    top2 = np.argsort(probs[:n_nodes])[::-1][:2]
    rc_set = set(rc_nodes)
    n_rel = len(rc_set)
    if n_rel == 0:
        return 0.0
    ap, hits = 0.0, 0
    for rank, pred in enumerate(top2, 1):
        if pred in rc_set:
            hits += 1
            ap += hits / rank
    return ap / n_rel


def _bootstrap_ci(values,
                  n_boot: int = 1000,
                  alpha: float = 0.05,
                  seed: int = 0):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * alpha)), float(
        np.percentile(boot, 100 * (1 - alpha)))


# ── Preprocessing ──────────────────────────────────────────────────────────────


def to_model_input(obs: np.ndarray, int_data: np.ndarray, max_nodes: int,
                   n_nodes: int):
    """Z-normalise with obs stats, clip ±CLIP, scale by max_nodes/n_nodes, pad."""
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-8
    scale = max_nodes / n_nodes

    obs_z = np.clip((obs - obs_mean) / obs_std, -CLIP, CLIP)
    int_z = np.clip((int_data - obs_mean) / obs_std, -CLIP, CLIP)

    obs_p = np.zeros((obs.shape[0], max_nodes), dtype=np.float32)
    int_p = np.zeros((int_data.shape[0], max_nodes), dtype=np.float32)
    obs_p[:, :n_nodes] = (obs_z * scale).astype(np.float32)
    int_p[:, :n_nodes] = (int_z * scale).astype(np.float32)
    return obs_p, int_p


# ── Inference ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def _predict(model, obs_t, int_t, mask_t, n_nodes: int, device) -> np.ndarray:
    model.eval()
    max_nodes = obs_t.shape[-1]
    logits = model(obs_t.to(device),
                   int_t.to(device),
                   target_mask=mask_t.to(device),
                   n_nodes=n_nodes)
    pad = torch.zeros_like(logits)
    if n_nodes < max_nodes:
        pad[:, n_nodes:] = float("-inf")
    return torch.softmax(logits + pad, dim=-1).cpu().numpy()[0]


# ── Realisation sampler (fixed evaluation DAG) ─────────────────────────────────


def _sample_realisation(dag: nx.DiGraph,
                        rc_nodes: list[int],
                        n_obs: int,
                        n_int: int,
                        eq_type: str = "nn") -> tuple[np.ndarray, np.ndarray]:
    """
    Draw one realisation: fresh SCM (NN or GP equations) on dag, apply weight
    interventions to all RC nodes, return (obs, int_data).
    """
    scm = StructuralCausalModel(dag, equation_type=eq_type)
    int_scm = scm
    for rc in rc_nodes:
        int_scm = apply_weight_intervention(int_scm, rc, flip_signs=True)

    obs = scm.sample(n_obs)
    int_data = int_scm.sample(n_int)
    return obs, int_data


# ── Episode runners ────────────────────────────────────────────────────────────


def run_realisations(model,
                     device,
                     max_nodes: int,
                     n_obs: int,
                     n_int: int,
                     seed: int,
                     eq_type: str = "nn") -> np.ndarray:
    """
    Run N_REALISATIONS on the fixed evaluation DAG with equation type eq_type.
    Each realisation draws new SCM parameters — same graph, different functions.
    Returns array of MAP@2 scores (length N_REALISATIONS).
    """
    rng = np.random.default_rng(seed)
    scores = np.zeros(N_REALISATIONS)

    alarm_mask = np.zeros(max_nodes, dtype=np.float32)
    alarm_mask[EVAL_ALARM] = 1.0

    for ep in range(N_REALISATIONS):
        np.random.seed(int(rng.integers(0, 2**31)))
        torch.manual_seed(int(rng.integers(0, 2**31)))

        obs, int_data = _sample_realisation(FIXED_DAG,
                                            EVAL_RC_NODES,
                                            n_obs,
                                            n_int,
                                            eq_type=eq_type)
        obs_p, int_p = to_model_input(obs, int_data, max_nodes, EVAL_N_NODES)

        obs_t = torch.from_numpy(obs_p).unsqueeze(0)
        int_t = torch.from_numpy(int_p).unsqueeze(0)
        mask_t = torch.from_numpy(alarm_mask).unsqueeze(0)

        probs = _predict(model, obs_t, int_t, mask_t, EVAL_N_NODES, device)
        scores[ep] = map_at_2(probs, EVAL_RC_NODES, EVAL_N_NODES)

    return scores


# ── Sweep evaluators ──────────────────────────────────────────────────────────


def evaluate_vary_obs(model,
                      device,
                      max_nodes: int,
                      seed: int,
                      label: str = "PRIM",
                      eq_type: str = "nn") -> dict:
    res = {"n": SAMPLE_SIZES, "map_mean": [], "map_lo": [], "map_hi": []}
    print(
        f"\n{'='*60}\n{label} [{eq_type}]  —  MAP@2 vs n_obs  (n_int={N_INT_FIXED}, "
        f"{N_REALISATIONS} realisations)\n{'='*60}")
    for i, n_obs in enumerate(SAMPLE_SIZES):
        scores = run_realisations(model,
                                  device,
                                  max_nodes,
                                  n_obs,
                                  N_INT_FIXED,
                                  seed + i,
                                  eq_type=eq_type)
        m, (lo, hi) = scores.mean(), _bootstrap_ci(scores)
        res["map_mean"].append(m)
        res["map_lo"].append(lo)
        res["map_hi"].append(hi)
        print(f"  n_obs={n_obs:>4}  MAP@2={m:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    return res


def evaluate_vary_int(model,
                      device,
                      max_nodes: int,
                      seed: int,
                      label: str = "PRIM",
                      eq_type: str = "nn") -> dict:
    res = {"n": INT_SIZES, "map_mean": [], "map_lo": [], "map_hi": []}
    print(
        f"\n{'='*60}\n{label} [{eq_type}]  —  MAP@2 vs n_int  (n_obs={N_OBS_FIXED}, "
        f"{N_REALISATIONS} realisations)\n{'='*60}")
    for i, n_int in enumerate(INT_SIZES):
        scores = run_realisations(model,
                                  device,
                                  max_nodes,
                                  N_OBS_FIXED,
                                  n_int,
                                  seed + 1000 + i,
                                  eq_type=eq_type)
        m, (lo, hi) = scores.mean(), _bootstrap_ci(scores)
        res["map_mean"].append(m)
        res["map_lo"].append(lo)
        res["map_hi"].append(hi)
        print(f"  n_int={n_int:>4}  MAP@2={m:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    return res


# ── Finetuning environment (random DAGs, NN equations, 1–2 RCs) ───────────────


class _MultiRCEnv:
    """
    SCM on a 5–10 node DAG with up to 2 simultaneous weight-change interventions.
    Equation type is chosen randomly: 50% NN, 50% GP.  Used only for finetuning.
    """

    def __init__(self,
                 max_nodes: int,
                 n_scenarios: int = FT_N_SCENARIOS,
                 n_pool: int = 400):
        n = np.random.randint(*FT_NODE_RANGE)
        self.dag = generate_dag(n, np.random.uniform(0.2, 0.5), "erdos-renyi")
        self.n_nodes = len(self.dag.nodes)
        self.max_nodes = max_nodes

        # Source nodes (no parents) are the best RC candidates
        sources = [nd for nd in self.dag.nodes if self.dag.in_degree(nd) == 0]
        if len(sources) < 2:
            sources = list(self.dag.nodes)
        self._sources = sources
        self._desc = {
            nd: nx.descendants(self.dag, nd)
            for nd in self.dag.nodes
        }

        # Randomly pick equation type: 50% NN, 50% GP
        self._eq_type = "nn" if np.random.rand() < 0.5 else "gp"
        self._obs_scm = StructuralCausalModel(self.dag,
                                              equation_type=self._eq_type)
        pool = self._obs_scm.sample(n_pool)
        self._obs_mean = pool.mean(0)
        self._obs_std = pool.std(0) + 1e-8
        self._pool = pool

        # Build scenarios (each with its own interventional SCM)
        self.scenarios: list[dict] = []
        for _ in range(n_scenarios):
            rc, alarm, int_scm = self._pick_scenario()
            self.scenarios.append({
                "rc": rc,
                "alarm": alarm,
                "int_scm": int_scm
            })

    def _pick_scenario(self):
        n_faults = np.random.randint(1, 3)  # 1 or 2 RCs
        cands = list(self._sources)
        random.shuffle(cands)
        rc, excluded = [], set()
        for c in cands:
            if c in excluded:
                continue
            rc.append(c)
            excluded |= self._desc[c] | {c}
            if len(rc) >= n_faults:
                break
        if not rc:
            rc = [cands[0]]

        # Alarm: common descendant or any descendant of first RC
        common = None
        for r in rc:
            d = self._desc[r]
            common = d if common is None else common & d
        non_rc = set(self.dag.nodes) - set(rc)
        if common:
            sinks = [nd for nd in common if self.dag.out_degree(nd) == 0
                     ] or list(common)
            alarm = int(random.choice(sinks))
        elif non_rc - excluded:
            alarm = int(random.choice(list(non_rc - excluded)))
        elif non_rc:
            alarm = int(random.choice(list(non_rc)))
        else:
            alarm = rc[0]

        # Build interventional SCM
        int_scm = self._obs_scm
        for r in rc:
            int_scm = apply_weight_intervention(int_scm, r, flip_signs=True)

        return rc, alarm, int_scm

    def make_sample(self, si: int, n_obs: int, n_int: int) -> dict:
        K = self.max_nodes
        n = self.n_nodes
        sc = self.scenarios[si]

        idx = np.random.choice(len(self._pool),
                               n_obs,
                               replace=n_obs > len(self._pool))
        obs_raw = self._pool[idx]
        int_raw = sc["int_scm"].sample(n_int)

        scale = K / n
        obs_z = np.clip((obs_raw - self._obs_mean) / self._obs_std, -CLIP,
                        CLIP)
        int_z = np.clip((int_raw - self._obs_mean) / self._obs_std, -CLIP,
                        CLIP)

        obs_p = np.zeros((n_obs, K), dtype=np.float32)
        int_p = np.zeros((n_int, K), dtype=np.float32)
        obs_p[:, :n] = (obs_z * scale).astype(np.float32)
        int_p[:, :n] = (int_z * scale).astype(np.float32)

        multi_hot = np.zeros(K, dtype=np.float32)
        for r in sc["rc"]:
            multi_hot[r] = 1.0
        target_mask = np.zeros(K, dtype=np.float32)
        target_mask[sc["alarm"]] = 1.0

        return {
            "obs": obs_p,
            "int": int_p,
            "multi_hot": multi_hot,
            "target_mask": target_mask
        }


# ── Finetuning ─────────────────────────────────────────────────────────────────


def _train_step(model, optimizer, ctx_obs, ctx_int, ctx_masks, q_obs, q_int,
                q_masks, q_mhot, n_nodes, max_nodes, device):
    model.train()
    optimizer.zero_grad()
    n_q = q_obs.shape[0]

    all_obs = torch.cat([ctx_obs, q_obs], dim=0)
    all_int = torch.cat([ctx_int, q_int], dim=0)
    all_mask = torch.cat([ctx_masks, q_masks], dim=0)

    logits = model(all_obs, all_int, target_mask=all_mask,
                   n_nodes=n_nodes)[-n_q:]
    pad = torch.full((n_q, max_nodes), float("-inf"), device=device)
    pad[:, :n_nodes] = 0.0
    masked = logits + pad

    terms = []
    for qi in range(n_q):
        pos = (q_mhot[qi, :n_nodes] > 0).nonzero(as_tuple=False).squeeze(1)
        for pi in pos:
            terms.append(F.cross_entropy(masked[qi:qi + 1], pi.unsqueeze(0)))
        pos_set = set(pos.tolist())
        for ai in (q_masks[qi, :n_nodes]
                   > 0).nonzero(as_tuple=False).squeeze(1):
            if ai.item() not in pos_set:
                terms.append(F.softplus(masked[qi, ai]))

    if terms:
        loss = torch.stack(terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()
    return 0.0


def _build_episode(env: _MultiRCEnv, si: int, n_obs: int, n_int: int, device,
                   max_nodes: int):
    K = max_nodes
    k_ctx = np.random.randint(FT_K_CTX[0], FT_K_CTX[1] + 1)

    def _t(x):
        return torch.from_numpy(np.array(x, dtype=np.float32)).to(device)

    ctx_o, ctx_i, ctx_m = [], [], []
    other = [j for j in range(len(env.scenarios)) if j != si]
    for ci in random.sample(other, min(k_ctx, len(other))):
        s = env.make_sample(ci, n_obs, n_int)
        ctx_o.append(s["obs"])
        ctx_i.append(s["int"])
        ctx_m.append(s["target_mask"])

    q_o, q_i, q_mh, q_m = [], [], [], []
    for _ in range(FT_N_QUERIES):
        s = env.make_sample(si, n_obs, n_int)
        q_o.append(s["obs"])
        q_i.append(s["int"])
        q_mh.append(s["multi_hot"])
        q_m.append(s["target_mask"])

    if ctx_o:
        ctx_obs = _t(np.stack(ctx_o))
        ctx_int = _t(np.stack(ctx_i))
        ctx_masks = _t(np.stack(ctx_m))
    else:
        ctx_obs = torch.empty(0, n_obs, K, dtype=torch.float32, device=device)
        ctx_int = torch.empty(0, n_int, K, dtype=torch.float32, device=device)
        ctx_masks = torch.empty(0, K, dtype=torch.float32, device=device)

    return (ctx_obs, ctx_int, ctx_masks, _t(np.stack(q_o)), _t(np.stack(q_i)),
            _t(np.stack(q_m)), _t(np.stack(q_mh)))


def finetune_multi_rc(model,
                      device,
                      max_nodes: int,
                      lr: float = FT_LR,
                      n_epochs: int = FT_EPOCHS,
                      output_path: str = "multi_rc_finetuned.pt"):
    """Full-model finetuning on random multi-RC NN-equation DAGs."""
    for p in model.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters())
    print(f"Trainable (all params): {n_trainable:,}")

    print(f"\nGenerating {FT_N_ENVS} multi-RC NN environments …")
    envs = []
    for _ in range(FT_N_ENVS):
        try:
            envs.append(_MultiRCEnv(max_nodes))
        except Exception as e:
            print(f"  [warn] {e}")
    if not envs:
        raise RuntimeError("No environments could be built.")
    print(f"Built {len(envs)} environments.")

    all_pairs = [(env, si) for env in envs for si in range(len(env.scenarios))]
    total_steps = len(all_pairs) * n_epochs
    warmup_steps = max(1, total_steps // 20)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=lr,
                            weight_decay=0.01)

    def _lr(s):
        if s < warmup_steps:
            return s / warmup_steps
        p = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr)

    print(
        f"\n{'='*60}\nFINETUNING  (multi-RC, NN+GP equations 50/50, full model)\n{'='*60}"
    )
    print(
        f"Envs={len(envs)}  Pairs={len(all_pairs)}  LR={lr}  Epochs={n_epochs}"
    )

    for epoch in range(n_epochs):
        print(f"\n── Epoch {epoch+1}/{n_epochs} ──")
        pairs = list(all_pairs)
        random.shuffle(pairs)
        tot_loss = n_steps = 0

        for env, si in tqdm(pairs, desc=f"Epoch {epoch+1}"):
            n_obs = np.random.randint(*FT_N_OBS_RANGE)
            n_int = np.random.randint(*FT_N_INT_RANGE)
            (ctx_obs, ctx_int, ctx_masks, q_obs, q_int, q_masks,
             q_mhot) = _build_episode(env, si, n_obs, n_int, device, max_nodes)
            loss = _train_step(model,
                               optimizer,
                               ctx_obs,
                               ctx_int,
                               ctx_masks,
                               q_obs,
                               q_int,
                               q_masks,
                               q_mhot,
                               n_nodes=env.n_nodes,
                               max_nodes=max_nodes,
                               device=device)
            scheduler.step()
            tot_loss += loss
            n_steps += 1

        print(f"Epoch {epoch+1}  avg_loss={tot_loss/max(n_steps,1):.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), output_path)
    print(f"\nSaved: {output_path}")
    return model


# ── CSV helpers ───────────────────────────────────────────────────────────────


def save_results_csv(obs_results: dict, int_results: dict | None,
                     csv_path: str):
    """Save obs-sweep and optional int-sweep results to *csv_path*."""
    import pandas as pd
    rows = []
    for sweep_type, res_dict in [("obs", obs_results),
                                 ("int", int_results or {})]:
        for key, res in res_dict.items():
            if res is None:
                continue
            for n, m, lo, hi in zip(res["n"], res["map_mean"], res["map_lo"],
                                    res["map_hi"]):
                rows.append({
                    "series": key,
                    "sweep": sweep_type,
                    "n": n,
                    "mean": m,
                    "lo": lo,
                    "hi": hi
                })
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
            "map_mean": grp["mean"].tolist(),
            "map_lo": grp["lo"].tolist(),
            "map_hi": grp["hi"].tolist(),
        }
    return results


def _plot_from_csv(csv_path: str, out_obs: str, out_int: str):
    """Regenerate obs and int plots from a previously saved CSV."""
    obs = _load_series_from_csv(csv_path, "obs")
    int_ = _load_series_from_csv(csv_path, "int")

    plot_vary_obs(
        obs.get("nn_zero"),
        obs.get("gp_zero"),
        obs.get("nn_ft"),
        obs.get("gp_ft"),
        out_obs,
    )
    plot_vary_int(
        int_.get("nn_zero"),
        int_.get("gp_zero"),
        int_.get("nn_ft"),
        int_.get("gp_ft"),
        out_int,
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
_C = {
    "nn_zero": "#2166ac",  # blue   — NN zero-shot
    "nn_ft": "#d6604d",  # red    — NN finetuned
    "gp_zero": "#1a9850",  # green  — GP zero-shot
    "gp_ft": "#9970ab",  # purple — GP finetuned
}

# Random MAP@2 baseline: E[AP@2] = (1/2)(2/n + 1/(n-1))  for n nodes, 2 RCs
_RANDOM_MAP2 = 0.5 * (2 / EVAL_N_NODES + 1 / (EVAL_N_NODES - 1))


def _draw(ax, series):
    for key, res, marker, ls, lbl in series:
        if res is None:
            continue
        ns = np.array(res["n"])
        means = np.array(res["map_mean"])
        lo = np.array(res["map_lo"])
        hi = np.array(res["map_hi"])
        ax.errorbar(ns,
                    means,
                    yerr=[means - lo, hi - means],
                    fmt=marker + ls,
                    color=_C[key],
                    lw=2.0,
                    label=lbl,
                    zorder=3,
                    markersize=4,
                    capsize=2,
                    elinewidth=0.8)
    ax.axhline(_RANDOM_MAP2,
               ls=":",
               color="gray",
               lw=0.9,
               label=f"Random (MAP@2≈{_RANDOM_MAP2:.2f}, n={EVAL_N_NODES})")


def _save(ax, fig, xlabel, out_path, x_vals):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAP@2")
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log")
    ax.set_xlim(left=x_vals[0] * 0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def plot_vary_obs(res_nn_zero, res_gp_zero, res_nn_ft, res_gp_ft,
                  out_path: str):
    series = [
        ("nn_zero", res_nn_zero, "o", "-", "PRIM zero-shot (NN)"),
        ("gp_zero", res_gp_zero, "^", "-", "PRIM zero-shot (GP)"),
        ("nn_ft", res_nn_ft, "o", "--", "PRIM finetuned (NN)"),
        ("gp_ft", res_gp_ft, "^", "--", "PRIM finetuned (GP)"),
    ]
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw(ax, series)
    _save(ax, fig, r"$n_{\mathrm{obs}}$", out_path, SAMPLE_SIZES)


def plot_vary_int(res_nn_zero, res_gp_zero, res_nn_ft, res_gp_ft,
                  out_path: str):
    series = [
        ("nn_zero", res_nn_zero, "o", "-", "PRIM zero-shot (NN)"),
        ("gp_zero", res_gp_zero, "^", "-", "PRIM zero-shot (GP)"),
        ("nn_ft", res_nn_ft, "o", "--", "PRIM finetuned (NN)"),
        ("gp_ft", res_gp_ft, "^", "--", "PRIM finetuned (GP)"),
    ]
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _draw(ax, series)
    _save(ax, fig, r"$n_{\mathrm{int}}$", out_path, INT_SIZES)


# ── Baseline comparison (table only, no plots) ────────────────────────────────

# String node names for the fixed evaluation DAG
_EVAL_NODE_NAMES = [f"X{i+1}" for i in range(EVAL_N_NODES)]
# Fixed causal edges with string names
_EVAL_EDGES_STR = [(f"X{s+1}", f"X{t+1}") for s, t in _EVAL_EDGES]


def _build_baseline_methods():
    """Import petshop baseline methods. Returns dict name -> (callable, needs_graph)."""
    import importlib
    if _PETSHOP_CODE not in sys.path:
        sys.path.insert(0, _PETSHOP_CODE)

    methods = {}
    for bl_name, mod_str, fn_str in [
        ("epsilon_diag", "epsilon_diagnosis", "make_epsilon_diagnosis"),
        ("rcd", "hierarchical_rcd", "make_hierarchical_rcd"),
    ]:
        try:
            mod = importlib.import_module(mod_str)
            methods[bl_name] = (getattr(mod,
                                        fn_str)(root_cause_top_k=EVAL_N_NODES),
                                False)
            print(f"  [ok] {bl_name}")
        except Exception as e:
            print(f"  [warn] {bl_name}: {e}")

    try:
        from baseline_anomaly_traversal import make_baseline_analyze_root_causes  # type: ignore
        methods["traversal"] = (make_baseline_analyze_root_causes(), True)
        print("  [ok] traversal")
    except Exception as e:
        print(f"  [warn] traversal: {e}")

    try:
        from circa import make_circa  # type: ignore
        methods["circa"] = (make_circa(root_cause_top_k=EVAL_N_NODES), True)
        print("  [ok] circa")
    except Exception as e:
        print(f"  [warn] circa: {e}")

    # Simple reviewer-requested baselines (graph-free, numpy/pandas only).
    from prim.evaluation.simple_baselines import (
        make_gaussian_divergence_baseline,
        make_percentile_baseline,
    )
    methods["percentile95"] = (make_percentile_baseline(), False)
    methods["gaussian_kl"] = (make_gaussian_divergence_baseline(), False)
    print("  [ok] percentile95")
    print("  [ok] gaussian_kl")

    return methods


def _to_baseline_df(array: np.ndarray, node_names: list) -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples(
        [(n, "value", "raw") for n in node_names],
        names=["node", "metric", "statistic"],
    )
    return pd.DataFrame(array.astype(np.float64), columns=cols)


def _baseline_map2(ranked_names: list, rc_nodes: list[int]) -> float:
    """MAP@2 from a ranked list of string node names."""
    rc_names = {_EVAL_NODE_NAMES[r] for r in rc_nodes}
    top2 = ranked_names[:2]
    n_rel = len(rc_names)
    if n_rel == 0:
        return 0.0
    ap, hits = 0.0, 0
    for rank, pred in enumerate(top2, 1):
        if pred in rc_names:
            hits += 1
            ap += hits / rank
    return ap / n_rel


def run_baseline_realisations(
    method_fn,
    needs_graph: bool,
    n_obs: int,
    n_int: int,
    seed: int,
    eq_type: str = "nn",
    n_realisations: int = N_REALISATIONS,
) -> np.ndarray:
    """Run n_realisations of a baseline method on the fixed evaluation DAG.
    Returns array of MAP@2 scores."""
    import warnings
    rng = np.random.default_rng(seed)

    causal_graph = nx.DiGraph(_EVAL_EDGES_STR)
    for nm in _EVAL_NODE_NAMES:
        if nm not in causal_graph:
            causal_graph.add_node(nm)
    call_graph = causal_graph.reverse(
    )  # petshop methods expect call/reversed graph

    alarm_name = _EVAL_NODE_NAMES[EVAL_ALARM]
    scores = np.zeros(n_realisations)

    for ep in range(n_realisations):
        np.random.seed(int(rng.integers(0, 2**31)))
        torch.manual_seed(int(rng.integers(0, 2**31)))

        obs, int_data = _sample_realisation(FIXED_DAG,
                                            EVAL_RC_NODES,
                                            n_obs,
                                            n_int,
                                            eq_type=eq_type)
        obs_mean = obs.mean(axis=0, keepdims=True)
        obs_std = obs.std(axis=0, keepdims=True) + 1e-8
        obs_norm = (obs - obs_mean) / obs_std
        int_norm = (int_data - obs_mean) / obs_std

        normal_df = _to_baseline_df(obs_norm, _EVAL_NODE_NAMES)
        abnormal_df = _to_baseline_df(int_norm, _EVAL_NODE_NAMES)

        graph_arg = call_graph if needs_graph else nx.DiGraph(
            [(f"X{i+1}", f"X{i+2}") for i in range(EVAL_N_NODES - 1)])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = method_fn(graph_arg, alarm_name, "value", "raw",
                                    normal_df, abnormal_df)
        except Exception:
            results = []

        if results:
            results_sorted = sorted(results,
                                    key=lambda x: x.score,
                                    reverse=True)
            ranked = [
                r.node for r in results_sorted if r.node in _EVAL_NODE_NAMES
            ]
        else:
            ranked = []

        scores[ep] = _baseline_map2(ranked, EVAL_RC_NODES)

    return scores


def evaluate_baselines_vary_obs(methods: dict,
                                seed: int,
                                eq_type: str = "nn",
                                n_realisations: int = N_REALISATIONS) -> dict:
    """Evaluate all baselines across SAMPLE_SIZES (fixed n_int=N_INT_FIXED)."""
    results = {
        name: {
            "n": SAMPLE_SIZES,
            "map_mean": [],
            "map_lo": [],
            "map_hi": []
        }
        for name in methods
    }

    print(
        f"\n{'='*60}\nBASELINES [{eq_type}]  —  MAP@2 vs n_obs  (n_int={N_INT_FIXED}, "
        f"{n_realisations} realisations)\n{'='*60}")
    for i, n_obs in enumerate(SAMPLE_SIZES):
        for name, (fn, needs_graph) in methods.items():
            scores = run_baseline_realisations(fn,
                                               needs_graph,
                                               n_obs,
                                               N_INT_FIXED,
                                               seed + i,
                                               eq_type=eq_type,
                                               n_realisations=n_realisations)
            m, (lo, hi) = scores.mean(), _bootstrap_ci(scores)
            results[name]["map_mean"].append(m)
            results[name]["map_lo"].append(lo)
            results[name]["map_hi"].append(hi)
            print(
                f"  n_obs={n_obs:>4}  {name:<14}  MAP@2={m:.3f}  CI=[{lo:.3f}, {hi:.3f}]"
            )
    return results


def evaluate_baselines_vary_int(methods: dict,
                                seed: int,
                                eq_type: str = "nn",
                                n_realisations: int = N_REALISATIONS) -> dict:
    """Evaluate all baselines across INT_SIZES (fixed n_obs=N_OBS_FIXED)."""
    results = {
        name: {
            "n": INT_SIZES,
            "map_mean": [],
            "map_lo": [],
            "map_hi": []
        }
        for name in methods
    }

    print(
        f"\n{'='*60}\nBASELINES [{eq_type}]  —  MAP@2 vs n_int  (n_obs={N_OBS_FIXED}, "
        f"{n_realisations} realisations)\n{'='*60}")
    for i, n_int in enumerate(INT_SIZES):
        for name, (fn, needs_graph) in methods.items():
            scores = run_baseline_realisations(fn,
                                               needs_graph,
                                               N_OBS_FIXED,
                                               n_int,
                                               seed + 1000 + i,
                                               eq_type=eq_type,
                                               n_realisations=n_realisations)
            m, (lo, hi) = scores.mean(), _bootstrap_ci(scores)
            results[name]["map_mean"].append(m)
            results[name]["map_lo"].append(lo)
            results[name]["map_hi"].append(hi)
            print(
                f"  n_int={n_int:>4}  {name:<14}  MAP@2={m:.3f}  CI=[{lo:.3f}, {hi:.3f}]"
            )
    return results


def print_baseline_table(
    bl_obs: dict,
    bl_int: dict,
    prim_obs_zero: dict,
    prim_int_zero: dict,
    prim_obs_ft: dict | None,
    prim_int_ft: dict | None,
    eq_type: str = "nn",
):
    """Print MAP@2 comparison table: PRIM vs baselines, vary-obs and vary-int."""

    def _fmt(val):
        return f"{val:.3f}" if val is not None else "  — "

    all_methods_obs = {}
    all_methods_obs["PRIM zero-shot"] = prim_obs_zero
    if prim_obs_ft is not None:
        all_methods_obs["PRIM finetuned"] = prim_obs_ft
    all_methods_obs.update(bl_obs)

    all_methods_int = {}
    all_methods_int["PRIM zero-shot"] = prim_int_zero
    if prim_int_ft is not None:
        all_methods_int["PRIM finetuned"] = prim_int_ft
    all_methods_int.update(bl_int)

    col_w = 12
    name_w = 18

    for sweep_label, methods_dict, x_vals, x_label in [
        ("Vary n_obs (n_int=10)", all_methods_obs, SAMPLE_SIZES, "n_obs"),
        ("Vary n_int (n_obs=100)", all_methods_int, INT_SIZES, "n_int"),
    ]:
        print(f"\n{'='*70}")
        print(f"MAP@2 Comparison  —  {sweep_label}  [{eq_type}]")
        print(f"{'='*70}")
        header = f"{'Method':<{name_w}}" + "".join(f"{x:>{col_w}}"
                                                   for x in x_vals)
        print(header)
        print("-" * (name_w + col_w * len(x_vals)))
        for name, res in methods_dict.items():
            if res is None:
                continue
            row = f"{name:<{name_w}}" + "".join(f"{_fmt(m):>{col_w}}"
                                                for m in res["map_mean"])
            print(row)
        print(f"\n{'Random baseline':<{name_w}}" +
              f"{_fmt(_RANDOM_MAP2):>{col_w}}" * len(x_vals))


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-root-cause MAP@2 evaluation for PRIM")
    parser.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        help=
        "Pretrained checkpoint (e.g. Finalmodels/FinalPRIMS.pt); not required with --plot_from_csv"
    )
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Finetune on random multi-RC data before evaluating")
    parser.add_argument("--ft_output",
                        type=str,
                        default=str(_PROJECT_ROOT / "Finalmodels" /
                                    "multi_rc_finetuned.pt"))
    parser.add_argument("--ft_lr", type=float, default=FT_LR)
    parser.add_argument("--ft_epochs", type=int, default=FT_EPOCHS)
    parser.add_argument("--out_obs",
                        type=str,
                        default=str(RESULTS_DIR / "multi_rca_vary_obs.pdf"))
    parser.add_argument("--out_int",
                        type=str,
                        default=str(RESULTS_DIR / "multi_rca_vary_int.pdf"))
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RESULTS_DIR / "multi_rca_results.csv"),
        help="Path to save (or load, with --plot_from_csv) results CSV")
    parser.add_argument(
        "--plot_from_csv",
        action="store_true",
        help="Skip experiments; regenerate plots from an existing --csv file")
    parser.add_argument(
        "--font_size",
        type=int,
        default=12,
        help="Font size for axis labels, ticks, and legend (default: 12)")
    parser.add_argument(
        "--baselines_only",
        action="store_true",
        help="Skip loading/running the PRIM model entirely — only run baselines "
        "(optionally filtered by --methods) on the fixed multi-RC DAG. "
        "No checkpoint needed. Fast, CPU-only.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Restrict baselines to these method names "
        "(e.g. --methods percentile95 gaussian_kl).",
    )
    parser.add_argument(
        "--n_realisations",
        type=int,
        default=N_REALISATIONS,
        help=
        f"Realisations per point for --baselines_only (default {N_REALISATIONS}; "
        "lower for a quicker sanity check, e.g. 50).",
    )
    parser.add_argument(
        "--eq_type",
        choices=["nn", "gp"],
        default="nn",
        help="Structural-equation type for --baselines_only (default nn).",
    )
    args = parser.parse_args()

    # Apply font size before any plotting
    _RCPARAMS["font.size"] = args.font_size
    _RCPARAMS["axes.labelsize"] = args.font_size
    _RCPARAMS["xtick.labelsize"] = args.font_size
    _RCPARAMS["ytick.labelsize"] = args.font_size
    _RCPARAMS["legend.fontsize"] = max(args.font_size - 1, 8)

    if args.plot_from_csv:
        _plot_from_csv(args.csv, args.out_obs, args.out_int)
        return

    if args.baselines_only:
        # No PRIM model / checkpoint needed — CPU-only, fast. Runs the
        # (optionally filtered) baselines on the fixed multi-RC evaluation DAG.
        print("Baselines-only mode: skipping PRIM model.")
        print(f"Evaluation DAG: {EVAL_N_NODES} nodes, edges={_EVAL_EDGES}")
        print(f"RC nodes: {EVAL_RC_NODES}  |  Alarm: {EVAL_ALARM}")
        print(
            f"Realisations per point: {args.n_realisations}  |  eq_type={args.eq_type}"
        )

        methods = _build_baseline_methods()
        if args.methods:
            methods = {k: v for k, v in methods.items() if k in args.methods}
        print(f"Methods: {list(methods.keys())}")

        bl_obs = evaluate_baselines_vary_obs(
            methods,
            args.seed,
            eq_type=args.eq_type,
            n_realisations=args.n_realisations)

        print("\n" + "=" * 65)
        print("Summary: MAP@2 (baselines only)")
        print("=" * 65)
        for name, res in bl_obs.items():
            print(f"\n  {name}")
            for n, m, lo, hi in zip(res["n"], res["map_mean"], res["map_lo"],
                                    res["map_hi"]):
                print(f"    n_obs={n:>4}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

        save_results_csv(bl_obs, None, args.csv)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  max_nodes={args.max_nodes}")
    print(f"Evaluation DAG: {EVAL_N_NODES} nodes, edges={_EVAL_EDGES}")
    print(f"RC nodes: {EVAL_RC_NODES}  |  Alarm: {EVAL_ALARM}")
    print(
        f"Realisations per point: {N_REALISATIONS}  |  Bootstrap CI: 1000 samples"
    )

    # ── Load model ───────────────────────────────────────────────────────────
    model = PRIMNeuralProcess(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=10,
        dropout=0.1,
    ).to(device)
    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print(
        f"Loaded: {args.checkpoint}  ({sum(p.numel() for p in model.parameters()):,} params)"
    )

    # ── Zero-shot evaluation (NN + GP) ───────────────────────────────────────
    print("\n" + "=" * 60 + "\nZERO-SHOT EVALUATION\n" + "=" * 60)
    res_nn_zero_obs = evaluate_vary_obs(model,
                                        device,
                                        args.max_nodes,
                                        args.seed,
                                        label="PRIM zero-shot",
                                        eq_type="nn")
    res_nn_zero_int = evaluate_vary_int(model,
                                        device,
                                        args.max_nodes,
                                        args.seed,
                                        label="PRIM zero-shot",
                                        eq_type="nn")
    res_gp_zero_obs = evaluate_vary_obs(model,
                                        device,
                                        args.max_nodes,
                                        args.seed + 2000,
                                        label="PRIM zero-shot",
                                        eq_type="gp")
    res_gp_zero_int = evaluate_vary_int(model,
                                        device,
                                        args.max_nodes,
                                        args.seed + 2000,
                                        label="PRIM zero-shot",
                                        eq_type="gp")

    # ── Optional finetuning + evaluation (NN + GP) ────────────────────────────
    res_nn_ft_obs = res_nn_ft_int = None
    res_gp_ft_obs = res_gp_ft_int = None
    if args.finetune:
        model_ft = copy.deepcopy(model)
        finetune_multi_rc(model_ft,
                          device,
                          args.max_nodes,
                          lr=args.ft_lr,
                          n_epochs=args.ft_epochs,
                          output_path=args.ft_output)
        print("\n" + "=" * 60 + "\nFINETUNED EVALUATION\n" + "=" * 60)
        res_nn_ft_obs = evaluate_vary_obs(model_ft,
                                          device,
                                          args.max_nodes,
                                          args.seed,
                                          label="PRIM finetuned",
                                          eq_type="nn")
        res_nn_ft_int = evaluate_vary_int(model_ft,
                                          device,
                                          args.max_nodes,
                                          args.seed,
                                          label="PRIM finetuned",
                                          eq_type="nn")
        res_gp_ft_obs = evaluate_vary_obs(model_ft,
                                          device,
                                          args.max_nodes,
                                          args.seed + 2000,
                                          label="PRIM finetuned",
                                          eq_type="gp")
        res_gp_ft_int = evaluate_vary_int(model_ft,
                                          device,
                                          args.max_nodes,
                                          args.seed + 2000,
                                          label="PRIM finetuned",
                                          eq_type="gp")

    # ── Save results to CSV ───────────────────────────────────────────────────
    obs_results = {
        "nn_zero": res_nn_zero_obs,
        "gp_zero": res_gp_zero_obs,
        "nn_ft": res_nn_ft_obs,
        "gp_ft": res_gp_ft_obs
    }
    int_results = {
        "nn_zero": res_nn_zero_int,
        "gp_zero": res_gp_zero_int,
        "nn_ft": res_nn_ft_int,
        "gp_ft": res_gp_ft_int
    }
    save_results_csv(obs_results, int_results, args.csv)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_vary_obs(res_nn_zero_obs, res_gp_zero_obs, res_nn_ft_obs,
                  res_gp_ft_obs, args.out_obs)
    plot_vary_int(res_nn_zero_int, res_gp_zero_int, res_nn_ft_int,
                  res_gp_ft_int, args.out_int)


if __name__ == "__main__":
    main()
