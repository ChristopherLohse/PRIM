"""
Finetuning ablation: zero-shot vs decoder-only vs full-model finetuning.

OOD target: random DAGs (3–10 nodes) with sinusoidal structural equations
(not in the training prior), random weight-style interventions, random alarm.

Results saved to results/ablation_finetuning.csv and
results/ablation_finetuning.pdf.

Usage
─────
  uv run python prim/evaluation/ablation_finetuning.py \
      --checkpoint Finalmodels/FinalPRIMS.pt
"""

from __future__ import annotations

import argparse
import copy
import os
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

from prim.data.dag_generator_simplified import generate_dag
from prim.model.prim_neural_process import PRIMNeuralProcess as PRIM

_PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Sweep / eval settings ─────────────────────────────────────────────────────
NODE_SIZES = list(range(3, 11))  # 3–10 nodes, sampled per episode
N_OBS_FIXED = 100
INT_SIZES = [1, 2, 5, 10, 20, 50]
N_EPISODES = 200
CLIP = 10.0
_DAG_FAMILIES = ["erdos-renyi", "barabasi-albert", "bipartite"]

_CHANCE = float(np.mean([1 / n for n in NODE_SIZES]))

# ── Finetuning hyperparameters ────────────────────────────────────────────────
FT_N_EPISODES = 500
FT_LR = 5e-5
FT_EPOCHS = 10

# ── Sinusoidal equation (OOD functional form) ─────────────────────────────────


def _sin_fn(rng):
    a = float(rng.uniform(0.5, 2.0))
    b = float(rng.uniform(0.5, 3.0))
    c = float(rng.uniform(0.0, np.pi))
    return lambda x, _a=a, _b=b, _c=c: _a * np.sin(_b * x + _c)


# ── Episode generator ─────────────────────────────────────────────────────────


def _gen_episode(n_nodes, n_obs, n_int, rng, max_retries=10):
    """
    Random DAG, sinusoidal equations, random source-node intervention.
    Returns (obs, int_data, rc_node, alarm_node) or raises RuntimeError.
    """
    for _ in range(max_retries):
        try:
            edge_prob = float(rng.uniform(0.2, 0.5))
            dag_family = str(rng.choice(_DAG_FAMILIES))
            dag = generate_dag(n_nodes, edge_prob, dag_family)
            topo = list(nx.topological_sort(dag))

            sources = [nd for nd in dag.nodes if dag.in_degree(nd) == 0]
            if not sources:
                continue
            rc_node = int(rng.choice(sources))

            descendants = list(nx.descendants(dag, rc_node))
            if not descendants:
                continue
            alarm_node = int(rng.choice(descendants))

            # One sinusoidal function per edge
            edge_fns = {(u, v): _sin_fn(rng) for u, v in dag.edges}
            sigma = float(rng.uniform(0.5, 1.5))

            def _sample(n,
                        intervene=False,
                        _topo=topo,
                        _dag=dag,
                        _efns=edge_fns,
                        _sig=sigma,
                        _rc=rc_node):
                data = np.zeros((n, n_nodes), dtype=np.float32)
                for nd in _topo:
                    parents = list(_dag.predecessors(nd))
                    noise = rng.normal(0, _sig, n).astype(np.float32)
                    if not parents:
                        val = noise * (2.0 if intervene and nd == _rc else 1.0)
                    else:
                        val = noise.copy()
                        for p in parents:
                            contrib = _efns[(p,
                                             nd)](data[:,
                                                       p]).astype(np.float32)
                            if intervene and nd == _rc:
                                contrib *= 2.0
                            val += contrib
                    data[:, nd] = val
                return data

            obs = _sample(n_obs, intervene=False)
            int_data = _sample(n_int, intervene=True)
            return obs, int_data, rc_node, alarm_node

        except Exception:
            continue

    raise RuntimeError(
        f"Failed to generate episode after {max_retries} retries")


# ── Preprocessing ─────────────────────────────────────────────────────────────


def _normalize_pad(obs, int_data, max_nodes, n_nodes):
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-8
    scale = max_nodes / n_nodes
    obs_n = np.clip((obs - obs_mean) / obs_std, -CLIP, CLIP) * scale
    int_n = np.clip((int_data - obs_mean) / obs_std, -CLIP, CLIP) * scale
    obs_p = np.zeros((obs.shape[0], max_nodes), dtype=np.float32)
    int_p = np.zeros((int_data.shape[0], max_nodes), dtype=np.float32)
    obs_p[:, :n_nodes] = obs_n
    int_p[:, :n_nodes] = int_n
    return obs_p, int_p


# ── Evaluation ────────────────────────────────────────────────────────────────


def _bootstrap_ci(correct, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    boot = rng.choice(correct, size=(n_boot, len(correct)),
                      replace=True).mean(1)
    return float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def evaluate(model, device, max_nodes, n_int, rng):
    model.eval()
    correct = []
    with torch.no_grad():
        for _ in range(N_EPISODES):
            n_nodes = int(rng.choice(NODE_SIZES))
            obs, int_data, rc_node, alarm_node = _gen_episode(
                n_nodes, N_OBS_FIXED, n_int, rng)
            obs_p, int_p = _normalize_pad(obs, int_data, max_nodes, n_nodes)
            obs_t = torch.from_numpy(obs_p).unsqueeze(0).to(device)
            int_t = torch.from_numpy(int_p).unsqueeze(0).to(device)
            mask = torch.zeros(1, max_nodes, device=device)
            mask[0, alarm_node] = 1.0
            logits = model(obs_t, int_t, target_mask=mask, n_nodes=n_nodes)
            pad = torch.zeros_like(logits)
            pad[:, n_nodes:] = float("-inf")
            pred = (logits + pad).argmax(1).item()
            correct.append(int(pred == rc_node))
    return np.array(correct, dtype=bool)


def evaluate_sweep(model, device, max_nodes, seed):
    rng = np.random.default_rng(seed)
    result = {"n": INT_SIZES, "mean": [], "lo": [], "hi": []}
    for n_int in INT_SIZES:
        correct = evaluate(model, device, max_nodes, n_int, rng)
        m = correct.mean()
        lo, hi = _bootstrap_ci(correct)
        result["mean"].append(m)
        result["lo"].append(lo)
        result["hi"].append(hi)
        print(f"  n_int={n_int:>4}  acc={m:.3f}  [{lo:.3f}, {hi:.3f}]")
    return result


# ── Finetuning ────────────────────────────────────────────────────────────────


def _finetune(model, device, max_nodes, rng, freeze_backbone: bool):
    for name, p in model.named_parameters():
        p.requires_grad_(not freeze_backbone or name.startswith("decoder."))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters()
                   if not p.requires_grad)
    print(f"  Frozen: {n_frozen:,}  |  Trainable: {n_trainable:,}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=FT_LR,
                            weight_decay=0.01)
    total_steps = FT_N_EPISODES * FT_EPOCHS
    warmup_steps = max(1, total_steps // 20)

    def _lr(s):
        if s < warmup_steps:
            return s / warmup_steps
        p = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr)
    model.train()

    for epoch in range(FT_EPOCHS):
        tot_loss = n_steps = 0
        for _ in range(FT_N_EPISODES):
            n_nodes = int(rng.choice(NODE_SIZES))
            n_int_ft = int(rng.choice(INT_SIZES))
            try:
                obs, int_data, rc_node, alarm_node = _gen_episode(
                    n_nodes, N_OBS_FIXED, n_int_ft, rng)
            except RuntimeError:
                continue
            obs_p, int_p = _normalize_pad(obs, int_data, max_nodes, n_nodes)
            obs_t = torch.from_numpy(obs_p).unsqueeze(0).to(device)
            int_t = torch.from_numpy(int_p).unsqueeze(0).to(device)
            mask = torch.zeros(1, max_nodes, device=device)
            mask[0, alarm_node] = 1.0
            logits = model(obs_t, int_t, target_mask=mask, n_nodes=n_nodes)
            label = torch.tensor([rc_node], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item()
            n_steps += 1
        print(
            f"  Epoch {epoch+1}/{FT_EPOCHS}  avg_loss={tot_loss/max(n_steps,1):.4f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}")
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

_STYLES = {
    "zero_shot":
    dict(color="#2166ac", linestyle="-", marker="o", label="Zero-shot"),
    "decoder_only":
    dict(color="#1a9850", linestyle="--", marker="s", label="Decoder-only FT"),
    "full_model":
    dict(color="#d6604d", linestyle=":", marker="^", label="Full-model FT"),
}
_RCPARAMS = {
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def plot_ablation(results: dict, out_path: str):
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for condition, res in results.items():
        s = _STYLES[condition]
        ns = np.array(res["n"])
        means = np.array(res["mean"])
        lo = np.array(res["lo"])
        hi = np.array(res["hi"])
        ax.errorbar(ns,
                    means,
                    yerr=[means - lo, hi - means],
                    color=s["color"],
                    linestyle=s["linestyle"],
                    marker=s["marker"],
                    label=s["label"],
                    linewidth=2.0,
                    markersize=5,
                    capsize=3,
                    elinewidth=0.9,
                    zorder=3)
    ax.axhline(_CHANCE,
               ls=":",
               color="gray",
               lw=0.9,
               label=f"Chance ({_CHANCE:.0%})")
    ax.set_xlabel(r"$n_{\mathrm{int}}$")
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=
        "Finetuning ablation: sinusoidal OOD equations on random DAGs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv",
                        type=str,
                        default=str(RESULTS_DIR / "ablation_finetuning.csv"))
    parser.add_argument("--out_pdf",
                        type=str,
                        default=str(RESULTS_DIR / "ablation_finetuning.pdf"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  max_nodes={args.max_nodes}")
    print(
        f"OOD distribution: sinusoidal equations, random DAGs {min(NODE_SIZES)}–{max(NODE_SIZES)} nodes"
    )

    base_model = PRIM(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=args.max_nodes,
        dropout=0.0,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    base_model.load_state_dict(state)
    print(
        f"Loaded: {args.checkpoint}  ({sum(p.numel() for p in base_model.parameters()):,} params)"
    )

    results = {}

    print("\n" + "=" * 60 + "\nZERO-SHOT\n" + "=" * 60)
    results["zero_shot"] = evaluate_sweep(copy.deepcopy(base_model), device,
                                          args.max_nodes, args.seed)

    print("\n" + "=" * 60 + "\nDECODER-ONLY FINETUNING\n" + "=" * 60)
    model_dec = copy.deepcopy(base_model)
    _finetune(model_dec,
              device,
              args.max_nodes,
              np.random.default_rng(args.seed + 1000),
              freeze_backbone=True)
    results["decoder_only"] = evaluate_sweep(model_dec, device, args.max_nodes,
                                             args.seed + 1)

    print("\n" + "=" * 60 + "\nFULL-MODEL FINETUNING\n" + "=" * 60)
    model_full = copy.deepcopy(base_model)
    _finetune(model_full,
              device,
              args.max_nodes,
              np.random.default_rng(args.seed + 2000),
              freeze_backbone=False)
    results["full_model"] = evaluate_sweep(model_full, device, args.max_nodes,
                                           args.seed + 2)

    rows = [
        {
            "condition": c,
            "n_int": n,
            "mean": m,
            "lo": lo,
            "hi": hi
        } for c, res in results.items()
        for n, m, lo, hi in zip(res["n"], res["mean"], res["lo"], res["hi"])
    ]
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nResults saved to {args.out_csv}")

    print(f"\n{'Condition':<18}" + "".join(f"{n:>8}" for n in INT_SIZES))
    print("-" * (18 + 8 * len(INT_SIZES)))
    for c, res in results.items():
        print(f"{c:<18}" + "".join(f"{m:>8.3f}" for m in res["mean"]))

    plot_ablation(results, args.out_pdf)


if __name__ == "__main__":
    main()
