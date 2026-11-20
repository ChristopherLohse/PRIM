"""
Finetune the meta-learning model on synthetic CausRCA-like data,
with an integrated CausRCA benchmark runner.

Usage
─────
  # Finetune only:
  python finetune_causrca.py --resume checkpoint.pt --output finetuned.pt

  # Finetune then benchmark:
  python finetune_causrca.py --resume checkpoint.pt --output finetuned.pt --eval

  # Benchmark only (no finetuning):
  python finetune_causrca.py --resume checkpoint.pt --eval_only

Time-node encoding
──────────────────
Instead of adding temporal positional encodings inside the model (which
would require retraining), we add ONE extra data column at position n_nodes
that holds a linear time ramp.  For a window of length T:
    value[t] = CLIP_VAL × (K / n_actual) × (2t / (T-1) - 1)   ∈ [-A, +A]
where A matches the amplitude of a fully-saturated fault signal.

Key property: mean(ramp) = 0 for both obs and int, so
    diff = mean(int_emb) - mean(obs_emb)  ≈ 0  for the time node.
The time column therefore does NOT bias the output logit toward any node,
but it IS visible in every transformer attention layer, giving the model a
temporal coordinate that helps it rank "earliest changed node = RC" over
"later changed node = alarm."

The model is called with n_nodes_model = n_actual + 1 (so it sees the time
column) but predictions are masked to positions 0 … n_actual-1 (actual nodes).

Architecture insight recap (originally written against the old
mean-pool + explicit-diff model/prim.py; this file now runs on
PRIMNeuralProcess's attentive QueryPooling readout instead, see
model/prim_neural_process.py). The zero-mean argument below no longer
gives a literal guarantee under attention pooling (there's no explicit
diff to keep unbiased), but the time column is still exposed to every
attention layer, so the model can still learn to use it as a temporal
coordinate. Re-verify this after finetuning with the new architecture.
───────────────────────────────────────────────────────────
    diff = mean(int_emb, dim=time) - mean(obs_emb, dim=time)  # old model only
    logits = decoder(diff)

After z-norm + ±CLIP clipping every changed binary node looks identical in
magnitude.  With no temporal encoding the model cannot distinguish RC (changed
at t=0) from alarm (changed at t=delay) once both have been clipped to ±CLIP.
The time node fixes this by providing a coordinate at each step.

Probe training design
──────────────────────
  • Node range (25–40) matches sub-evaluation (~34 nodes) → same K/n scale.
  • Binary fraction 0.35: probe faults are mostly continuous measurement
    deviations (position, offset) with binary alarm flags cascading later.
  • Fault window n_int ∈ [5, 20], delay_per_hop ∈ [3, 8]: the cascade (alarm)
    fires WITHIN the window.  RC is active for all n_int steps; alarm fires at
    delay d → mean(int)_RC = 10, mean(int)_alarm = 10×(n_int−d)/n_int < 10.
  • The time node (linear ramp) provides a temporal coordinate: the model
    can see "this node changed early" vs "this node changed late."
  • Alarm penalty (λ=0.40): pushes alarm logit down to break the alarm shortcut.
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from prim.data.dag_generator_simplified import generate_dag


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


# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────

NODE_RANGE = {
    "probe": (25, 40),  # matches sub-eval ~34 nodes → same K/n scale
    "coolant": (12, 18),
    "hydraulics": (14, 20),
}

# Fault-window lengths (time steps).
N_INT_RANGE = {
    "probe": (5, 20),
    "coolant": (10, 60),
    "hydraulics": (8, 50),
    "combined": (10, 60),
    "full_scale": (5, 20),  # FullScaleEnvironment uses sub_env.mode for lookup
}
N_OBS_RANGE = (40, 150)

CLIP_VAL = 10.0  # z-score clip — matches test_causrca.py
FAULT_MAGNITUDE_SIGMA = 15.0  # RC z-score before clipping  (clips to CLIP_VAL)
ATTENUATION = 0.30  # magnitude multiplier per hop for continuous nodes

# Probe: short delay so alarm fires within the window → temporal gradient visible.
# RC is active all n_int steps; alarm fires at delay d → mean(int)_RC > mean(int)_alarm.
PROBE_DELAY_PER_HOP_RANGE = (3, 8)
CONT_DELAY_PER_HOP_RANGE = (2, 8)

BINARY_NOISE_SCALE = 0.005
CONT_NOISE_SCALE_RANGE = (0.03, 0.12)

ALARM_PENALTY_WEIGHT = 0.40  # weight of alarm-avoidance term in the loss

# ─────────────────────────────────────────────────────────────────────────────
# Time-node helpers
# ─────────────────────────────────────────────────────────────────────────────


def _time_ramp(n_steps: int, amplitude: float) -> np.ndarray:
    """Linear ramp from -amplitude to +amplitude over n_steps."""
    if n_steps == 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(-amplitude, amplitude, n_steps, dtype=np.float32)


def _insert_time_node(obs_padded: np.ndarray, int_padded: np.ndarray,
                      n_actual: int, K: int) -> None:
    """
    Write the time ramp into column n_actual (in-place).
    Amplitude = CLIP_VAL × K / n_actual  — matches a fully-saturated fault node.
    Mean of the ramp is 0, so diff = mean(int) - mean(obs) ≈ 0 for this column.
    """
    amplitude = CLIP_VAL * K / n_actual
    obs_padded[:, n_actual] = _time_ramp(obs_padded.shape[0], amplitude)
    int_padded[:, n_actual] = _time_ramp(int_padded.shape[0], amplitude)


# ─────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────────────────────


def _pick_root_cause_and_alarm(dag: nx.DiGraph, n_faults: int) -> tuple:
    """
    Pick 1–3 independent root causes and one alarm (leaf of first RC's subtree).
    Each additional RC is chosen to be outside all previously picked subtrees.
    """
    valid_rc = [n for n in dag.nodes if len(list(nx.descendants(dag, n))) >= 1]
    if not valid_rc:
        valid_rc = list(dag.nodes)

    rc1 = random.choice(valid_rc)
    desc1 = nx.descendants(dag, rc1)
    leaves = [d for d in desc1 if dag.out_degree(d) == 0]
    alarm = random.choice(leaves if leaves else list(desc1)) if desc1 else rc1

    rc_list = [rc1]
    excluded = desc1 | {rc1, alarm}

    for _ in range(n_faults - 1):
        independent = [n for n in valid_rc if n not in excluded]
        if not independent:
            break
        rc_new = random.choice(independent)
        desc_new = nx.descendants(dag, rc_new)
        excluded |= desc_new | {rc_new}
        rc_list.append(rc_new)

    return rc_list, alarm


# ─────────────────────────────────────────────────────────────────────────────
# CausRCAStyleEnvironment
# ─────────────────────────────────────────────────────────────────────────────


class CausRCAStyleEnvironment:
    """
    Synthetic OPC-UA-style causal machinery data with temporal propagation.

    Samples include a time-node at column n_nodes with a linear ramp
    [-A, +A] (A = CLIP_VAL × K / n_nodes).  The model receives
    n_nodes + 1 positions so the time column is visible to attention.
    Predictions are masked to positions 0 … n_nodes-1 (actual nodes).
    """

    def __init__(self,
                 mode: str,
                 max_nodes: int,
                 n_settings: int = 8,
                 n_normal_pool: int = 600):
        assert mode in ("probe", "coolant", "hydraulics")
        self.mode = mode
        self.max_nodes = max_nodes

        lo, hi = NODE_RANGE[mode]
        self.n_nodes = np.random.randint(lo, hi + 1)

        graph_type = ("bipartite" if mode == "probe" else random.choice(
            ["erdos-renyi", "barabasi-albert"]))
        edge_prob = np.random.uniform(0.20, 0.50)
        self.dag = generate_dag(self.n_nodes, edge_prob, graph_type)

        # Probe: mostly continuous (position/offset measurements) + some binary flags.
        # Real probe faults manifest as measurement deviations (continuous), not pure binary flips.
        # full_scale: mixed, mirrors the full 92-node graph which spans all subsystems.
        binary_probs = {"probe": 0.35, "coolant": 0.15, "hydraulics": 0.40}
        bp = binary_probs[mode]
        self.var_types = [
            "binary" if np.random.rand() < bp else "continuous"
            for _ in range(self.n_nodes)
        ]

        self.baselines = np.array([
            float(np.random.choice([0.0, 1.0]))
            if t == "binary" else np.random.uniform(-3.0, 3.0)
            for t in self.var_types
        ])
        self.noise_scales = np.array([
            BINARY_NOISE_SCALE if t == "binary" else np.random.uniform(
                *CONT_NOISE_SCALE_RANGE) for t in self.var_types
        ])
        self.fault_signs = np.where(
            np.array(self.var_types) == "binary",
            1.0,
            np.random.choice([-1.0, 1.0], size=self.n_nodes),
        )

        # Precompute shortest-path distances (avoids nx calls in hot loop).
        self._dist_from: dict[int, dict[int, int]] = {}
        for src in range(self.n_nodes):
            self._dist_from[src] = dict(
                nx.single_source_shortest_path_length(self.dag, src))

        self.normal_data = self._generate_normal(n_normal_pool)
        self.obs_mean = self.normal_data.mean(axis=0)
        self.obs_std = self.normal_data.std(axis=0) + 1e-8
        self.rc_magnitudes = self.obs_std * FAULT_MAGNITUDE_SIGMA * self.fault_signs

        self.scenarios: list[dict] = []
        for _ in range(n_settings):
            # 70% single, 20% dual, 10% triple — mirrors paper fault distribution
            r = np.random.rand()
            n_faults = 1 if r < 0.70 else (2 if r < 0.90 else 3)
            rc_idxs, alarm_idx = _pick_root_cause_and_alarm(self.dag, n_faults)
            delay = np.random.randint(*(PROBE_DELAY_PER_HOP_RANGE if mode ==
                                        "probe" else CONT_DELAY_PER_HOP_RANGE))
            self.scenarios.append({
                "root_cause_idxs": rc_idxs,
                "alarm_idx": alarm_idx,
                "delay_per_hop": delay,
            })

    # ── data generation ───────────────────────────────────────────────────────

    def _generate_normal(self, n_steps: int) -> np.ndarray:
        data = np.zeros((n_steps, self.n_nodes))
        for i in range(self.n_nodes):
            ns = self.noise_scales[i]
            if self.var_types[i] == "binary":
                data[:, i] = self.baselines[i] + np.random.normal(
                    0.0, ns, n_steps)
            else:
                val = self.baselines[i]
                noise = np.random.normal(0.0, ns, n_steps)
                for t in range(n_steps):
                    val = self.baselines[i] + 0.95 * (
                        val - self.baselines[i]) + noise[t]
                    data[t, i] = val
        return data

    def _generate_fault(self, scenario_idx: int, n_steps: int) -> np.ndarray:
        sc = self.scenarios[scenario_idx]
        delay_per_hop = sc["delay_per_hop"]
        rc_idxs = sc["root_cause_idxs"]
        rc_set = set(rc_idxs)

        # Precompute: for each node, the earliest step at which it fires
        # and the associated magnitude + cascade probability.
        node_info: list = []  # (fires_at, cascade_prob, magnitude) or None
        for i in range(self.n_nodes):
            if self.var_types[i] == "binary":
                node_info.append(None)
                continue
            best_d = None
            for rc in rc_idxs:
                d = self._dist_from[rc].get(i)
                if d is not None and (best_d is None or d < best_d):
                    best_d = d
            if best_d is None:
                node_info.append(None)
            else:
                node_info.append((
                    best_d * delay_per_hop,
                    0.70**best_d,
                    self.rc_magnitudes[i] * (ATTENUATION**best_d),
                ))

        data = np.zeros((n_steps, self.n_nodes))
        for i in range(self.n_nodes):
            ns = self.noise_scales[i]
            noise = np.random.normal(0.0, ns, n_steps)

            if self.var_types[i] == "binary":
                if i in rc_set:
                    # RC flips immediately at step 0
                    data[:, i] = (1.0 - self.baselines[i]) + noise
                else:
                    # Binary cascade: flip after delay proportional to hop distance
                    best_d = None
                    for rc in rc_idxs:
                        d = self._dist_from[rc].get(i)
                        if d is not None and (best_d is None or d < best_d):
                            best_d = d
                    if best_d is not None:
                        fires_at = best_d * delay_per_hop
                        cascade_prob = 0.75**best_d
                        if fires_at < n_steps and np.random.rand(
                        ) < cascade_prob:
                            for t in range(n_steps):
                                val = (
                                    1.0 - self.baselines[i]
                                ) if t >= fires_at else self.baselines[i]
                                data[t, i] = val + noise[t]
                        else:
                            data[:, i] = self.baselines[i] + noise
                    else:
                        data[:, i] = self.baselines[i] + noise
            else:
                info = node_info[i]
                if info is None:
                    data[:, i] = self.baselines[i] + noise
                else:
                    fires_at, prob, mag = info
                    if np.random.rand() < prob:
                        for t in range(n_steps):
                            data[t, i] = (self.baselines[i] +
                                          (mag if t >= fires_at else 0.0) +
                                          noise[t])
                    else:
                        data[:, i] = self.baselines[i] + noise
        return data

    # ── sample construction ───────────────────────────────────────────────────

    def _sample_window(self, arr: np.ndarray, n: int) -> np.ndarray:
        T = len(arr)
        if T <= n:
            return arr[np.random.choice(T, n, replace=True)]
        if np.random.rand() < 0.5:
            s = np.random.randint(0, T - n + 1)
            return arr[s:s + n]
        return arr[np.sort(np.random.choice(T, n, replace=False))]

    def make_sample(self,
                    scenario_idx: int,
                    n_obs: int,
                    n_int: int,
                    use_null: bool = False) -> dict:
        """
        Returns obs/int arrays of shape (T, K) with:
          • columns 0…n_nodes-1 : z-normalised actual data
          • column  n_nodes      : time ramp (the time node)
          • columns n_nodes+1…K-1: zero padding
        n_nodes_with_time = n_nodes + 1 is stored in the dict for the caller.
        """
        K = self.max_nodes
        n = self.n_nodes
        sc = self.scenarios[scenario_idx]

        obs_raw = self._sample_window(self.normal_data, n_obs)

        if use_null:
            int_raw = self._sample_window(self.normal_data, n_int)
            multi_hot = np.zeros(K, dtype=np.float32)
        else:
            int_raw = self._generate_fault(scenario_idx, n_int)
            multi_hot = np.zeros(K, dtype=np.float32)
            for idx in sc["root_cause_idxs"]:
                multi_hot[idx] = 1.0

        obs_z = np.clip((obs_raw - self.obs_mean) / self.obs_std, -CLIP_VAL,
                        CLIP_VAL)
        int_z = np.clip((int_raw - self.obs_mean) / self.obs_std, -CLIP_VAL,
                        CLIP_VAL)

        scale = K / n
        obs_padded = np.zeros((n_obs, K), dtype=np.float32)
        int_padded = np.zeros((n_int, K), dtype=np.float32)
        obs_padded[:, :n] = (obs_z * scale).astype(np.float32)
        int_padded[:, :n] = (int_z * scale).astype(np.float32)

        _insert_time_node(obs_padded, int_padded, n, K)

        target_mask = np.zeros(K, dtype=np.float32)
        target_mask[sc["alarm_idx"]] = 1.0

        return {
            "obs": obs_padded,
            "int": int_padded,
            "multi_hot": multi_hot,
            "target_mask": target_mask,
            "n_nodes_with_time": n + 1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MultiSubsystemEnvironment
# ─────────────────────────────────────────────────────────────────────────────


class MultiSubsystemEnvironment:
    """
    Combined probe + coolant + hydraulics environment (~51–78 nodes).
    Fault is injected in exactly one subsystem; the others stay silent.
    One time node at position n_combined (after all subsystem nodes).
    """

    mode = "combined"

    def __init__(self,
                 max_nodes: int,
                 n_settings: int = 8,
                 n_normal_pool: int = 600):
        self.max_nodes = max_nodes
        self.sub = {
            "probe":
            CausRCAStyleEnvironment("probe", max_nodes, n_settings,
                                    n_normal_pool),
            "coolant":
            CausRCAStyleEnvironment("coolant", max_nodes, n_settings,
                                    n_normal_pool),
            "hydraulics":
            CausRCAStyleEnvironment("hydraulics", max_nodes, n_settings,
                                    n_normal_pool),
        }

        self.offsets: dict[str, int] = {}
        cursor = 0
        for name, env in self.sub.items():
            self.offsets[name] = cursor
            cursor += env.n_nodes
        self.n_nodes = cursor  # total actual nodes (without time)

        assert self.n_nodes + 1 <= max_nodes, (
            f"Combined nodes+time ({self.n_nodes + 1}) > max_nodes ({max_nodes})"
        )

        fault_order = [
            "probe", "coolant", "hydraulics", "probe", "coolant", "hydraulics",
            "coolant", "hydraulics"
        ]
        self.scenarios: list[dict] = []
        for si in range(n_settings):
            faulty = fault_order[si % len(fault_order)]
            fenv = self.sub[faulty]
            sub_si = si % len(fenv.scenarios)
            sub_sc = fenv.scenarios[sub_si]
            offset = self.offsets[faulty]
            self.scenarios.append({
                "faulty":
                faulty,
                "sub_si":
                sub_si,
                "root_cause_idxs":
                [r + offset for r in sub_sc["root_cause_idxs"]],
                "alarm_idx":
                sub_sc["alarm_idx"] + offset,
            })

    def make_sample(self,
                    scenario_idx: int,
                    n_obs: int,
                    n_int: int,
                    use_null: bool = False) -> dict:
        K = self.max_nodes
        sc = self.scenarios[scenario_idx]

        obs_combined = np.zeros((n_obs, K), dtype=np.float32)
        int_combined = np.zeros((n_int, K), dtype=np.float32)

        for name, env in self.sub.items():
            offset = self.offsets[name]
            n = env.n_nodes
            sub_si = scenario_idx % len(env.scenarios)
            is_faulty = (name == sc["faulty"]) and not use_null
            s = env.make_sample(sub_si, n_obs, n_int, use_null=(not is_faulty))
            # Copy actual node data (columns 0…n-1) — skip time column
            obs_combined[:, offset:offset + n] = s["obs"][:, :n]
            int_combined[:, offset:offset + n] = s["int"][:, :n]

        multi_hot = np.zeros(K, dtype=np.float32)
        if not use_null:
            for idx in sc["root_cause_idxs"]:
                multi_hot[idx] = 1.0

        # Insert combined time node at position n_combined
        _insert_time_node(obs_combined, int_combined, self.n_nodes, K)

        target_mask = np.zeros(K, dtype=np.float32)
        target_mask[sc["alarm_idx"]] = 1.0

        return {
            "obs": obs_combined,
            "int": int_combined,
            "multi_hot": multi_hot,
            "target_mask": target_mask,
            "n_nodes_with_time": self.n_nodes + 1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FullScaleEnvironment — 92-node grid mirroring full-mode evaluation
# ─────────────────────────────────────────────────────────────────────────────


class FullScaleEnvironment:
    """
    One sub-environment (probe/coolant/hydraulics) embedded at random positions
    within a 92-node grid.  All remaining positions are constant (zero), exactly
    mirroring the OPC-UA forward-filled nodes from uninvolved subsystems in
    full-mode evaluation.

    Scale and magnitude are identical to the sub environment — no rescaling —
    so the model does not need to learn a new magnitude regime.
    Node positions are randomly shuffled so the model cannot memorise order.
    """

    FULL_N = 92
    mode = "full_scale"

    def __init__(self,
                 max_nodes: int,
                 n_settings: int = 8,
                 n_normal_pool: int = 600):
        assert self.FULL_N + 1 <= max_nodes, (
            f"FULL_N+1={self.FULL_N+1} > max_nodes={max_nodes}")
        self.max_nodes = max_nodes

        sub_mode = random.choice(["probe", "coolant", "hydraulics"])
        self.sub_env = CausRCAStyleEnvironment(sub_mode, max_nodes, n_settings,
                                               n_normal_pool)
        self.n_sub = self.sub_env.n_nodes
        self.n_nodes = self.FULL_N  # prediction range for train_step

        # Random permutation: first n_sub slots are the active nodes
        perm = np.random.permutation(self.FULL_N)
        self._sub_to_full = perm[:self.n_sub]  # shape (n_sub,)

        # Rebuild scenarios with permuted global indices
        self.scenarios = []
        for sc in self.sub_env.scenarios:
            self.scenarios.append({
                "root_cause_idxs":
                [int(self._sub_to_full[r]) for r in sc["root_cause_idxs"]],
                "alarm_idx":
                int(self._sub_to_full[sc["alarm_idx"]]),
            })

    def make_sample(self,
                    scenario_idx: int,
                    n_obs: int,
                    n_int: int,
                    use_null: bool = False) -> dict:
        K = self.max_nodes
        sc = self.scenarios[scenario_idx]
        sub = self.sub_env.make_sample(scenario_idx, n_obs, n_int, use_null)

        obs_full = np.zeros((n_obs, K), dtype=np.float32)
        int_full = np.zeros((n_int, K), dtype=np.float32)
        for sub_idx, full_idx in enumerate(self._sub_to_full):
            obs_full[:, full_idx] = sub["obs"][:, sub_idx]
            int_full[:, full_idx] = sub["int"][:, sub_idx]

        multi_hot = np.zeros(K, dtype=np.float32)
        if not use_null:
            for idx in sc["root_cause_idxs"]:
                multi_hot[idx] = 1.0

        target_mask = np.zeros(K, dtype=np.float32)
        target_mask[sc["alarm_idx"]] = 1.0

        _insert_time_node(obs_full, int_full, self.FULL_N, K)

        return {
            "obs": obs_full,
            "int": int_full,
            "multi_hot": multi_hot,
            "target_mask": target_mask,
            "n_nodes_with_time": self.FULL_N + 1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_environments(n_graphs: int = 200,
                          n_settings: int = 8,
                          max_nodes: int = 100,
                          seed: int = 42) -> list:
    """
    35% probe / 20% full_scale / 15% coolant / 10% hydraulics / 20% combined.

    full_scale (n=80-95) is critical: covers the K/n≈1× scale seen in full-mode
    evaluation (n≈92 nodes). Without it the model collapses to zero on full-mode.
    Probe is over-represented because it's the hardest sub-mode case.
    """
    np.random.seed(seed)
    random.seed(seed)

    n_combined = n_graphs * 20 // 100
    n_probe = n_graphs * 35 // 100
    n_full_scale = n_graphs * 20 // 100
    n_cool = n_graphs * 15 // 100
    n_hyd = n_graphs - n_probe - n_full_scale - n_cool - n_combined

    envs = []
    for mode, count in [("probe", n_probe), ("coolant", n_cool),
                        ("hydraulics", n_hyd)]:
        for _ in range(count):
            try:
                envs.append(
                    CausRCAStyleEnvironment(mode, max_nodes, n_settings))
            except Exception as e:
                print(f"  Warning: {mode} env failed: {e}")

    for _ in range(n_full_scale):
        try:
            envs.append(FullScaleEnvironment(max_nodes, n_settings))
        except Exception as e:
            print(f"  Warning: full_scale env failed: {e}")

    for _ in range(n_combined):
        try:
            envs.append(MultiSubsystemEnvironment(max_nodes, n_settings))
        except Exception as e:
            print(f"  Warning: combined env failed: {e}")

    random.shuffle(envs)

    counts = {
        m: sum(1 for e in envs if e.mode == m)
        for m in ["probe", "full_scale", "coolant", "hydraulics", "combined"]
    }
    multi = sum(1 for e in envs for sc in e.scenarios
                if len(sc["root_cause_idxs"]) > 1)
    print(f"Generated {len(envs)} environments  {counts}  multi-fault={multi}")
    return envs


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────


def train_step(model, optimizer, ctx_obs, ctx_int, ctx_tmasks, q_obs, q_int,
               q_tmasks, q_multi_hot, n_nodes, n_nodes_with_time, max_nodes):
    """
    One gradient step.

    n_nodes            – number of actual (physical) nodes; prediction targets.
    n_nodes_with_time  – n_nodes + 1; passed to model so it sees the time column.

    Loss:
      CE  over each true root-cause node  (ranking-aligned with MAP@k)
    + alarm penalty: push alarm-node logit toward −∞  (breaks alarm-shortcut)
    """
    model.train()
    optimizer.zero_grad()

    n_q = q_obs.shape[0]
    all_obs = torch.cat([ctx_obs, q_obs], dim=0)
    all_int = torch.cat([ctx_int, q_int], dim=0)
    all_tms = torch.cat([ctx_tmasks, q_tmasks], dim=0)

    # Forward: model sees n_nodes_with_time positions (includes time column)
    logits = model(all_obs,
                   all_int,
                   target_mask=all_tms,
                   n_nodes=n_nodes_with_time)
    query_logits = logits[-n_q:]

    # Prediction mask: only actual nodes (exclude time node + padding)
    pad_mask = torch.full((n_q, max_nodes),
                          float("-inf"),
                          device=query_logits.device)
    pad_mask[:, :n_nodes] = 0.0
    masked = query_logits + pad_mask

    loss_terms = []
    for qi in range(n_q):
        pos_mask = q_multi_hot[qi, :n_nodes] > 0
        pos_idxs = pos_mask.nonzero(as_tuple=False).squeeze(1)

        for pi in pos_idxs:
            loss_terms.append(
                F.cross_entropy(masked[qi:qi + 1], pi.unsqueeze(0)))

        # Alarm penalty: push alarm positions' logits down
        for ai in (q_tmasks[qi, :n_nodes]
                   > 0).nonzero(as_tuple=False).squeeze(1):
            if not pos_mask[ai]:
                loss_terms.append(
                    F.softplus(masked[qi, ai]) * ALARM_PENALTY_WEIGHT)

    if loss_terms:
        loss = torch.stack(loss_terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_val = loss.item()
    else:
        loss_val = 0.0

    with torch.no_grad():
        pred = masked.argmax(dim=-1)
        gt = q_multi_hot[:, :n_nodes].argmax(dim=-1)
        has_label = q_multi_hot[:, :n_nodes].sum(dim=-1) > 0
        acc = (pred[has_label] == gt[has_label]).float().mean().item() \
              if has_label.any() else float("nan")

    return loss_val, acc


# ─────────────────────────────────────────────────────────────────────────────
# Episode construction
# ─────────────────────────────────────────────────────────────────────────────


def build_episode(env, scenario_idx: int, all_envs: list, n_obs: int,
                  n_int: int, k_context_range: tuple, n_queries: int,
                  device: torch.device) -> tuple:
    K = env.max_nodes

    k_max = min(k_context_range[1], len(env.scenarios) - 1)
    k_context = np.random.randint(k_context_range[0], max(k_max + 1, 1))

    other_sc = [i for i in range(len(env.scenarios)) if i != scenario_idx]
    ctx_indices = random.sample(other_sc, min(k_context, len(other_sc)))

    extra = k_context - len(ctx_indices)
    borrowed = []
    if extra > 0 and len(all_envs) > 1:
        others = [e for e in all_envs if e is not env]
        borrowed = random.sample(others, min(extra, len(others)))

    def _t(x):
        return torch.from_numpy(np.array(x, dtype=np.float32)).to(device)

    ctx_o, ctx_i, ctx_m = [], [], []
    for ci in ctx_indices:
        s = env.make_sample(ci, n_obs, n_int)
        ctx_o.append(s["obs"])
        ctx_i.append(s["int"])
        ctx_m.append(s["target_mask"])

    for be in borrowed:
        s = be.make_sample(np.random.randint(len(be.scenarios)), n_obs, n_int)
        ctx_o.append(s["obs"])
        ctx_i.append(s["int"])
        ctx_m.append(s["target_mask"])

    q_o, q_i, q_mhot, q_tm = [], [], [], []
    for _ in range(n_queries):
        s = env.make_sample(scenario_idx, n_obs, n_int, use_null=False)
        q_o.append(s["obs"])
        q_i.append(s["int"])
        q_mhot.append(s["multi_hot"])
        q_tm.append(s["target_mask"])

    # n_nodes_with_time from the last query sample
    n_nwt = s["n_nodes_with_time"]

    q_obs = _t(np.stack(q_o))
    q_int = _t(np.stack(q_i))
    q_mhot_t = _t(np.stack(q_mhot))
    q_tmasks = _t(np.stack(q_tm))

    if ctx_o:
        ctx_obs = _t(np.stack(ctx_o))
        ctx_int = _t(np.stack(ctx_i))
        ctx_tmasks = _t(np.stack(ctx_m))
    else:
        ctx_obs = torch.empty(0, n_obs, K, dtype=torch.float32, device=device)
        ctx_int = torch.empty(0, n_int, K, dtype=torch.float32, device=device)
        ctx_tmasks = torch.empty(0, K, dtype=torch.float32, device=device)

    return (ctx_obs, ctx_int, ctx_tmasks, q_obs, q_int, q_tmasks, q_mhot_t,
            n_nwt)


# ─────────────────────────────────────────────────────────────────────────────
# Main fine-tuning function
# ─────────────────────────────────────────────────────────────────────────────


def finetune_causrca(model,
                     device,
                     environments: list,
                     lr=3e-5,
                     max_nodes=100,
                     k_context_range=(0, 3),
                     n_queries=4,
                     n_epochs=5,
                     output_path="finetune_causrca_final.pt",
                     warmup_fraction=0.05):
    # Freeze backbone to prevent catastrophic forgetting.
    # Only the decoder MLP is updated — the transformer and encoders keep
    # their pre-trained weights.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("decoder.")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters()
                   if not p.requires_grad)
    print(
        f"Frozen backbone: {n_frozen:,} params  |  Trainable decoder: {n_trainable:,} params"
    )

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=lr,
                            weight_decay=0.01)
    all_pairs = [(env, si) for env in environments
                 for si in range(len(env.scenarios))]
    total_steps = len(all_pairs) * n_epochs
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    counts = {
        m: sum(1 for e in environments if e.mode == m)
        for m in ["probe", "full_scale", "coolant", "hydraulics", "combined"]
    }
    multi = sum(1 for e in environments for sc in e.scenarios
                if len(sc["root_cause_idxs"]) > 1)

    print("=" * 70)
    print("FINETUNING ON SYNTHETIC CausRCA-LIKE DATA  (time-node enabled)")
    print("=" * 70)
    print(f"Envs    : {len(environments)}  {counts}  multi-fault={multi}")
    print(
        f"Pairs   : {len(all_pairs)}   Loss: CE + alarm-penalty (λ={ALARM_PENALTY_WEIGHT})"
    )
    print(
        f"n_int   : probe={N_INT_RANGE['probe']}  "
        f"coolant={N_INT_RANGE['coolant']}  hydraulics={N_INT_RANGE['hydraulics']}"
    )
    print(f"LR      : {lr}  warmup={warmup_steps}  cosine  epochs={n_epochs}")
    print("=" * 70)

    model.train()
    for epoch in range(n_epochs):
        print(f"\n── Epoch {epoch + 1}/{n_epochs} ──")
        episodes = list(all_pairs)
        random.shuffle(episodes)

        tot_loss = tot_acc = n_steps = n_acc = 0

        for env, si in tqdm(episodes, desc=f"Epoch {epoch+1}"):
            mode = (env.sub[env.scenarios[si]["faulty"]].mode if hasattr(
                env, "sub") else env.mode)
            n_int_lo, n_int_hi = N_INT_RANGE.get(mode, N_INT_RANGE["combined"])
            n_obs = np.random.randint(N_OBS_RANGE[0], N_OBS_RANGE[1] + 1)
            n_int = np.random.randint(n_int_lo, n_int_hi + 1)

            (ctx_obs, ctx_int, ctx_tmasks, q_obs, q_int, q_tmasks, q_mhot,
             n_nwt) = build_episode(env, si, environments, n_obs, n_int,
                                    k_context_range, n_queries, device)

            loss, acc = train_step(
                model,
                optimizer,
                ctx_obs,
                ctx_int,
                ctx_tmasks,
                q_obs,
                q_int,
                q_tmasks,
                q_mhot,
                n_nodes=env.n_nodes,
                n_nodes_with_time=n_nwt,
                max_nodes=max_nodes,
            )

            scheduler.step()
            tot_loss += loss
            n_steps += 1
            if not np.isnan(acc):
                tot_acc += acc
                n_acc += 1

        cur_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}  avg_loss={tot_loss/max(n_steps,1):.4f}  "
              f"avg_acc={tot_acc/max(n_acc,1):.4f}  lr={cur_lr:.2e}")

    torch.save(model.state_dict(), output_path)
    print(f"\nSaved: {output_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CausRCA benchmark (evaluation on real digital-twin data)
# ─────────────────────────────────────────────────────────────────────────────


def _run_causrca_benchmark(
    model, device, max_nodes: int, modes: list[str] = ("sub", "full")) -> None:
    """
    Evaluate model on CausRCA digital-twin dataset.

    Mirrors test_causrca.py logic exactly, with one addition: a time node is
    inserted at position n_nodes so the model receives the same temporal
    encoding as during finetuning.

    Args:
        model:      loaded InterventionZeroShotModel (eval mode)
        device:     torch device
        max_nodes:  K (must match model's max_nodes)
        modes:      list containing "sub", "full", or both
    """
    # --- lazy imports from test_causrca.py and causrca library ---
    try:
        project_root = Path(__file__).parent.parent.parent
        causrca_root = project_root / "causRCA-main"
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(causrca_root / "src"))

        import pandas as pd
        from causrca.utils.discretize import __discretize_dataset_helper
        from causrca.utils.utils import get_encoding_dict

        from prim.evaluation.test_causrca import (
            DATASETS,
            TIME_STEP_MS,
            MIN_NORMAL_STEPS,
            load_relevant_nodes,
            load_full_nodes,
            collect_all_runs,
            calculate_apk,
        )
    except ImportError as e:
        print(f"  [benchmark] Cannot import causRCA helpers: {e}")
        print("  [benchmark] Skipping evaluation.")
        return

    CATEGORICAL_ENCODING_DICT = get_encoding_dict()

    def _discretize(df, max_time_s):
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

    def _run_to_model_input(run: dict, node_names: list):
        """
        Convert a CausRCA run to (obs_padded, int_padded) including time node.
        Returns None if insufficient data.
        """
        df = run["csv_df"]
        t_fault = run["cause_start_at"]
        t_diag = run["diagnosis_at"]

        df_normal = df[df["time_s"] <= t_fault].copy()
        df_abnormal = df[(df["time_s"] > t_fault)
                         & (df["time_s"] <= t_diag)].copy()
        if df_normal.empty or df_abnormal.empty:
            return None

        t_off = df_abnormal["time_s"].min()
        df_abnormal = df_abnormal.copy()
        df_abnormal["time_s"] -= t_off

        dur_n = df_normal["time_s"].max()
        dur_a = df_abnormal["time_s"].max()
        if dur_n < (TIME_STEP_MS / 1000.0) * MIN_NORMAL_STEPS:
            return None

        wide_n = _discretize(df_normal, dur_n).reindex(columns=node_names,
                                                       fill_value=np.nan)
        wide_a = _discretize(df_abnormal, dur_a).reindex(columns=node_names,
                                                         fill_value=np.nan)

        arr_n = wide_n.values.astype(np.float64)
        arr_a = wide_a.values.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            col_means = np.nanmean(arr_n, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for c in range(arr_n.shape[1]):
            arr_n[np.isnan(arr_n[:, c]), c] = col_means[c]
            arr_a[np.isnan(arr_a[:, c]), c] = col_means[c]

        n_nodes = len(node_names)
        K = max_nodes
        _CLIP = 10.0

        obs_mean = arr_n.mean(axis=0, keepdims=True)
        obs_std = arr_n.std(axis=0, keepdims=True) + 1e-8
        obs_norm = np.clip((arr_n - obs_mean) / obs_std, -_CLIP, _CLIP)
        int_norm = np.clip((arr_a - obs_mean) / obs_std, -_CLIP, _CLIP)

        scale = K / n_nodes
        n_obs, n_int = obs_norm.shape[0], int_norm.shape[0]

        obs_padded = np.zeros((n_obs, K), dtype=np.float32)
        int_padded = np.zeros((n_int, K), dtype=np.float32)
        obs_padded[:, :n_nodes] = (obs_norm * scale).astype(np.float32)
        int_padded[:, :n_nodes] = (int_norm * scale).astype(np.float32)

        # Insert time node at column n_nodes (same as finetuning)
        _insert_time_node(obs_padded, int_padded, n_nodes, K)

        return obs_padded, int_padded, n_nodes  # return n_nodes (actual), not +1

    model.eval()
    all_dfs: dict[str, list] = {m: [] for m in modes}

    for eval_mode in modes:
        full_node_names = load_full_nodes() if eval_mode == "full" else None
        rows = []

        for ds_name, ds_path in DATASETS.items():
            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name}  [mode={eval_mode}]")
            node_names = full_node_names if eval_mode == "full" \
                         else load_relevant_nodes(ds_path)
            if not node_names:
                print("  WARNING: no nodes found, skipping.")
                continue

            n_nodes = len(node_names)
            print(
                f"  Nodes ({n_nodes}): {node_names[:4]}{'...' if n_nodes > 4 else ''}"
            )

            for run in collect_all_runs(ds_path):
                result = _run_to_model_input(run, node_names)
                if result is None:
                    print(f"  [{run['run_name']}] SKIP: insufficient data")
                    continue
                obs_padded, int_padded, n_act = result

                target_mask = np.zeros(max_nodes, dtype=np.float32)
                for alarm in run["alarm_nodes"]:
                    if alarm in node_names:
                        target_mask[node_names.index(alarm)] = 1.0

                q_obs_t = torch.from_numpy(obs_padded).unsqueeze(0).to(device)
                q_int_t = torch.from_numpy(int_padded).unsqueeze(0).to(device)
                q_mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(
                    device)

                with torch.no_grad():
                    # n_nodes_with_time = n_act + 1
                    logits = model(q_obs_t,
                                   q_int_t,
                                   target_mask=q_mask_t,
                                   n_nodes=n_act + 1)
                    # Mask: keep only actual nodes; exclude time node, padding, and
                    # known alarm nodes (problem is "given alarm, find RC" — alarm is
                    # never the correct answer and always has a large z-score diff).
                    pred_mask = torch.full_like(logits, float("-inf"))
                    pred_mask[:, :n_act] = 0.0
                    pred_mask[:, :
                              n_act] -= q_mask_t[:, :
                                                 n_act] * 1e9  # zero out alarms
                    probs = torch.softmax(logits + pred_mask,
                                          dim=-1).cpu().numpy()[0]

                ranked_order = np.argsort(-probs[:n_act])
                ranked = [node_names[i] for i in ranked_order]
                gt = run["manipulated_vars"]

                def in_topk(r, g, k):
                    return int(any(x in r[:k] for x in g))

                rows.append({
                    "dataset": ds_name,
                    "exp": run["exp_name"],
                    "run": run["run_name"],
                    "manipulated": "|".join(gt),
                    "top1_pred": ranked[0] if ranked else "",
                    "rec@1": in_topk(ranked, gt, 1),
                    "rec@3": in_topk(ranked, gt, 3),
                    "rec@5": in_topk(ranked, gt, 5),
                    "_ranked": ranked,
                    "_gt": gt,
                    "n_nodes": n_act,
                })
                print(f"  {run['exp_name']}/{run['run_name']:6s}  "
                      f"gt={gt}  top1={ranked[0] if ranked else '?'}  "
                      f"rec@3={'Y' if rows[-1]['rec@3'] else 'N'}")

        all_dfs[eval_mode] = pd.DataFrame(rows)

    # Print results
    for eval_mode in modes:
        df = all_dfs[eval_mode]
        if df.empty:
            continue
        print(f"\n{'='*72}")
        print(f"CausRCA Benchmark  [{eval_mode}]  MAP@3 and Recall")
        print(f"{'='*72}")
        print(
            f"{'dataset':<14} {'MAP@3':>8}  {'ci_lo':>8}  {'ci_hi':>8}  {'R@1':>6}  {'ci_lo':>8}  {'ci_hi':>8}  {'R@3':>6}  {'ci_lo':>8}  {'ci_hi':>8}  {'n':>5}"
        )
        print("-" * 100)
        for ds in list(DATASETS.keys()) + ["OVERALL"]:
            sub = df if ds == "OVERALL" else df[df["dataset"] == ds]
            if len(sub) == 0:
                continue
            ap_vals = [
                calculate_apk(gt, pred, k=3) for gt, pred in zip(
                    sub["_gt"].tolist(), sub["_ranked"].tolist())
            ]
            map3 = float(np.mean(ap_vals))
            map3_lo, map3_hi = _bootstrap_ci(ap_vals)
            r1 = sub["rec@1"].mean()
            r1_lo, r1_hi = _bootstrap_ci(sub["rec@1"].values)
            r3 = sub["rec@3"].mean()
            r3_lo, r3_hi = _bootstrap_ci(sub["rec@3"].values)
            print(
                f"{ds:<14} {map3:>8.3f}  {map3_lo:>8.3f}  {map3_hi:>8.3f}  {r1:>6.3f}  {r1_lo:>8.3f}  {r1_hi:>8.3f}  {r3:>6.3f}  {r3_lo:>8.3f}  {r3_hi:>8.3f}  {len(sub):>5}"
            )

    if len(modes) == 2 and all(not all_dfs[m].empty for m in modes):
        df_s, df_f = all_dfs["sub"], all_dfs["full"]
        print(f"\n{'='*75}")
        print("Sub vs Full Mode  –  MAP@3")
        print(f"{'='*75}")
        print(
            f"{'dataset':<14} {'sub MAP@3':>10} {'ci_lo':>8} {'ci_hi':>8} {'full MAP@3':>11} {'ci_lo':>8} {'ci_hi':>8} {'delta':>7}  {'n':>5}"
        )
        print("-" * 82)
        for ds in list(DATASETS.keys()) + ["OVERALL"]:
            s = df_s if ds == "OVERALL" else df_s[df_s["dataset"] == ds]
            f = df_f if ds == "OVERALL" else df_f[df_f["dataset"] == ds]
            if len(s) == 0 and len(f) == 0:
                continue

            def _map_ci(sub):
                if len(sub) == 0:
                    return float("nan"), float("nan"), float("nan")
                ap_vals = [
                    calculate_apk(gt, pred, k=3) for gt, pred in zip(
                        sub["_gt"].tolist(), sub["_ranked"].tolist())
                ]
                lo, hi = _bootstrap_ci(ap_vals)
                return float(np.mean(ap_vals)), lo, hi

            ms, ss_lo, ss_hi = _map_ci(s)
            mf, sf_lo, sf_hi = _map_ci(f)
            dl = mf - ms if not (np.isnan(ms)
                                 or np.isnan(mf)) else float("nan")
            sign = "+" if dl > 0 else ""
            print(
                f"{ds:<14} {ms:>10.3f} {ss_lo:>8.3f} {ss_hi:>8.3f} {mf:>11.3f} {sf_lo:>8.3f} {sf_hi:>8.3f} {sign}{dl:>6.3f}  {max(len(s),len(f)):>5}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",
                        type=str,
                        required=True,
                        help="Pretrained or finetuned checkpoint (.pt)")
    parser.add_argument("--output",
                        type=str,
                        default="finetune_causrca_final.pt")
    parser.add_argument("--n_graphs", type=int, default=200)
    parser.add_argument("--n_settings", type=int, default=8)
    parser.add_argument("--max_nodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_queries", type=int, default=4)
    parser.add_argument("--k_context_min", type=int, default=0)
    parser.add_argument("--k_context_max", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval",
                        action="store_true",
                        help="Run CausRCA benchmark after finetuning")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip finetuning; run benchmark only (uses --resume checkpoint)")
    parser.add_argument("--eval_mode",
                        type=str,
                        nargs="+",
                        choices=["sub", "full"],
                        default=["sub", "full"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from prim.model.prim_neural_process import PRIMNeuralProcess as PRIM

    model = PRIM(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=args.max_nodes,
        dropout=0.0,
    ).to(device)

    state = torch.load(args.resume, map_location=device)
    model.load_state_dict(state)
    print(
        f"Loaded: {args.resume}  ({sum(p.numel() for p in model.parameters()):,} params)"
    )

    if not args.eval_only:
        print("\nGenerating synthetic CausRCA-like environments ...")
        envs = generate_environments(
            n_graphs=args.n_graphs,
            n_settings=args.n_settings,
            max_nodes=args.max_nodes,
            seed=args.seed,
        )
        if not envs:
            raise RuntimeError("No environments generated.")

        finetune_causrca(
            model,
            device,
            envs,
            lr=args.lr,
            max_nodes=args.max_nodes,
            k_context_range=(args.k_context_min, args.k_context_max),
            n_queries=args.n_queries,
            n_epochs=args.n_epochs,
            output_path=args.output,
        )

    if args.eval or args.eval_only:
        print("\nRunning CausRCA digital-twin benchmark ...")
        _run_causrca_benchmark(model,
                               device,
                               max_nodes=args.max_nodes,
                               modes=args.eval_mode)


if __name__ == "__main__":
    main()
