"""
TabICL-style structural-causal-model prior, adapted to the Root-Cause-Analysis
(RCA) meta-learning setting.

--------------------------------------------------------------------------------
What is taken from TabICL  (Qu et al., 2026, "TabICL",
                            https://github.com/soda-inria/tabicl/tree/main/src/tabicl/prior)
--------------------------------------------------------------------------------
* A per-node structural equation drawn from a MIXTURE of function classes:
    - ``mlp``  : a small randomly-initialised MLP with diverse activations,
                 log-uniform init-std, per-layer Gaussian noise injection and
                 block-wise weight dropout (sparse mechanisms).
    - ``tree`` : a decision-tree / random-forest / extra-trees / gradient-boosting
                 regressor fitted to *random* targets -- yields piecewise-constant,
                 threshold-style mechanisms that look like real tabular / monitoring
                 signals (TabICL's ``TreeSCM``).
    - ``linear`` / ``gp`` : kept from the existing prior for extra diversity.
  Mixture weights default to TabICL's ``mix_probs`` = (mlp 0.6, tree 0.25, ...).
* Root ("cause") variables sampled from a mixture of base distributions
  (normal / uniform / laplace / student-t / multi-modal / mixed) -- TabICL's
  ``XSampler``.
* Feature-wise post-processing that makes columns look like real tabular data:
    - non-linear WARPING  (rank->gaussian, Kumaraswamy CDF, signed-log, power,
      exp, affine-abs)  fitted on the OBSERVATIONAL window only;
    - DISCRETISATION of a random subset of columns into ordinal / categorical
      codes (TabICL's ``_num2cat``: per-column prob ``cat_prob`` = 0.2, number
      of categories ~ clamp(Gamma(1, 10), 2, max_categories), rank-based bin
      edges taken from the observational window).
  => the prior now emits DISCRETE columns as well as continuous ones, and the
     model sees a ``discrete_mask`` telling it which is which.

--------------------------------------------------------------------------------
RCA-specific adaptations (kept from the existing prior)
--------------------------------------------------------------------------------
* Every episode produces an OBSERVATIONAL window and an INTERVENTIONAL window
  from the *same* SCM, plus a one-hot ``intervention_label`` (the root cause)
  and a one-hot ``target_mask`` (the observed symptom = the intervention node or
  one of its descendants).
* Interventions are equation-agnostic and propagate through the graph:
    - ``weight`` : y' = mu_obs + (+/-) factor * (y - mu_obs)     (soft, may flip sign)
    - ``shift``  : y' = y + delta * sigma_obs                    (soft mean shift)
    - ``do``     : y' = mu_obs +/- m * sigma_obs                 (hard "stuck sensor")
    - ``noise``  : inflate the driving noise std of the node
* Optional AR(1) temporal autocorrelation of the driving noise (slow time-series).
* Optional near-constant "availability" mode (high baseline, tiny noise) so a
  fault saturates the +/-10 z-score clip -- matches causRCA / PetShop availability
  metrics.
* PFN-style encoding: zero-pad to ``max_nodes`` and scale by ``max_nodes / k``.

--------------------------------------------------------------------------------
!!!  NORMALISATION INVARIANT  (this is the bug the RCA prior must not have)  !!!
--------------------------------------------------------------------------------
The *whole* RCA signal is the distribution shift between the observational and
interventional windows.  If you standardise the interventional window with its
OWN mean/std, every mean shift is subtracted away and the root cause becomes
invisible.

Therefore ``normalize_observational_only`` computes ``mean`` / ``std`` (and warp
params, and discretisation bin edges) from the OBSERVATIONAL window ONLY and
applies exactly those to BOTH windows.  This matches every real-data evaluation
path in this repo:

    prim/data/generate_dataset.py            (obs_mean/obs_std from obs_data)
    prim/training/train_meta_rca.py          (_generate_sample_from_scm)
    prim/evaluation/test_petshop.py          (normal_arr stats)
    prim/evaluation/test_causrca.py          (normal_arr stats)
    prim/evaluation/test_multi_rca.py        (obs / pool stats)
    prim/evaluation/test_threenode_synthetic.py(to_model_input: obs stats)
    prim/evaluation/test_prior_eval.py       (_normalize_pad: obs stats)
    prim/evaluation/ablation_finetuning*.py  (_normalize_pad: obs stats)

Keep it that way.  ``normalize_observational_only`` is the single choke point.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from prim.data.dag_generator_simplified import (
    NOISE_TYPES,
    GPEquation,
    _sample_noise,
    generate_dag,
)

try:  # tree-based structural equations are optional
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.tree import DecisionTreeRegressor

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

CLIP = 10.0  # z-score clip, matches every eval path in the repo

# =============================================================================
# Config
# =============================================================================


@dataclass
class TabICLPriorConfig:
    """Sampling ranges for one episode.  Call :meth:`sample` to draw concrete HPs."""

    # ---- graph -------------------------------------------------------------
    n_nodes_range: tuple = (10, 20)
    max_nodes: int = 50
    graph_families: tuple = ("erdos-renyi", "barabasi-albert", "bipartite")
    expected_degree_range: tuple = (1.5, 3.0)

    # ---- alarm / aggregator nodes + sub-vs-full-system fault confinement --
    # Topology stays the plain er/ba/bipartite families above -- only the
    # ROLES assigned on top of them change:
    #   * a few high-in-degree sinks are "aggregators" (SLIs) where an alarm
    #     fires -- the observed symptom is drawn from one of these more often
    #     than a random leaf;
    #   * the fault is confined to a local (undirected) neighbourhood around a
    #     random anchor node.  On a SMALL graph the neighbourhood is close to
    #     the whole graph (sub-system-like); on a LARGE graph it is a small
    #     fraction (full-system-like) and the rest of the graph is active but
    #     UNSHIFTED -- the same size-driven distinction as causRCA sub vs full
    #     mode, without hard-coding its topology.
    aggregator_frac_range: tuple = (0.03, 0.12)
    aggregator_symptom_prob: float = 0.6
    confine_fault_prob: float = 0.7
    confine_radius_range: tuple = (1, 3)
    rc_from_source_prob: float = 0.5

    # ---- sparse embedding (full-scale causRCA) ----------------------
    # A small dense SCM embedded at random positions in a k-node grid whose
    # other columns are silent constants (full-mode causRCA structure).
    # DISABLED by default: it regressed PetShop and did not help full-mode
    # causRCA.  Code kept for experiments -- set >0 to re-enable.
    sparse_embed_prob: float = 0.0
    sparse_active_range: tuple = (8, 35)

    # ---- windows ---------------------------------------------------------
    n_obs_range: tuple = (100, 400)
    n_int_range: tuple = (50, 150)

    # ---- structural-equation mixture (TabICL mix_probs, extended) --------
    #   mlp / tree / linear / gp
    # Tree-SCM makes jagged piecewise-constant mechanisms; causRCA's digital
    # twin has smoother dynamics, so keep the tree share modest and lean on
    # mlp / linear / gp (the old prior used linear / nn / constant_baseline / gp).
    equation_mix: tuple = (0.50, 0.13, 0.22, 0.15)

    # ---- MLP-SCM (TabICL _mlp_scm) --------------------------------------
    mlp_num_layers_range: tuple = (2, 5)
    mlp_hidden_dim_range: tuple = (8, 64)  # log-uniform
    mlp_init_std_range: tuple = (0.05, 3.0)  # log-uniform
    mlp_noise_std_range: tuple = (1e-3, 0.3)  # log-uniform
    mlp_block_dropout_prob: float = 0.3
    mlp_sparsity_range: tuple = (0.1, 0.55)

    # ---- Tree-SCM (TabICL _tree_scm) -----------------------------------
    tree_kinds: tuple = ("dt", "rf", "et", "gb")
    tree_depth_lambda: float = 0.5  # depth = 2 + Exp(1/lambda)
    tree_n_estimators_lambda: float = 0.5
    tree_depth_cap: int = 8
    tree_n_estimators_cap: int = 24
    tree_noise_std_range: tuple = (1e-3, 0.25)

    # ---- root ("cause") distribution (TabICL XSampler) ----------------
    root_kinds: tuple = (
        "normal",
        "uniform",
        "laplace",
        "student_t",
        "multimodal",
        "mixed",
    )

    # ---- noise -------------------------------------------------------
    # causRCA monitoring data is low-noise and strongly auto-correlated
    # (slow time series).  Matches generate_dataset.py's causRCA-tuned run:
    # noise_scale_range=(0.1,0.2), temporal_autocorr_range=(0.8,0.99).
    noise_scale_range: tuple = (0.05, 0.5)  # log-uniform
    temporal_autocorr_prob: float = 0.6
    temporal_autocorr_range: tuple = (0.8, 0.98)

    # ---- availability / constant-baseline mode -----------------------
    # The old prior's `constant_baseline` equation type + linear high-baseline
    # mode (dag_generator_simplified.py): near-constant ~100 metrics with tiny
    # noise, so any fault saturates the +/-10 z-score clip -- the availability /
    # success-rate signature in both causRCA and PetShop.
    availability_prob: float = 0.35
    availability_baseline_range: tuple = (80.0, 120.0)
    availability_noise_range: tuple = (1e-4, 5e-3)
    # fault on an availability node = drop by this fraction of the baseline
    # (cf. apply_shift_intervention high_baseline branch: -baseline*U(0.03,0.15))
    availability_fault_frac_range: tuple = (0.03, 0.20)

    # ---- feature realism post-processing (TabICL _reg2cls / _num2cat) --
    warp_prob: float = 0.6  # per-column prob of a non-linear warp
    cat_prob: float = 0.2  # per-column prob of discretisation
    max_categories: int = 12
    warp_kinds: tuple = (
        "rank_gauss",
        "kumaraswamy",
        "signed_log",
        "power",
        "exp",
        "affine_abs",
    )

    # ---- intervention mixture --------------------------------------
    #   weight / shift / do (stuck) / noise-inflation / cascade
    #
    # Magnitudes are the MMD-notebook-calibrated values (prior_mmd_calibration
    # notebook): weight is the weak/ambiguous signal -> needs factor ~5 to be
    # reliably detectable and never hits reject>=0.95, so high flip prob;
    # shift / do reach reject>=0.95 at sigma ~1.
    #
    # But two things stay AWAY from the notebook's aggressive settings:
    #   * `do` (stuck sensor, variance -> 0) is kept at ~old-prior levels (0.08).
    #     At 0.18-0.20 the model learned "low-variance node == root cause",
    #     catastrophic on full-mode causRCA (dozens of silent-subsystem nodes
    #     are constant).
    #   * `cascade` (non-propagating post-pass overlay) and sparse embedding are
    #     DISABLED -- both regressed PetShop and did not help causRCA.  Code
    #     kept; raise the 5th mixture entry / sparse_embed_prob to experiment.
    #
    # weight / shift / do PROPAGATE through the SCM equations (applied inside the
    # topological sampling loop) -- that is what makes them look like real faults.
    #   weight / shift / do / noise / cascade
    intervention_mix: tuple = (0.62, 0.25, 0.08, 0.05, 0.0)
    # Number of simultaneous, causally-independent root causes per episode:
    # P(1 RC), P(2 RCs), P(3 RCs).  causRCA is a multi-root-cause benchmark;
    # a model trained only on single-RC episodes learns to suppress every node
    # outside the one anomalous cone, so at eval it ranks a genuine 2nd/3rd root
    # cause low (hurts recall@3 / MAP@3).  Keep it majority-single so PetShop
    # (single-RC) is unaffected -- and with the multi-label loss a 1-RC episode
    # is identical to plain cross-entropy.
    n_faults_probs: tuple = (0.7, 0.2, 0.1)
    weight_change_range: tuple = (5.0, 9.0)
    weight_flip_prob: float = 0.8
    shift_sigma_range: tuple = (1.0, 4.0)
    stuck_sigma_range: tuple = (1.0, 4.0)
    noise_inflate_range: tuple = (3.0, 6.0)

    # ---- cascade (disabled by default; see note above) --------------
    cascade_sigma_range: tuple = (1.5, 6.0)
    cascade_delay_per_hop_range: tuple = (2, 8)
    cascade_attenuation: float = 0.45
    cascade_branch_prob: float = 0.8
    cascade_n_int_range: tuple = (8, 45)

    def sample(self, rng: np.random.Generator) -> dict:
        """Draw one concrete hyper-parameter dict for a single episode."""

        def logu(lo, hi):
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

        eq_type = rng.choice(["mlp", "tree", "linear", "gp"],
                             p=_norm(self.equation_mix))
        if eq_type == "tree" and not _HAS_SKLEARN:
            eq_type = "mlp"

        availability = rng.random() < self.availability_prob
        noise_scale = (rng.uniform(*self.availability_noise_range)
                       if availability else logu(*self.noise_scale_range))

        return dict(
            equation_type=eq_type,
            graph_family=str(rng.choice(list(self.graph_families))),
            expected_degree=float(rng.uniform(*self.expected_degree_range)),
            aggregator_frac=float(rng.uniform(*self.aggregator_frac_range)),
            aggregator_symptom_prob=float(self.aggregator_symptom_prob),
            confine_fault_prob=float(self.confine_fault_prob),
            confine_radius=int(
                rng.integers(self.confine_radius_range[0],
                             self.confine_radius_range[1] + 1)),
            rc_from_source_prob=float(self.rc_from_source_prob),
            # mlp
            mlp_num_layers=int(
                rng.integers(self.mlp_num_layers_range[0],
                             self.mlp_num_layers_range[1] + 1)),
            mlp_hidden_dim=int(round(logu(*self.mlp_hidden_dim_range))),
            mlp_init_std=logu(*self.mlp_init_std_range),
            mlp_noise_std=logu(*self.mlp_noise_std_range),
            mlp_block_dropout=bool(rng.random() < self.mlp_block_dropout_prob),
            mlp_sparsity=float(rng.uniform(*self.mlp_sparsity_range)),
            mlp_activation=str(rng.choice(_ACTIVATION_NAMES)),
            # tree
            tree_kind=str(rng.choice(list(self.tree_kinds))),
            tree_depth=int(
                min(
                    self.tree_depth_cap,
                    2 + int(rng.exponential(1.0 / self.tree_depth_lambda)),
                )),
            tree_n_estimators=int(
                min(
                    self.tree_n_estimators_cap,
                    2 +
                    int(rng.exponential(1.0 / self.tree_n_estimators_lambda)),
                )),
            tree_noise_std=logu(*self.tree_noise_std_range),
            # roots / noise
            root_kind=str(rng.choice(list(self.root_kinds))),
            noise_type=str(rng.choice(NOISE_TYPES)),
            noise_scale=noise_scale,
            temporal_autocorr=(float(rng.uniform(
                *self.temporal_autocorr_range)) if rng.random()
                               < self.temporal_autocorr_prob else 0.0),
            # availability
            availability=availability,
            availability_baseline=float(
                rng.uniform(*self.availability_baseline_range)),
            availability_fault_frac=float(
                rng.uniform(*self.availability_fault_frac_range)),
            # feature realism
            warp_prob=self.warp_prob,
            cat_prob=self.cat_prob,
            max_categories=self.max_categories,
            # intervention
            intervention_kind=str(
                rng.choice(
                    ["weight", "shift", "do", "noise", "cascade"],
                    p=_norm(self.intervention_mix),
                )),
            weight_change=float(rng.uniform(*self.weight_change_range)),
            weight_flip=bool(rng.random() < self.weight_flip_prob),
            shift_sigma=float(rng.uniform(*self.shift_sigma_range)),
            stuck_sigma=float(rng.uniform(*self.stuck_sigma_range)),
            noise_inflate=float(rng.uniform(*self.noise_inflate_range)),
            cascade_sigma=float(rng.uniform(*self.cascade_sigma_range)),
            cascade_delay_per_hop=int(
                rng.integers(
                    self.cascade_delay_per_hop_range[0],
                    self.cascade_delay_per_hop_range[1] + 1,
                )),
            cascade_attenuation=float(self.cascade_attenuation),
            cascade_branch_prob=float(self.cascade_branch_prob),
        )


def _norm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p / p.sum()


# =============================================================================
# Activations (TabICL _activations, numpy)
# =============================================================================


def _act_identity(x):
    return x


def _act_tanh(x):
    return np.tanh(x)


def _act_relu(x):
    return np.maximum(x, 0.0)


def _act_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _act_sin(x):
    return np.sin(x)


def _act_abs(x):
    return np.abs(x)


def _act_square(x):
    return np.clip(x * x, 0.0, 8.0)


def _act_softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _act_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


_ACTIVATIONS = {
    "identity": _act_identity,
    "tanh": _act_tanh,
    "relu": _act_relu,
    "sigmoid": _act_sigmoid,
    "sin": _act_sin,
    "abs": _act_abs,
    "square": _act_square,
    "softplus": _act_softplus,
    "gelu": _act_gelu,
}
_ACTIVATION_NAMES = list(_ACTIVATIONS)

_WARP_KINDS = ("rank_gauss", "kumaraswamy", "signed_log", "power", "exp",
               "affine_abs")

# =============================================================================
# Root ("cause") sampler  (TabICL XSampler)
# =============================================================================


def build_root_sampler(kind: str, rng: np.random.Generator):
    """Return a ``callable(n, rng) -> ndarray`` for one root node.

    The distribution's PARAMETERS (modes, mixture weights, ...) are drawn ONCE
    here and frozen.  Every ``obs`` / ``int`` window of the same episode then
    samples from the *same* frozen distribution -- otherwise the observational
    and interventional windows of a root node would differ for reasons that have
    nothing to do with the intervention.
    """
    if kind == "normal":
        return lambda n, r: r.normal(0.0, 1.0, n)
    if kind == "uniform":
        b = math.sqrt(3.0)
        return lambda n, r: r.uniform(-b, b, n)
    if kind == "laplace":
        s = 1.0 / math.sqrt(2.0)
        return lambda n, r: r.laplace(0.0, s, n)
    if kind == "student_t":
        df = int(rng.integers(3, 9))
        sc = math.sqrt(df / (df - 2.0))
        return lambda n, r: r.standard_t(df, n) / sc
    if kind == "multimodal":
        m = int(rng.integers(2, 5))
        centers = rng.uniform(-3.0, 3.0, m)
        weights = rng.dirichlet(np.ones(m))
        sd = float(rng.uniform(0.2, 0.6))

        def _mm(n, r):
            idx = r.choice(m, size=n, p=weights)
            return centers[idx] + r.normal(0.0, sd, n)

        return _mm
    if kind == "mixed":
        base_norm = rng.random() < 0.8
        mu = float(rng.uniform(-2.0, 2.0))
        sd = float(rng.uniform(0.3, 1.2))
        add_uniform = rng.random() < 0.5

        def _mix(n, r):
            parts = [
                r.normal(mu, sd, n) if base_norm else r.normal(0.0, 1.0, n)
            ]
            if add_uniform:
                parts.append(r.uniform(-2.0, 2.0, n))
            return sum(parts) / math.sqrt(len(parts))

        return _mix
    return lambda n, r: r.normal(0.0, 1.0, n)


# =============================================================================
# Structural equations  (immutable after construction / first fit)
# =============================================================================


def _block_dropout(W: np.ndarray, sparsity: float,
                   rng: np.random.Generator) -> np.ndarray:
    """TabICL block-wise dropout: zero out random rectangular blocks of W."""
    out = W.copy()
    r, c = W.shape
    n_blocks = max(1, int(round(sparsity * 4)))
    for _ in range(n_blocks):
        r0 = rng.integers(0, r)
        c0 = rng.integers(0, c)
        r1 = min(r, r0 + int(rng.integers(1, max(2, r // 2 + 1))))
        c1 = min(c, c0 + int(rng.integers(1, max(2, c // 2 + 1))))
        out[r0:r1, c0:c1] = 0.0
    # guarantee every output unit keeps at least one input
    for j in range(c):
        if not np.any(out[:, j]):
            out[rng.integers(0, r),
                j] = W[rng.integers(0, r), j] or rng.normal()
    return out


class MLPEquation:
    """Small randomly-initialised MLP mechanism  y = MLP(parents) + noise."""

    def __init__(self, n_parents: int, hp: dict, rng: np.random.Generator):
        self.n_parents = int(n_parents)
        in_dim = max(self.n_parents, 1)
        hidden = max(2, hp["mlp_hidden_dim"])
        n_layers = max(1, hp["mlp_num_layers"])
        dims = [in_dim] + [hidden] * (n_layers - 1) + [1]

        self.W, self.b, self.acts = [], [], []
        for i in range(len(dims) - 1):
            w = rng.normal(0.0,
                           hp["mlp_init_std"],
                           size=(dims[i], dims[i + 1]))
            if hp["mlp_block_dropout"]:
                w = _block_dropout(w, hp["mlp_sparsity"], rng)
            else:
                mask = rng.random(w.shape) > hp["mlp_sparsity"]
                mask |= ~mask.any(axis=0, keepdims=True)  # keep >=1 per column
                w = w * mask
            self.W.append(w)
            self.b.append(
                rng.normal(0.0, 0.1 * hp["mlp_init_std"], size=dims[i + 1]))
            self.acts.append(_ACTIVATIONS[hp["mlp_activation"]] if i <
                             len(dims) - 2 else _act_identity)
        self.noise_std = hp["mlp_noise_std"]
        self._in_mean = None
        self._in_std = None

    def forward_batch(self, parents, noise, rng):
        if self._in_mean is None:  # fit standardisation on FIRST (observational) call
            self._in_mean = parents.mean(axis=0, keepdims=True)
            self._in_std = parents.std(axis=0, keepdims=True) + 1e-8
        h = (parents - self._in_mean) / self._in_std
        for k, (w, b, act) in enumerate(zip(self.W, self.b, self.acts)):
            h = h @ w + b
            if k < len(self.W) - 1:
                h = np.clip(act(h), -8.0, 8.0)
                h = h + rng.normal(0.0, self.noise_std, size=h.shape)
        return h[:, 0] + noise


class TreeEquation:
    """TabICL TreeSCM mechanism: a tree regressor fitted to random targets."""

    def __init__(self, hp: dict, rng: np.random.Generator):
        self.kind = hp["tree_kind"]
        self.max_depth = int(hp["tree_depth"])
        self.n_estimators = int(hp["tree_n_estimators"])
        self.noise_std = hp["tree_noise_std"]
        self.seed = int(rng.integers(0, 2**31 - 1))
        self._model = None
        self._in_mean = None
        self._in_std = None
        self._w = None

    def _make(self):
        if self.kind == "dt":
            return DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.seed)
        if self.kind == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
                n_jobs=1,
            )
        if self.kind == "et":
            return ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
                n_jobs=1,
            )
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=min(self.max_depth, 4),
            random_state=self.seed,
        )

    def forward_batch(self, parents, noise, rng):
        n, p = parents.shape
        if self._model is None:
            self._in_mean = parents.mean(axis=0, keepdims=True)
            self._in_std = parents.std(axis=0, keepdims=True) + 1e-8
            xs = (parents - self._in_mean) / self._in_std
            self._w = rng.normal(0.0, 1.0, size=p)
            # random-but-structured targets: linear trend + noise -> tree carves
            # a piecewise-constant approximation (threshold-style mechanism)
            y = xs @ self._w + rng.normal(0.0, 0.5, size=n)
            if n >= 4:
                self._model = self._make().fit(xs, y)
            else:  # too few points to fit a tree -> fall back to the linear part
                self._model = None
                self._lin_only = True
        xs = (parents - self._in_mean) / self._in_std
        if self._model is None:
            out = xs @ self._w
        else:
            out = self._model.predict(xs)
        return out + noise * (1.0 + self.noise_std)


class LinearEquation:
    """Plain linear mechanism (kept from the existing prior for diversity)."""

    def __init__(self, n_parents: int, rng: np.random.Generator):
        self.weights = rng.normal(0.0, 2.0, size=max(n_parents, 1))
        self._in_mean = None
        self._in_std = None

    def forward_batch(self, parents, noise, rng):
        if self._in_mean is None:
            self._in_mean = parents.mean(axis=0, keepdims=True)
            self._in_std = parents.std(axis=0, keepdims=True) + 1e-8
        xs = (parents - self._in_mean) / self._in_std
        return xs @ self.weights + noise


class GPEquationWrapper:
    """Thin adapter around the repo's GPEquation with fixed obs standardisation."""

    def __init__(self, rng: np.random.Generator):
        self.gp = GPEquation()
        self.gp._init_hyperparams()
        self._in_mean = None
        self._in_std = None

    def forward_batch(self, parents, noise, rng):
        if self._in_mean is None:
            self._in_mean = parents.mean(axis=0, keepdims=True)
            self._in_std = parents.std(axis=0, keepdims=True) + 1e-8
        xs = (parents - self._in_mean) / self._in_std
        return self.gp.forward_batch(xs, noise)


# =============================================================================
# SCM
# =============================================================================


class TabICLSCM:
    """Structural causal model over an explicit DAG with a mixed function class."""

    def __init__(self, dag: nx.DiGraph, hp: dict, rng: np.random.Generator):
        self.dag = dag
        self.hp = hp
        self.n_nodes = dag.number_of_nodes()
        self.topo = list(nx.topological_sort(dag))
        self.noise_type = hp["noise_type"]
        self.noise_scale = hp["noise_scale"]
        self.temporal_autocorr = hp["temporal_autocorr"]
        self.availability = hp["availability"]
        self.availability_baseline = hp["availability_baseline"]

        # per-node driving-noise scale overrides (used by the "noise" intervention)
        self.noise_overrides: dict[int, float] = {}
        # output-level intervention overrides, applied in topological order so
        # they PROPAGATE to descendants
        self.do_overrides: dict[int, float] = {}
        self.gain_overrides: dict[int,
                                  tuple] = {}  # node -> (mu, factor, sign)
        self.shift_overrides: dict[int, float] = {}  # node -> additive shift
        # causRCA-style temporal cascade: node -> (onset_step, signed_magnitude)
        self.cascade_spec: Optional[dict] = None

        self.equations: dict[int, object] = {}
        self.root_samplers: dict[int, object] = {}
        for node in self.dag.nodes:
            n_par = self.dag.in_degree(node)
            if n_par == 0:
                self.equations[node] = None  # root -> frozen root distribution
                self.root_samplers[node] = build_root_sampler(
                    hp["root_kind"], rng)
                continue
            self.equations[node] = self._make_equation(n_par, rng)

    def _make_equation(self, n_parents: int, rng: np.random.Generator):
        et = self.hp["equation_type"]
        if et == "mlp":
            return MLPEquation(n_parents, self.hp, rng)
        if et == "tree":
            return TreeEquation(self.hp, rng)
        if et == "gp":
            return GPEquationWrapper(rng)
        return LinearEquation(n_parents, rng)

    # -- sampling --------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        n_nodes = self.n_nodes
        data = np.zeros((n, n_nodes), dtype=np.float64)

        # driving randomness: root nodes -> XSampler draw, others -> noise
        drive = _sample_noise(self.noise_type, self.noise_scale, (n, n_nodes))
        for node in self.topo:
            if self.dag.in_degree(node) == 0:
                drive[:, node] = self.root_samplers[node](n, rng)
        for node, scale in self.noise_overrides.items():
            drive[:, node] = _sample_noise(self.noise_type, scale, n)

        # AR(1) temporal autocorrelation (variance preserving)
        rho = self.temporal_autocorr
        if rho > 0.0:
            ar = np.empty_like(drive)
            ar[0] = drive[0]
            s = math.sqrt(max(0.0, 1.0 - rho * rho))
            for t in range(1, n):
                ar[t] = rho * ar[t - 1] + s * drive[t]
            drive = ar

        base = self.availability_baseline if self.availability else 0.0

        for node in self.topo:
            if self.dag.in_degree(node) == 0:
                val = base + drive[:, node] * (self.noise_scale
                                               if self.availability else 1.0)
            else:
                parents = list(self.dag.predecessors(node))
                eq = self.equations[node]
                pv = data[:, parents]
                if self.availability:
                    pv = pv - base  # propagate deviations, not absolute levels
                val = eq.forward_batch(pv, drive[:, node], rng)
                if self.availability:
                    val = base + 0.3 * val + drive[:, node] * self.noise_scale

            # ---- interventions, applied here so descendants see them --------
            if node in self.do_overrides:
                val = np.full(n, self.do_overrides[node], dtype=np.float64)
            elif node in self.gain_overrides:
                mu, factor, sign = self.gain_overrides[node]
                val = mu + sign * factor * (val - mu)
            if node in self.shift_overrides:
                val = val + self.shift_overrides[node]

            data[:, node] = val

        return data

    # -- interventions --------------------------------------------------

    def copy(self) -> "TabICLSCM":
        new = copy.copy(self)  # equations are immutable -> shared
        new.noise_overrides = dict(self.noise_overrides)
        new.do_overrides = dict(self.do_overrides)
        new.gain_overrides = dict(self.gain_overrides)
        new.shift_overrides = dict(self.shift_overrides)
        new.cascade_spec = dict(
            self.cascade_spec) if self.cascade_spec else None
        return new

    def intervened(
        self,
        node: int,
        kind: str,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        hp: dict,
        rng: np.random.Generator,
    ) -> "TabICLSCM":
        """Return a COPY of this SCM with a single intervention on ``node``.

        ``obs_mean`` / ``obs_std`` are the per-node statistics of the
        OBSERVATIONAL window -- used to scale the intervention so it is a
        meaningful outlier regardless of the node's native scale.
        """
        new = self.copy()
        mu, sd = float(obs_mean[node]), float(obs_std[node])
        direction = float(rng.choice([-1.0, 1.0]))

        # On an availability / constant-baseline node the noise is tiny, so a
        # sigma-scaled fault is invisible in absolute terms.  Instead drop the
        # metric by a fraction of its baseline (old prior: apply_shift/do
        # intervention high_baseline branch) -> the z-score saturates the
        # +/-10 clip, the availability-fault signature.
        avail = self.availability
        base = self.availability_baseline
        frac = hp.get("availability_fault_frac", 0.1)

        if kind == "weight":
            sign = -1.0 if hp["weight_flip"] else 1.0
            new.gain_overrides[node] = (mu, hp["weight_change"], sign)
        elif kind == "shift":
            new.shift_overrides[node] = (-frac * base if avail else direction *
                                         hp["shift_sigma"] * sd)
        elif kind == "do":
            new.do_overrides[node] = (base * (1.0 - frac) if avail else mu +
                                      direction * hp["stuck_sigma"] * sd)
        elif kind == "noise":
            new.noise_overrides[node] = self.noise_scale * hp["noise_inflate"]
        elif kind == "cascade":
            # Store (onset_step, signed magnitude in std-units).  The magnitude is
            # applied POST-warp, in the z-space the model sees (see
            # apply_cascade_post_warp), so a compressive feature warp can't
            # swallow the fault signal.
            reach = nx.single_source_shortest_path_length(self.dag, node)
            sigma = hp["cascade_sigma"]
            dph = hp["cascade_delay_per_hop"]
            atten = hp["cascade_attenuation"]
            bprob = hp["cascade_branch_prob"]
            spec: dict[int, tuple] = {}
            for v, d in reach.items():
                if d > 0 and rng.random() >= bprob**d:
                    continue  # this branch does not cascade
                spec[v] = (d * dph, direction * sigma * (atten**d))
            new.cascade_spec = {
                **(new.cascade_spec or {}),
                **spec
            }  # merge (multi-RC)
        else:
            raise ValueError(f"unknown intervention kind: {kind!r}")
        return new


# =============================================================================
# Feature realism  (TabICL _reg2cls / _num2cat) -- fitted on OBS only
# =============================================================================


@dataclass
class ColumnTransform:
    """A per-column warp (+ optional discretisation) fitted on the obs window."""

    warp_kind: str = "identity"
    params: dict = field(default_factory=dict)
    discretise: bool = False
    bin_edges: Optional[np.ndarray] = None
    n_categories: int = 0

    # -- fit ----------------------------------------------------------
    @classmethod
    def fit(cls, col: np.ndarray, hp: dict,
            rng: np.random.Generator) -> "ColumnTransform":
        t = cls()
        x = col.astype(np.float64)

        if rng.random() < hp["warp_prob"]:
            t.warp_kind = str(rng.choice(_WARP_KINDS))
            lo, hi = np.percentile(x, [1, 99])
            t.params = dict(
                lo=float(lo),
                hi=float(hi),
                mn=float(x.min()),
                mx=float(x.max()),
                mean=float(x.mean()),
                std=float(x.std() + 1e-8),
                a=float(np.exp(rng.uniform(-0.7, 0.7))),
                b=float(np.exp(rng.uniform(-0.7, 0.7))),
                sorted=np.sort(x) if t.warp_kind == "rank_gauss" else None,
                power=float(rng.choice([2.0, 3.0, 0.5, 1.5])),
            )

        warped = t._warp(x)

        if rng.random() < hp["cat_prob"]:
            n_cat = int(
                np.clip(round(rng.gamma(1.0, 10.0)), 2, hp["max_categories"]))
            qs = np.linspace(0.0, 1.0, n_cat + 1)[1:-1]
            edges = np.unique(np.quantile(warped, qs))
            if edges.size >= 1:
                t.discretise = True
                t.bin_edges = edges
                t.n_categories = int(edges.size + 1)
        return t

    # -- apply (identical transform for obs and int) ----------------
    def _warp(self, x: np.ndarray) -> np.ndarray:
        k, p = self.warp_kind, self.params
        if k == "identity" or not p:
            return x
        if k == "rank_gauss":
            srt = p["sorted"]
            ranks = np.searchsorted(srt, x, side="right") / (len(srt) + 1.0)
            ranks = np.clip(ranks, 1e-4, 1 - 1e-4)
            return _norm_ppf(ranks)
        if k == "kumaraswamy":
            u = np.clip((x - p["mn"]) / (p["mx"] - p["mn"] + 1e-12), 1e-6,
                        1 - 1e-6)
            return 1.0 - (1.0 - u**p["a"])**p["b"]
        if k == "signed_log":
            return np.sign(x) * np.log1p(np.abs(x))
        if k == "power":
            z = (x - p["mean"]) / p["std"]
            return np.sign(z) * (np.abs(z)**p["power"])
        if k == "exp":
            z = np.clip((x - p["mean"]) / p["std"], -4, 4)
            return np.exp(z)
        if k == "affine_abs":
            return p["a"] * np.abs(x - p["mean"]) + p["b"] * x
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = self._warp(x.astype(np.float64))
        if self.discretise and self.bin_edges is not None:
            out = np.digitize(out, self.bin_edges).astype(np.float64)
        return out


def _norm_ppf(q: np.ndarray) -> np.ndarray:
    """Vectorised standard-normal quantile (Acklam's rational approximation)."""
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    q = np.clip(q, 1e-10, 1 - 1e-10)
    out = np.empty_like(q)
    lo, hi = q < 0.02425, q > 1 - 0.02425
    mid = ~(lo | hi)
    for mask, sign, qq in ((lo, 1.0, q), (hi, -1.0, 1.0 - q)):
        r = np.sqrt(-2.0 * np.log(qq[mask])) if np.any(mask) else np.array([])
        if r.size:
            out[mask] = (
                sign *
                (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r +
                 c[5]) /
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0))
    if np.any(mid):
        r = q[mid] - 0.5
        rr = r * r
        out[mid] = ((
            ((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr +
            a[5]) * r / ((((
                (b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr
                         + 1.0))
    return out


def fit_column_transforms(obs_raw: np.ndarray, hp: dict,
                          rng: np.random.Generator):
    return [
        ColumnTransform.fit(obs_raw[:, j], hp, rng)
        for j in range(obs_raw.shape[1])
    ]


def apply_column_transforms(x: np.ndarray,
                            transforms) -> tuple[np.ndarray, np.ndarray]:
    out = np.empty_like(x, dtype=np.float64)
    discrete = np.zeros(x.shape[1], dtype=bool)
    for j, t in enumerate(transforms):
        out[:, j] = t.transform(x[:, j])
        discrete[j] = t.discretise
    return out, discrete


def apply_cascade_post_warp(int_w: np.ndarray, cascade_spec: dict,
                            obs_w: np.ndarray) -> np.ndarray:
    """Inject the causRCA-style cascade into the WARPED interventional window.

    Done here (not inside the SCM) so a compressive feature warp cannot swallow
    the fault: each affected node ``v`` is shifted by
    ``unit_mag * std(obs_w[:, v])`` from its onset step onward.  In a short
    window the root cause (onset 0, whole window) ends up with a bigger mean
    shift than a leaf several hops down (onset late, attenuated).
    """
    out = int_w.copy()
    t = np.arange(out.shape[0])
    col_std = obs_w.std(axis=0) + 1e-8
    for v, (onset, unit_mag) in cascade_spec.items():
        out[:, v] = out[:, v] + unit_mag * col_std[v] * (t >= onset)
    return out


# =============================================================================
# Normalisation  --  THE choke point.  Observational statistics ONLY.
# =============================================================================


def normalize_observational_only(
    obs_raw: np.ndarray,
    int_raw: np.ndarray,
    n_nodes: int,
    max_nodes: int,
    clip: float = CLIP,
):
    """z-score BOTH windows with the OBSERVATIONAL mean/std, clip, pad, scale K/k.

    This is the only place normalisation happens.  ``int_raw`` is standardised
    with ``obs`` statistics on purpose: that is what keeps the intervention's
    mean shift visible to the model.  Normalising ``int_raw`` with its own
    statistics would erase the RCA signal.
    """
    obs_mean = obs_raw.mean(axis=0, keepdims=True)
    obs_std = obs_raw.std(axis=0, keepdims=True) + 1e-8

    obs_z = np.clip((obs_raw - obs_mean) / obs_std, -clip, clip)
    int_z = np.clip((int_raw - obs_mean) / obs_std, -clip, clip)  # obs stats!

    scale = max_nodes / n_nodes
    obs_p = np.zeros((obs_z.shape[0], max_nodes), dtype=np.float32)
    int_p = np.zeros((int_z.shape[0], max_nodes), dtype=np.float32)
    obs_p[:, :n_nodes] = (obs_z * scale).astype(np.float32)
    int_p[:, :n_nodes] = (int_z * scale).astype(np.float32)
    return obs_p, int_p


# =============================================================================
# DAG helper
# =============================================================================


def _pick_aggregators(dag: nx.DiGraph, frac: float,
                      rng: np.random.Generator) -> list:
    """A few high-in-degree SINK nodes ("aggregators" / SLIs) -- the kind of
    node a monitoring alarm actually fires on.  Works on any DAG topology."""
    sinks = sorted(
        (n for n in dag.nodes if dag.out_degree(n) == 0),
        key=lambda n: -dag.in_degree(n),
    )
    n_agg = max(1, round(frac * dag.number_of_nodes()))
    agg = [int(s) for s in sinks[:n_agg] if dag.in_degree(s) > 0]
    return agg or [int(s) for s in sinks[:1]]


def _sample_dag(hp: dict, k: int,
                rng: np.random.Generator) -> tuple[nx.DiGraph, list, list]:
    edge_prob = min(hp["expected_degree"] / (k - 1), 0.95) if k > 1 else 0.5
    for _ in range(20):
        # generate_dag uses the legacy global RNG -> reseed it for reproducibility
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        random.seed(int(rng.integers(0, 2**31 - 1)))
        dag = generate_dag(k, edge_prob, hp["graph_family"])
        non_leaf = [nd for nd in dag.nodes if dag.out_degree(nd) > 0]
        if not non_leaf:
            non_leaf = [nd for nd in dag.nodes if dag.in_degree(nd) > 0]
        if non_leaf:
            aggregators = _pick_aggregators(dag, hp["aggregator_frac"], rng)
            return dag, non_leaf, aggregators
    raise RuntimeError("could not sample a usable DAG")


def _symptom_node(dag: nx.DiGraph, intervention_node: int,
                  rng: np.random.Generator) -> int:
    """Observed symptom (= the "alarm") -- always a STRICT descendant of the
    root cause, biased toward leaves.

    The symptom is NEVER the root cause itself.  This is the causRCA / PetShop
    task: an alarm fires on some downstream node and you must trace back UP the
    graph to the root cause.  Letting symptom == root cause taught the model a
    "just predict the alarm" shortcut that collapses on real data, where the
    alarm is given and is never the answer (cf.
    finetune_causrca._pick_root_cause_and_alarm + its alarm-penalty loss).

    ``intervention_node`` is guaranteed non-leaf by ``_sample_dag`` so it always
    has at least one descendant.
    """
    desc = list(nx.descendants(dag, intervention_node))
    if not desc:  # defensive; should not happen
        return int(intervention_node)
    leaves = [d for d in desc if dag.out_degree(d) == 0]
    if leaves and rng.random() < 0.7:
        return int(rng.choice(leaves))
    return int(rng.choice(desc))


def _confine_pool(dag: nx.DiGraph, valid_nodes, hp: dict,
                  rng: np.random.Generator) -> list:
    """With ``confine_fault_prob``, restrict the RC candidate pool to a local
    (undirected) neighbourhood around a random anchor node -- a fault confined
    to one "subsystem", on ANY graph topology (no modular graph construction
    needed).  On a small graph the neighbourhood covers ~the whole graph
    (sub-system-like); on a large graph it's a small fraction, and the rest of
    the graph stays active but UNSHIFTED (full-system-like).
    """
    if rng.random() >= hp["confine_fault_prob"] or len(valid_nodes) <= 6:
        return list(valid_nodes)
    anchor = int(rng.choice(valid_nodes))
    und = dag.to_undirected()
    nbhd = set(
        nx.single_source_shortest_path_length(und,
                                              anchor,
                                              cutoff=hp["confine_radius"]))
    pool = [n for n in valid_nodes if n in nbhd]
    return pool if len(pool) >= 2 else list(valid_nodes)


def _pick_independent_rcs(dag: nx.DiGraph, valid_nodes, n_faults: int,
                          hp: dict, rng: np.random.Generator) -> list[int]:
    """Pick up to ``n_faults`` root causes that are causally INDEPENDENT --
    none is an ancestor or descendant of another, so each is a genuine separate
    fault (cf. finetune_causrca._pick_root_cause_and_alarm /
    test_multi_rca._MultiRCEnv._pick_scenario).  The pool is first confined to
    a local neighbourhood (see ``_confine_pool``), and a graph SOURCE
    (in_degree 0) is preferred so there is a real chain to trace up from the
    alarm.  Always returns >= 1.
    """
    pool = _confine_pool(dag, valid_nodes, hp, rng)
    rng.shuffle(pool)
    if rng.random() < hp["rc_from_source_prob"]:
        pool.sort(key=lambda n: dag.in_degree(n))  # sources first

    rcs: list[int] = []
    excluded: set = set()
    for c in pool:
        if c in excluded:
            continue
        rcs.append(int(c))
        excluded |= nx.descendants(dag, c) | nx.ancestors(dag, c) | {c}
        if len(rcs) >= n_faults:
            break
    return rcs or [int(pool[0])]


def _multi_symptom(
    dag: nx.DiGraph,
    rcs: list[int],
    aggregators: list,
    hp: dict,
    rng: np.random.Generator,
) -> int:
    """Observed alarm for a (possibly multi-) root-cause episode.

    With ``aggregator_symptom_prob``, the alarm is a fault-reachable
    AGGREGATOR node (an SLI where a monitor actually fires) -- the farthest
    one, for a longer trace.  Otherwise: a shared descendant of ALL root
    causes if one exists (leaf-biased), else a strict descendant of one
    randomly chosen root cause.  Never a root cause itself.
    """
    reach: set = set()
    for r in rcs:
        reach |= nx.descendants(dag, r)
    reach -= set(rcs)

    agg = [a for a in aggregators if a in reach]
    if agg and rng.random() < hp["aggregator_symptom_prob"]:
        dist = {}
        for a in agg:
            ds = [
                nx.shortest_path_length(dag, r, a) for r in rcs
                if nx.has_path(dag, r, a)
            ]
            if ds:
                dist[a] = min(ds)
        if dist:
            dmax = max(dist.values())
            return int(
                rng.choice([a for a, d in dist.items() if d >= dmax - 1]))
        return int(rng.choice(agg))

    common = None
    for r in rcs:
        d = set(nx.descendants(dag, r))
        common = d if common is None else (common & d)
    common = (common or set()) - set(rcs)
    if common:
        leaves = [n for n in common if dag.out_degree(n) == 0]
        pool = leaves if (leaves and rng.random() < 0.7) else list(common)
        return int(rng.choice(pool))
    return _symptom_node(dag, int(rng.choice(rcs)), rng)


def _apply_multi_intervention(scm, rcs, kind, obs_mean, obs_std, hp, rng):
    """Chain one intervention of ``kind`` per root cause; faults propagate
    independently through the SCM equations."""
    sc = scm
    for rc in rcs:
        sc = sc.intervened(rc, kind, obs_mean, obs_std, hp, rng)
    return sc


def _sample_n_faults(cfg: "TabICLPriorConfig",
                     rng: np.random.Generator) -> int:
    return int(rng.choice([1, 2, 3], p=_norm(cfg.n_faults_probs)))


def _cascade_window(hp: dict, n_int: int, cfg: "TabICLPriorConfig",
                    rng: np.random.Generator) -> int:
    """A cascade fault needs a SHORT interventional window: the hop lag only
    produces a root-cause-vs-alarm mean-shift gradient while a meaningful
    fraction of the window predates the alarm's onset.  A long window washes
    it out (every affected node ends up faulted for ~100% of steps)."""
    if hp["intervention_kind"] != "cascade":
        return n_int
    lo, hi = cfg.cascade_n_int_range
    return int(rng.integers(lo, hi + 1))


# =============================================================================
# Sparse embedding  --  full-scale causRCA structure
# =============================================================================
#
# In full-mode causRCA evaluation the graph has ~92 monitored signals but the
# fault (+ its cascade) touches only a small cluster; the rest belong to
# uninvolved subsystems and are forward-filled CONSTANTS.  After obs-only
# normalisation those columns are ~0 in both windows.  A model trained only on
# dense graphs (every node active + noisy) has never seen "small active
# subgraph in a sea of zero columns" and collapses on full-mode.
#
# `SparseLayout` embeds a dense k_active-node SCM at random positions inside a
# k_total-node grid, filling the rest with silent columns (mostly exact
# constants, some with tiny noise -- mirrors FullScaleEnvironment /
# MultiSubsystemEnvironment in prim/finetuning/finetune_causrca.py).


@dataclass
class SparseLayout:
    k_total: int
    active_pos: np.ndarray  # sorted indices of the active subgraph
    silent_baseline: np.ndarray  # (k_total,) constant level of each column
    silent_std: np.ndarray  # (k_total,) 0 at active + constant-silent cols

    @classmethod
    def sample(cls, k_active: int, k_total: int,
               rng: np.random.Generator) -> "SparseLayout":
        active = np.sort(rng.permutation(k_total)[:k_active])
        baseline = rng.uniform(-3.0, 3.0, k_total)
        # 70% of silent columns are exact constants (-> 0 after z-norm),
        # 30% carry a tiny measurement noise (log-uniform std in [1e-3, 1e-1]).
        std = np.where(
            rng.random(k_total) < 0.7,
            0.0,
            10.0**rng.uniform(-3.0, -1.0, k_total),
        )
        std[active] = 0.0
        baseline[active] = 0.0
        return cls(k_total, active, baseline, std)

    def embed(self, obs_a, int_a, disc_a, rc_a, sym_a, dag_a, rng):
        """Place the active arrays into the full grid; fill silent columns."""
        n_o, n_i, K = obs_a.shape[0], int_a.shape[0], self.k_total
        obs_f = (self.silent_baseline[None, :] + rng.standard_normal(
            (n_o, K)) * self.silent_std[None, :])
        int_f = (self.silent_baseline[None, :] + rng.standard_normal(
            (n_i, K)) * self.silent_std[None, :])
        obs_f[:, self.active_pos] = obs_a
        int_f[:, self.active_pos] = int_a
        disc_f = np.zeros(K, dtype=bool)
        disc_f[self.active_pos] = disc_a
        remap = {
            old: int(self.active_pos[old])
            for old in range(len(self.active_pos))
        }
        dag_f = nx.relabel_nodes(dag_a, remap)
        return (
            obs_f,
            int_f,
            disc_f,
            int(self.active_pos[rc_a]),
            int(self.active_pos[sym_a]),
            dag_f,
        )


def _maybe_sparse_layout(cfg: "TabICLPriorConfig", k_total: int,
                         rng: np.random.Generator):
    """Return a SparseLayout with prob ``cfg.sparse_embed_prob``, else None.

    ``k_total`` is the graph the model is told about; the active subgraph is
    ``sparse_active_range`` nodes.  Needs enough headroom for the silent block.
    """
    lo, hi = cfg.sparse_active_range
    if k_total <= hi + 4 or rng.random() >= cfg.sparse_embed_prob:
        return None, k_total
    k_active = int(rng.integers(lo, min(hi, k_total - 4) + 1))
    return SparseLayout.sample(k_active, k_total, rng), k_active


# =============================================================================
# Public API
# =============================================================================


def generate_observational_and_interventional_data(
    n_nodes: int = 15,
    n_obs_samples: int = 300,
    n_int_samples: int = 100,
    intervention_node: Optional[int] = None,
    config: Optional[TabICLPriorConfig] = None,
    seed: Optional[int] = None,
    return_extra: bool = False,
):
    """Drop-in-style replacement for the same-named function in
    ``dag_generator_simplified``.

    Returns RAW (un-normalised, post feature-realism) arrays so that the caller
    (e.g. ``generate_dataset.py``) can apply its own observational-only
    normalisation -- exactly like the existing prior.

    This path is always DENSE (no sparse full-scale-causRCA embedding -- that
    would break the fixed-shape array contract of ``generate_dataset.py``).
    Use ``generate_rca_episode`` / ``TabICLRCADataGenerator`` for the
    sparse-embedded episodes.

    Returns
    -------
    obs_data : (n_obs_samples, n_nodes)
    int_data : (n_int_samples, n_nodes)
    dag      : networkx.DiGraph
    intervention_node : int
    (extra) : dict with ``discrete_mask``, ``symptom_node``, ``hp`` when
              ``return_extra=True``
    """
    cfg = config or TabICLPriorConfig()
    rng = np.random.default_rng(seed)
    hp = cfg.sample(rng)

    dag, valid_nodes, aggregators = _sample_dag(hp, n_nodes, rng)
    scm = TabICLSCM(dag, hp, rng)

    obs_raw = scm.sample(n_obs_samples, rng)

    if intervention_node is None:
        intervention_node = int(rng.choice(valid_nodes))

    obs_mean = obs_raw.mean(axis=0)
    obs_std = obs_raw.std(axis=0) + 1e-8
    int_scm = scm.intervened(intervention_node, hp["intervention_kind"],
                             obs_mean, obs_std, hp, rng)
    int_raw = int_scm.sample(n_int_samples, rng)

    # feature realism: fit warps + discretisation on the OBSERVATIONAL window,
    # apply the identical transform to both windows
    transforms = fit_column_transforms(obs_raw, hp, rng)
    obs_raw, disc = apply_column_transforms(obs_raw, transforms)
    int_raw, _ = apply_column_transforms(int_raw, transforms)
    if int_scm.cascade_spec:
        int_raw = apply_cascade_post_warp(int_raw, int_scm.cascade_spec,
                                          obs_raw)

    if not return_extra:
        return obs_raw, int_raw, dag, intervention_node

    extra = dict(
        discrete_mask=disc,
        symptom_node=_multi_symptom(dag, [intervention_node], aggregators, hp,
                                    rng),
        aggregators=aggregators,
        hp=hp,
        intervention_kind=hp["intervention_kind"],
    )
    return obs_raw, int_raw, dag, intervention_node, extra


def generate_rca_episode(
    config: Optional[TabICLPriorConfig] = None,
    seed: Optional[int] = None,
) -> dict:
    """Generate one fully-encoded RCA episode (obs + int + labels + masks).

    All arrays are PFN-encoded: padded to ``config.max_nodes`` and scaled by
    ``max_nodes / k``.  Normalisation uses observational statistics only.
    """
    cfg = config or TabICLPriorConfig()
    rng = np.random.default_rng(seed)
    hp = cfg.sample(rng)

    k = int(rng.integers(cfg.n_nodes_range[0], cfg.n_nodes_range[1] + 1))
    n_obs = int(rng.integers(*cfg.n_obs_range))
    n_int = int(rng.integers(*cfg.n_int_range))
    K = cfg.max_nodes

    layout, k_active = _maybe_sparse_layout(cfg, k, rng)

    dag, valid_nodes, aggregators = _sample_dag(hp, k_active, rng)
    scm = TabICLSCM(dag, hp, rng)

    obs_raw = scm.sample(n_obs, rng)
    rcs = _pick_independent_rcs(dag, valid_nodes, _sample_n_faults(cfg, rng),
                                hp, rng)
    obs_mean, obs_std = obs_raw.mean(axis=0), obs_raw.std(axis=0) + 1e-8
    int_scm = _apply_multi_intervention(scm, rcs, hp["intervention_kind"],
                                        obs_mean, obs_std, hp, rng)
    int_raw = int_scm.sample(_cascade_window(hp, n_int, cfg, rng), rng)

    transforms = fit_column_transforms(obs_raw, hp, rng)
    obs_raw, disc = apply_column_transforms(obs_raw, transforms)
    int_raw, _ = apply_column_transforms(int_raw, transforms)
    if int_scm.cascade_spec:
        int_raw = apply_cascade_post_warp(int_raw, int_scm.cascade_spec,
                                          obs_raw)

    symptom = _multi_symptom(dag, rcs, aggregators, hp, rng)

    if layout is not None:
        # remap active-subgraph indices -> full grid (rc_a/sym_a placeholders)
        obs_raw, int_raw, disc, _rc0, symptom, dag = layout.embed(
            obs_raw, int_raw, disc, rcs[0], symptom, dag, rng)
        rcs = [int(layout.active_pos[r]) for r in rcs]
        n_model = layout.k_total
    else:
        n_model = k_active

    obs_p, int_p = normalize_observational_only(obs_raw, int_raw, n_model, K)

    label = np.zeros(K, dtype=np.int64)  # MULTI-HOT
    for r in rcs:
        label[r] = 1
    target_mask = np.zeros(K, dtype=np.float32)
    target_mask[symptom] = 1.0
    discrete_mask = np.zeros(K, dtype=bool)
    discrete_mask[:n_model] = disc

    return dict(
        observational_data=obs_p,
        interventional_data=int_p,
        intervention_label=label,
        intervention_node=int(rcs[0]),  # first RC (back-compat)
        intervention_nodes=[int(r) for r in rcs],
        n_faults=len(rcs),
        target_mask=target_mask,
        symptom_node=symptom,
        discrete_mask=discrete_mask,
        n_nodes=n_model,
        dag_edges=list(dag.edges()),
        equation_type=hp["equation_type"],
        intervention_kind=hp["intervention_kind"],
        graph_family=hp["graph_family"],
        aggregators=[int(a) for a in aggregators],
    )


# =============================================================================
# Training adapter -- mirrors prim.training.train_meta_rca.ContextQueryDataGenerator
# =============================================================================


class TabICLRCADataGenerator:
    """Context+query episode generator, drop-in for ``ContextQueryDataGenerator``.

    Same ``generate_episode()`` return signature (8-tuple), so it can be wired
    into ``ContextQueryDataset`` by swapping the import and the ``data_generator``
    line.  One DAG + SCM + feature-realism transform is drawn per episode; every
    context / query sample is a fresh (obs, int) pair from that same SCM with a
    fresh intervention node, normalised with its OWN observational window.
    """

    def __init__(
        self,
        n_nodes_range=(10, 20),
        max_nodes=50,
        obs_range=(50, 300),
        int_range=(5, 50),
        k_context_range=(0, 0),
        n_queries=16,
        equation_type="mixed",  # accepted for signature compatibility
        nn_config_path=None,  # unused
        expected_degree_range=(1.5, 3.0),
        config: Optional[TabICLPriorConfig] = None,
    ):
        self.n_nodes_range = n_nodes_range
        self.max_nodes = max_nodes
        self.obs_range = obs_range
        self.int_range = int_range
        self.k_context_range = k_context_range
        self.n_queries = n_queries
        self.cfg = config or TabICLPriorConfig(
            n_nodes_range=n_nodes_range,
            max_nodes=max_nodes,
            expected_degree_range=expected_degree_range,
        )

    def generate_episode(self):
        rng = np.random.default_rng(random.randrange(2**31))
        hp = self.cfg.sample(rng)

        k = int(rng.integers(self.n_nodes_range[0], self.n_nodes_range[1] + 1))
        n_obs = int(rng.integers(self.obs_range[0], self.obs_range[1]))
        n_int = int(rng.integers(self.int_range[0], self.int_range[1]))
        n_int = _cascade_window(hp, n_int, self.cfg,
                                rng)  # short window for cascade
        k_context = int(
            rng.integers(self.k_context_range[0], self.k_context_range[1] + 1))
        K = self.max_nodes

        # sparse full-scale-causRCA embedding: fixed for the whole episode
        layout, k_active = _maybe_sparse_layout(self.cfg, k, rng)
        n_model = layout.k_total if layout is not None else k_active

        dag, valid_nodes, aggregators = _sample_dag(hp, k_active, rng)
        scm = TabICLSCM(dag, hp, rng)

        # fit feature-realism transforms ONCE per episode on a reference obs draw
        ref_obs = scm.sample(max(n_obs, 128), rng)
        transforms = fit_column_transforms(ref_obs, hp, rng)

        def one_sample():
            obs_raw = scm.sample(n_obs, rng)
            rcs = _pick_independent_rcs(dag, valid_nodes,
                                        _sample_n_faults(self.cfg, rng), hp,
                                        rng)
            om, os_ = obs_raw.mean(axis=0), obs_raw.std(axis=0) + 1e-8
            int_scm = _apply_multi_intervention(scm, rcs,
                                                hp["intervention_kind"], om,
                                                os_, hp, rng)
            int_raw = int_scm.sample(n_int, rng)
            obs_w, disc_w = apply_column_transforms(obs_raw, transforms)
            int_w, _ = apply_column_transforms(int_raw, transforms)
            if int_scm.cascade_spec:
                int_w = apply_cascade_post_warp(int_w, int_scm.cascade_spec,
                                                obs_w)
            symptom = _multi_symptom(dag, rcs, aggregators, hp, rng)
            if layout is not None:
                obs_w, int_w, _d, _rc0, symptom, _g = layout.embed(
                    obs_w, int_w, disc_w, rcs[0], symptom, dag, rng)
                rcs = [int(layout.active_pos[r]) for r in rcs]
            obs_p, int_p = normalize_observational_only(
                obs_w, int_w, n_model, K)
            tmask = np.zeros(K, dtype=np.float32)
            tmask[symptom] = 1.0
            label = np.zeros(K, dtype=np.float32)  # MULTI-HOT
            for r in rcs:
                label[r] = 1.0
            return obs_p, int_p, label, tmask

        ctx_obs, ctx_int, ctx_mask = [], [], []
        for _ in range(k_context):
            o, i, _lbl, m = one_sample()
            ctx_obs.append(o)
            ctx_int.append(i)
            ctx_mask.append(m)

        q_obs, q_int, q_lbl, q_mask = [], [], [], []
        for _ in range(self.n_queries):
            o, i, lbl, m = one_sample()
            q_obs.append(o)
            q_int.append(i)
            q_lbl.append(lbl)
            q_mask.append(m)

        query_obs = np.stack(q_obs)
        query_int = np.stack(q_int)
        query_labels = np.stack(q_lbl).astype(
            np.float32)  # (n_queries, K) multi-hot
        query_masks = np.stack(q_mask)

        if k_context == 0:
            ctx_obs_a = np.empty((0, n_obs, K), dtype=np.float32)
            ctx_int_a = np.empty((0, n_int, K), dtype=np.float32)
            ctx_mask_a = np.empty((0, K), dtype=np.float32)
        else:
            ctx_obs_a = np.stack(ctx_obs)
            ctx_int_a = np.stack(ctx_int)
            ctx_mask_a = np.stack(ctx_mask)

        return (
            ctx_obs_a,
            ctx_int_a,
            query_obs,
            query_int,
            query_labels,
            query_masks,
            ctx_mask_a,
            n_model,
        )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    import time

    cfg = TabICLPriorConfig(n_nodes_range=(17, 90), max_nodes=100)

    print("=" * 72)
    print("TabICL-RCA prior smoke test")
    print("=" * 72)

    t0 = time.time()
    eq_counts, kind_counts, nfault_counts = {}, {}, {}
    n_discrete_cols = 0
    total_cols = 0
    shift_ratio = []
    n_sparse = 0
    zero_col_frac = []
    sym_is_rc = 0

    for i in range(60):
        ep = generate_rca_episode(config=cfg, seed=1000 + i)
        eq_counts[ep["equation_type"]] = eq_counts.get(ep["equation_type"],
                                                       0) + 1
        kind_counts[ep["intervention_kind"]] = (
            kind_counts.get(ep["intervention_kind"], 0) + 1)
        nfault_counts[ep["n_faults"]] = nfault_counts.get(ep["n_faults"],
                                                          0) + 1
        sym_is_rc += int(ep["symptom_node"] in ep["intervention_nodes"])
        k = ep["n_nodes"]
        n_discrete_cols += ep["discrete_mask"][:k].sum()
        total_cols += k

        obs = ep["observational_data"][:, :k]
        intr = ep["interventional_data"][:, :k]
        rc = ep["intervention_node"]
        # fraction of ~silent columns (proxy for sparse full-scale episodes)
        silent = np.mean((np.abs(obs).max(axis=0) < 1e-3)
                         & (np.abs(intr).max(axis=0) < 1e-3))
        zero_col_frac.append(silent)
        if silent > 0.2:
            n_sparse += 1
        # after obs-only normalisation, the observational window is ~centred at 0
        # while the intervened column should carry a clear mean shift
        shift_ratio.append(
            abs(intr[:, rc].mean()) / (abs(obs[:, rc].mean()) + 1e-6))

        if i < 3:
            print(
                f"\nepisode {i}: k={k} eq={ep['equation_type']:6s} "
                f"int={ep['intervention_kind']:6s} rc={rc} symptom={ep['symptom_node']}"
            )
            print(f"  obs shape {ep['observational_data'].shape}  "
                  f"int shape {ep['interventional_data'].shape}")
            print(f"  discrete cols: {int(ep['discrete_mask'][:k].sum())}/{k}")
            print(
                f"  obs mean(rc)={obs[:, rc].mean():+.3f}  "
                f"int mean(rc)={intr[:, rc].mean():+.3f}  "
                f"obs std(rc)={obs[:, rc].std():.3f}  int std(rc)={intr[:, rc].std():.3f}"
            )

    print("\n" + "-" * 72)
    print(f"generated 60 episodes in {time.time() - t0:.1f}s")
    print(f"equation types  : {eq_counts}")
    print(f"intervention    : {kind_counts}")
    print(f"n_faults / episode: {dict(sorted(nfault_counts.items()))}   "
          f"(symptom == a root cause: {sym_is_rc}/60, must be 0)")
    print(f"discrete columns: {n_discrete_cols}/{total_cols} "
          f"({100 * n_discrete_cols / total_cols:.1f}%)")
    print(f"sparse full-scale episodes: {n_sparse}/60  "
          f"(mean silent-column fraction {np.mean(zero_col_frac):.2f})")
    print(
        f"median |int_mean(rc)| / |obs_mean(rc)|  (obs-only norm): "
        f"{np.median(shift_ratio):.1f}   (>> 1  ==  mean shift is preserved)")

    print("\n--- training adapter ---")
    gen = TabICLRCADataGenerator(n_nodes_range=(17, 90),
                                 max_nodes=100,
                                 n_queries=4)
    out = gen.generate_episode()
    names = [
        "ctx_obs",
        "ctx_int",
        "q_obs",
        "q_int",
        "q_labels",
        "q_masks",
        "ctx_masks",
        "k",
    ]
    for nm, v in zip(names, out):
        print(f"  {nm:10s}: {getattr(v, 'shape', v)}")
    print("\nOK")
