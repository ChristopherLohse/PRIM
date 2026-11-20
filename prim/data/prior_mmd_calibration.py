"""
MMD-based calibration of the intervention strength in ``tabicl_rca_prior``.

Goal
----
Pick a sensible range for how hard the prior perturbs a node -- i.e. the
``weight_change_range`` (soft weight-change), ``shift_sigma_range`` (mean shift)
and ``stuck_sigma_range`` (hard "stuck sensor") in :class:`TabICLPriorConfig`.

Method
------
For a grid of intervention magnitudes we draw many episodes, each with a FIXED
intervention kind and a FIXED magnitude, and for every episode run a two-sample
kernel MMD permutation test between the observational window and the
interventional window (in the same obs-only-normalised representation the model
consumes).  We report, per magnitude:

* ``reject_rate``  -- fraction of episodes where the MMD test rejects
  H0: P_obs == P_int at alpha = 0.05.  This is "detectability": how often the
  fault is statistically visible at all.
* ``mmd_full``     -- mean MMD^2 statistic over the full multivariate window.
* ``mmd_rc``       -- mean MMD^2 on the root-cause column only.
* ``shift_rc / shift_desc / shift_other`` -- mean |E[int] - E[obs]| (in z-units)
  at the root cause / its descendants / the untouched nodes.  Shows the shift is
  localised to the causal cone.

A good threshold is the smallest magnitude with ``reject_rate`` close to 1 while
the shift on untouched nodes stays near 0.

Run ``python -m prim.data.prior_mmd_calibration`` for a quick console sweep, or
use :func:`run_sweep` from the companion notebook
``notebooks/prior_mmd_calibration.ipynb``.
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np

from prim.data.tabicl_rca_prior import TabICLPriorConfig, generate_rca_episode

# =============================================================================
# MMD  (RBF kernel, median heuristic, permutation test)
# =============================================================================


def _rbf_gram(Z: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.einsum("ij,ij->i", Z, Z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _median_gamma(Z: np.ndarray,
                  rng: np.random.Generator,
                  cap: int = 400) -> float:
    if len(Z) > cap:
        Z = Z[rng.choice(len(Z), cap, replace=False)]
    sq = np.einsum("ij,ij->i", Z, Z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    d2 = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(np.sqrt(np.maximum(d2, 0.0)))
    med = med if med > 1e-8 else 1.0
    return 1.0 / (2.0 * med * med)


def _mmd2_unbiased(K: np.ndarray, n: int) -> float:
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]
    m = K.shape[0] - n
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sxy = Kxy.mean()
    return float(sxx + syy - 2.0 * sxy)


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 150,
    max_x: int = 160,
    max_y: int = 120,
    seed: int = 0,
) -> tuple[float, float]:
    """Return ``(mmd2_unbiased, p_value)`` for H0: X and Y are from one dist."""
    rng = np.random.default_rng(seed)
    if len(X) > max_x:
        X = X[rng.choice(len(X), max_x, replace=False)]
    if len(Y) > max_y:
        Y = Y[rng.choice(len(Y), max_y, replace=False)]

    Z = np.vstack([X, Y]).astype(np.float64)
    n = len(X)
    gamma = _median_gamma(Z, rng)
    K = _rbf_gram(Z, gamma)

    obs_stat = _mmd2_unbiased(K, n)

    count = 0
    N = K.shape[0]
    for _ in range(n_perm):
        p = rng.permutation(N)
        Kp = K[np.ix_(p, p)]
        if _mmd2_unbiased(Kp, n) >= obs_stat:
            count += 1
    p_value = (1.0 + count) / (1.0 + n_perm)
    return obs_stat, p_value


# =============================================================================
# Episode helpers
# =============================================================================


def episode_windows(cfg: TabICLPriorConfig, seed: int):
    """One episode -> (obs, int, root_cause, descendants, k).

    ``obs`` / ``int`` are in the obs-only-normalised z representation the model
    sees, with the ``K/k`` padding-scale divided back out and padding removed.
    """
    ep = generate_rca_episode(config=cfg, seed=seed)
    k = ep["n_nodes"]
    scale = cfg.max_nodes / k
    obs = (ep["observational_data"][:, :k] / scale).astype(np.float64)
    intr = (ep["interventional_data"][:, :k] / scale).astype(np.float64)
    g = nx.DiGraph()
    g.add_nodes_from(range(k))
    g.add_edges_from(ep["dag_edges"])
    rc = ep["intervention_node"]
    desc = sorted(nx.descendants(g, rc))
    return obs, intr, rc, desc, k


def _role_shifts(obs, intr, rc, desc, k):
    shift = np.abs(intr.mean(axis=0) - obs.mean(axis=0))  # obs.mean ~ 0
    other = [j for j in range(k) if j != rc and j not in set(desc)]
    return (
        float(shift[rc]),
        float(np.mean(shift[desc])) if desc else 0.0,
        float(np.mean(shift[other])) if other else 0.0,
    )


# =============================================================================
# Sweep
# =============================================================================

# intervention_mix order: weight / shift / do / noise / cascade
_KIND_TO_RANGE = {
    "weight": "weight_change_range",
    "shift": "shift_sigma_range",
    "do": "stuck_sigma_range",
    "cascade": "cascade_sigma_range",
}
_KIND_TO_MIX = {
    "weight": (1.0, 0.0, 0.0, 0.0, 0.0),
    "shift": (0.0, 1.0, 0.0, 0.0, 0.0),
    "do": (0.0, 0.0, 1.0, 0.0, 0.0),
    "cascade": (0.0, 0.0, 0.0, 0.0, 1.0),
}


def _cfg_for(base: TabICLPriorConfig, kind: str, magnitude: float, flip: bool,
             fast_equations: bool, iid_only: bool) -> TabICLPriorConfig:
    kw = {
        "intervention_mix": _KIND_TO_MIX[kind],
        _KIND_TO_RANGE[kind]: (magnitude, magnitude),
    }
    if kind == "weight":
        kw["weight_flip_prob"] = 1.0 if flip else 0.0
    if fast_equations:
        # drop the (slow) tree-SCM share for a quick sweep
        kw["equation_mix"] = (0.72, 0.05, 0.15, 0.08)
    if iid_only:
        # The MMD permutation test assumes exchangeable samples.  AR(1) noise and
        # the near-constant availability mode both break that and inflate the
        # rejection rate even under H0 (no intervention).  Turn them off so the
        # sweep measures the *pure* intervention effect and the test is
        # well-calibrated (reject_rate -> alpha at magnitude 1.0 / sigma 0).
        kw["temporal_autocorr_prob"] = 0.0
        kw["availability_prob"] = 0.0
    return replace(base, **kw)


def run_sweep(
    kind: str = "weight",
    magnitudes=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
    n_episodes: int = 110,
    flip: bool = False,
    fast_equations: bool = True,
    iid_only: bool = True,
    base_config: TabICLPriorConfig | None = None,
    n_perm: int = 150,
    alpha: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """Sweep one intervention magnitude; return a list of per-magnitude rows.

    Total episodes generated = ``len(magnitudes) * n_episodes``
    (defaults ~= 1000).
    """
    assert kind in _KIND_TO_RANGE, f"kind must be one of {list(_KIND_TO_RANGE)}"
    base = base_config or TabICLPriorConfig()
    rows = []
    ep_seed = seed * 1_000_003

    for mag in magnitudes:
        cfg = _cfg_for(base, kind, mag, flip, fast_equations, iid_only)
        mmd_full, mmd_rc, rejects = [], [], []
        s_rc, s_desc, s_other = [], [], []

        for _ in range(n_episodes):
            ep_seed += 1
            try:
                obs, intr, rc, desc, k = episode_windows(cfg, ep_seed)
            except Exception:
                continue
            m2, p = mmd_permutation_test(obs,
                                         intr,
                                         n_perm=n_perm,
                                         seed=ep_seed)
            mmd_full.append(m2)
            rejects.append(p < alpha)
            m2_rc, p_rc = mmd_permutation_test(obs[:, [rc]],
                                               intr[:, [rc]],
                                               n_perm=n_perm,
                                               seed=ep_seed + 1)
            mmd_rc.append(m2_rc)
            a, b, c = _role_shifts(obs, intr, rc, desc, k)
            s_rc.append(a)
            s_desc.append(b)
            s_other.append(c)

        row = dict(
            kind=kind,
            flip=flip,
            magnitude=float(mag),
            n=len(mmd_full),
            reject_rate=float(np.mean(rejects)) if rejects else float("nan"),
            mmd_full=float(np.mean(mmd_full)) if mmd_full else float("nan"),
            mmd_full_p10=float(np.percentile(mmd_full, 10))
            if mmd_full else float("nan"),
            mmd_rc=float(np.mean(mmd_rc)) if mmd_rc else float("nan"),
            shift_rc=float(np.mean(s_rc)) if s_rc else float("nan"),
            shift_desc=float(np.mean(s_desc)) if s_desc else float("nan"),
            shift_other=float(np.mean(s_other)) if s_other else float("nan"),
        )
        rows.append(row)
        if verbose:
            print(
                f"{kind:6s} flip={int(flip)} mag={mag:5.2f}  "
                f"n={row['n']:4d}  reject={row['reject_rate']:.2f}  "
                f"mmd_full={row['mmd_full']:.4f}  mmd_rc={row['mmd_rc']:.4f}  "
                f"shift rc/desc/other={row['shift_rc']:.2f}/"
                f"{row['shift_desc']:.2f}/{row['shift_other']:.2f}",
                flush=True,
            )
    return rows


def recommend_threshold(rows: list[dict],
                        target_reject: float = 0.9) -> float | None:
    """Smallest swept magnitude whose ``reject_rate`` >= ``target_reject``."""
    ok = [r["magnitude"] for r in rows if r["reject_rate"] >= target_reject]
    return min(ok) if ok else None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="weight", choices=list(_KIND_TO_RANGE))
    ap.add_argument("--n_episodes", type=int, default=110)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--full_equations",
                    action="store_true",
                    help="keep the slow tree-SCM share in the mixture")
    ap.add_argument(
        "--with_autocorr",
        action="store_true",
        help="keep AR(1) noise + availability mode (inflates reject_rate)")
    ap.add_argument("--magnitudes", type=float, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kw = dict(
        kind=args.kind,
        n_episodes=args.n_episodes,
        flip=args.flip,
        fast_equations=not args.full_equations,
        iid_only=not args.with_autocorr,
        seed=args.seed,
    )
    if args.magnitudes:
        kw["magnitudes"] = tuple(args.magnitudes)

    print(
        f"\nsweeping intervention kind = {args.kind!r}  "
        f"(total episodes ~ {len(kw.get('magnitudes', range(9))) * args.n_episodes})\n"
    )
    rows = run_sweep(**kw)

    thr = recommend_threshold(rows, 0.9)
    thr95 = recommend_threshold(rows, 0.95)
    print("\n" + "=" * 60)
    print(f"smallest magnitude with reject_rate >= 0.90 : {thr}")
    print(f"smallest magnitude with reject_rate >= 0.95 : {thr95}")
    print("=" * 60)
