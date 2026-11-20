"""
Silent-node filtering for full-mode causRCA evaluation.

Full-mode causRCA feeds all ~92 expert-graph nodes to the model, but for any
one run most of them belong to uninvolved subsystems and are forward-filled
CONSTANTS -- after obs-only z-normalisation they are ~0 columns in both the
normal and abnormal windows.  The tabicl / scm priors never generate
mostly-zero-column graphs, so full-mode is out-of-distribution on two axes at
once: K/n approx 1 (vs 2-6 in training) and dozens of identical zero-distractors.

A metric that is constant before the fault AND does not shift during it carries
zero root-cause information -- it is provably not causally involved.  Dropping
those columns is legitimate preprocessing (observable from the data, not
privileged) and turns full-mode into an in-distribution, sub-mode-sized problem.

``active_columns`` returns the keep-mask; the caller reduces its node list and
data arrays before the usual normalise / pad / scale.
"""

from __future__ import annotations

import numpy as np


def active_columns(
    normal_arr: np.ndarray,
    abnormal_arr: np.ndarray,
    force_keep=(),
    *,
    std_eps: float = 1e-4,
    shift_tau: float = 0.5,
    min_keep: int = 8,
) -> np.ndarray:
    """Boolean keep-mask over columns of ``normal_arr`` / ``abnormal_arr``.

    A column is KEPT iff any of:
      * it varies in the normal window   (std > std_eps),
      * it varies in the abnormal window (std > std_eps),
      * its mean shifts between windows   (|d_mean| > shift_tau * (std_normal + eps))
        -- catches stuck-sensor faults on an otherwise-flat metric.

    ``force_keep`` : column indices always kept (e.g. the alarm nodes).
    If fewer than ``min_keep`` survive, fall back to the ``min_keep`` most
    active columns by (std_normal + std_abnormal + |d_mean|).
    """
    normal_arr = np.asarray(normal_arr, dtype=np.float64)
    abnormal_arr = np.asarray(abnormal_arr, dtype=np.float64)

    std_n = normal_arr.std(axis=0)
    std_a = abnormal_arr.std(axis=0)
    d_mean = np.abs(abnormal_arr.mean(axis=0) - normal_arr.mean(axis=0))

    keep = ((std_n > std_eps)
            | (std_a > std_eps)
            | (d_mean > shift_tau * (std_n + std_eps)))

    force_keep = [int(j) for j in force_keep if 0 <= int(j) < keep.size]
    keep[force_keep] = True

    if keep.sum() < min_keep:
        score = std_n + std_a + d_mean
        top = np.argsort(-score)[:min_keep]
        keep = np.zeros_like(keep)
        keep[top] = True
        keep[force_keep] = True

    return keep


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_active = 12
    normal = np.zeros((200, 92))
    abnormal = np.zeros((80, 92))
    # active cluster: real variance
    normal[:, :n_active] = rng.standard_normal((200, n_active))
    abnormal[:, :n_active] = rng.standard_normal((80, n_active))
    # one stuck-sensor RC: flat in both windows but shifted
    normal[:, 20] = 5.0
    abnormal[:, 20] = 9.0
    # everything else: forward-filled constant
    for j in range(92):
        if j >= n_active and j != 20:
            normal[:, j] = abnormal[:, j] = rng.uniform(-2, 2)

    keep = active_columns(normal, abnormal, force_keep=[50])
    print(f"kept {keep.sum()}/92  (want ~{n_active + 2})")
    print(f"  stuck-sensor col 20 kept: {bool(keep[20])}  (must be True)")
    print(f"  forced col 50 kept:       {bool(keep[50])}  (must be True)")
    print(f"  idle col 80 kept:         {bool(keep[80])}  (should be False)")
