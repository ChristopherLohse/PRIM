"""
Two lightweight, graph-free RCA baselines requested by reviewers:

1. percentile      — ranks nodes by the max amount any anomalous sample exceeds
                      the 95th percentile (or falls below the 5th percentile) of
                      the normal distribution for that node.
2. gaussian_kl      — fits N(mu, sigma^2) separately to the normal and abnormal
                      samples of each node and ranks nodes by the (symmetric) KL
                      divergence between the two Gaussians.

Both match the `analyze_root_causes(graph, target_node, target_metric,
target_statistic, normal_metrics, abnormal_metrics) -> List[PotentialRootCause]`
signature used by the petshop baselines so they plug directly into the existing
baseline harnesses in test_threenode_synthetic.py and test_multi_rca.py. They are
graph-free (needs_graph=False) and depend only on numpy/pandas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class PotentialRootCause:
    node: str
    metric: str
    score: float


def _node_columns(df: pd.DataFrame, target_metric: str,
                  target_statistic: str) -> pd.DataFrame:
    """Flatten a (node, metric, statistic) MultiIndex DataFrame to node-named columns."""
    sub = df.loc[:, (slice(None), [target_metric], [target_statistic])].copy()
    sub.columns = [c[0] for c in sub.columns]
    return sub


def make_percentile_baseline(lower_q: float = 0.05, upper_q: float = 0.95):
    """Baseline [3]: rank nodes by the max amount an anomalous sample exceeds the
    upper_q percentile (or falls below the lower_q percentile) of the normal data.
    """

    def analyze_root_causes(
        graph,
        target_node,
        target_metric,
        target_statistic,
        normal_metrics,
        abnormal_metrics,
    ) -> List[PotentialRootCause]:
        normal = _node_columns(normal_metrics, target_metric, target_statistic)
        abnormal = _node_columns(abnormal_metrics, target_metric,
                                 target_statistic)
        common = normal.columns.intersection(abnormal.columns)

        results = []
        for node in common:
            n_vals = normal[node].dropna().to_numpy()
            a_vals = abnormal[node].dropna().to_numpy()
            if n_vals.size == 0 or a_vals.size == 0:
                continue
            p_hi = np.percentile(n_vals, 100 * upper_q)
            p_lo = np.percentile(n_vals, 100 * lower_q)
            score = max((a_vals - p_hi).max(), (p_lo - a_vals).max())
            results.append(
                PotentialRootCause(node=node,
                                   metric=target_metric,
                                   score=float(score)))
        return results

    return analyze_root_causes


def make_gaussian_divergence_baseline(symmetric: bool = True,
                                      eps: float = 1e-8):
    """Rank nodes by the KL divergence between N(mu, sigma^2) fit separately to the
    normal and abnormal samples of that node (Jeffreys/symmetric KL by default).
    """

    def _kl(mu_p, sigma_p, mu_q, sigma_q):
        return (np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) /
                (2 * sigma_q**2) - 0.5)

    def analyze_root_causes(
        graph,
        target_node,
        target_metric,
        target_statistic,
        normal_metrics,
        abnormal_metrics,
    ) -> List[PotentialRootCause]:
        normal = _node_columns(normal_metrics, target_metric, target_statistic)
        abnormal = _node_columns(abnormal_metrics, target_metric,
                                 target_statistic)
        common = normal.columns.intersection(abnormal.columns)

        results = []
        for node in common:
            n_vals = normal[node].dropna().to_numpy()
            a_vals = abnormal[node].dropna().to_numpy()
            if n_vals.size < 2 or a_vals.size < 2:
                continue
            mu_n, sigma_n = n_vals.mean(), n_vals.std() + eps
            mu_a, sigma_a = a_vals.mean(), a_vals.std() + eps
            kl_an = _kl(mu_a, sigma_a, mu_n, sigma_n)
            score = kl_an + _kl(mu_n, sigma_n, mu_a,
                                sigma_a) if symmetric else kl_an
            results.append(
                PotentialRootCause(node=node,
                                   metric=target_metric,
                                   score=float(score)))
        return results

    return analyze_root_causes
