"""
Minimal standalone wrapper exposing RCAEval's BARO implementation
(RCAEval.e2e.baro) as a PetShop-contract baseline, for use as the "BARO"
curve in test_prior_eval.py's Figure 3 sweep.

This intentionally does NOT depend on the RE1 benchmark harness
(prim/evaluation/test_rcaeval_re1.py) or on test_extra_baselines.py --
both are RE1/PetShop/CausRCA-specific evaluation scripts and pull in a lot
of code irrelevant to just wrapping one baseline function. It keeps only
the two things Figure 3 actually needs from that chain:
  1. The py3.13 compatibility patch for `RCAEval.utility.is_py312`
     (RCAEval/e2e/__init__.py's version gate only recognises
     3.10/3.12/3.14 and predates 3.13; the e2e methods used here are
     plain numpy/sklearn/torch and don't depend on 3.12-specific
     behaviour, so the gate is widened rather than requiring a separate
     interpreter).
  2. The PetShop-contract adapter that flattes a (node, "value", "raw")
     MultiIndex frame into the plain time-indexed DataFrame RCAEval's
     `analyze_root_causes(data, inject_time, ...)` contract expects, and
     wraps the ranked column names back into PotentialRootCause objects.

Requires RCAEval-main/ to be present (see pyproject.toml's
[tool.uv.sources] and the README's External Datasets section) and the
`baselines` extras installed (`bash scripts/install_baselines.sh`).
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PETSHOP_CODE = _PROJECT_ROOT / "petshop-root-cause-analysis-main" / "code"
if str(_PETSHOP_CODE) not in sys.path:
    sys.path.insert(0, str(_PETSHOP_CODE))

from rca_task import PotentialRootCause  # noqa: E402

import RCAEval.utility as _rcaeval_utility  # noqa: E402

if sys.version_info[:2] == (3, 13):
    _rcaeval_utility.is_py312 = lambda: True

try:
    import RCAEval.e2e as e2e
    _E2E_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on optional extras
    e2e = None
    _E2E_IMPORT_ERROR = e


def _rcaeval_as_petshop_method(rcaeval_func):
    """Wrap an RCAEval.e2e function (data, inject_time, ...) -> {"ranks": [...]}
    as a PetShop-contract analyze_root_causes(graph, target_node, target_metric,
    target_statistic, normal_metrics, abnormal_metrics) -> List[PotentialRootCause].
    `graph` is ignored; these RCAEval methods discover structure themselves."""

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
        if len(node_names) < 2:
            return []

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

        run_args = argparse.Namespace(root_path=str(_PROJECT_ROOT),
                                      data_path="")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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


def make_baro_method():
    """Returns a PetShop-contract analyze_root_causes callable for BARO, or
    None if RCAEval.e2e failed to import (missing optional dependency)."""
    if e2e is None:
        return None
    return _rcaeval_as_petshop_method(e2e.baro)
