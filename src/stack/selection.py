"""Out-of-sample validation of the ex-ante bidding-rule choice.

The bidding study (``bidding.py``) tested ten ex-ante rules on one year and reported the
best one, so its +2.4 k EUR/MW (2 h) is an in-sample pick. This module asks two questions
that do not need any new market data:

1. **Data snooping.** Is the best rule still better than ex-ante DA + FCR once we account
   for having searched over all of them? Hansen's (2005) Superior Predictive Ability test
   (consistent ``SPA_c`` and White's 2000 Reality Check, stationary bootstrap of
   Politis & Romano 1994).
2. **Walk-forward selection.** If a trader had picked the rule each day by its trailing
   realised revenue (days strictly before the decision), what would they have earned?

Why switching rules day by day is valid on the published daily table: every rule's day-D
decision depends only on market data up to D-1 (its acceptance calibration is computed on
its own *hypothetical* bids against published marginal prices, not on executed trades), and
the battery returns to 50 % SoC every day. So the day-D revenue of a rule does not depend on
which rule was used on earlier days. A rule's revenue for delivery day D-1 is known before
the D-1 08:00 FCR auction for day D (all its prices were published on D-2), so ``lag=1`` is
non-anticipative; ``lag=2`` is reported as a conservative check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wide_daily(daily: pd.DataFrame, duration_h: float, value: str = "total") -> pd.DataFrame:
    """Day x rule table of daily revenue for one duration (rows sorted by date)."""
    d = daily[daily.duration_h == duration_h]
    w = d.pivot(index="date", columns="rule", values=value).sort_index()
    if w.isna().any().any():
        raise ValueError("missing day x rule values")
    return w


def walk_forward_select(wide: pd.DataFrame, window: int | None, default: str, lag: int = 1,
                        min_history: int | None = None) -> pd.DataFrame:
    """Pick, for each day, the rule with the highest trailing revenue.

    The score for day i uses rows ``[i - lag - window + 1, i - lag]`` (``window=None``:
    expanding, all rows up to ``i - lag``). Until ``min_history`` days are available
    (default: ``window`` or 28 for expanding) the ``default`` rule is used. Ties keep the
    default if it is tied, otherwise the first column. Returns date, chosen rule, revenue.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1 (day D itself is never known)")
    if default not in wide.columns:
        raise ValueError(f"default rule {default!r} not in table")
    x = wide.to_numpy(float)
    cols = list(wide.columns)
    j_def = cols.index(default)
    need = min_history if min_history is not None else (window or 28)
    csum = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, axis=0)])
    picks, vals = [], []
    for i in range(len(x)):
        hi = i - lag + 1  # exclusive end row in x
        lo = 0 if window is None else max(0, hi - window)
        if hi - lo < need or hi <= 0:
            j = j_def
        else:
            score = csum[hi] - csum[lo]
            best = score.max()
            j = j_def if score[j_def] >= best - 1e-9 else int(np.argmax(score))
        picks.append(cols[j])
        vals.append(x[i, j])
    return pd.DataFrame({"date": wide.index, "rule": picks, "total": vals})


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (n_boot x n), circular wrap."""
    p = 1.0 / float(mean_block)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new = rng.random((n_boot, n)) < p
    fresh = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def spa_test(diffs: np.ndarray, mean_block: float = 7.0, n_boot: int = 2000, seed: int = 0) -> dict:
    """Hansen (2005) SPA test of H0: no model beats the benchmark (all E[d_k] <= 0).

    ``diffs``: n_days x K array of (model k − benchmark) daily revenue. Returns the studentised
    statistic and p-values for the consistent (``spa_c``), lower (``spa_l``) and upper
    (``spa_u`` = White's Reality Check, studentised) versions.
    """
    d = np.asarray(diffs, float)
    if d.ndim == 1:
        d = d[:, None]
    n, k = d.shape
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    dbar = d.mean(axis=0)
    boot_means = np.stack([d[idx[b]].mean(axis=0) for b in range(n_boot)])  # n_boot x K
    omega = np.sqrt(n) * boot_means.std(axis=0, ddof=0)
    omega = np.where(omega > 0, omega, np.inf)
    t_obs = max(0.0, float(np.max(np.sqrt(n) * dbar / omega)))
    # Hansen (2005): Z* = dbar* - g(dbar) = (dbar* - dbar) + (dbar - g(dbar)), with
    # g_u(x) = x (Reality Check), g_l(x) = max(x, 0), g_c(x) = x * 1{sqrt(n) x / omega >= -sqrt(2 log log n)}.
    thresh = -np.sqrt(2.0 * np.log(np.log(n)))
    shift_u = np.zeros(k)
    shift_l = np.minimum(dbar, 0.0)
    shift_c = np.where(np.sqrt(n) * dbar / omega >= thresh, 0.0, dbar)
    out = {"n_days": n, "n_models": k, "t_spa": t_obs}
    for name, shift in (("spa_l", shift_l), ("spa_c", shift_c), ("spa_u", shift_u)):
        z = boot_means - dbar + shift  # recentred bootstrap means
        t_b = np.maximum(0.0, np.max(np.sqrt(n) * z / omega, axis=1))
        out[f"p_{name}"] = float((t_b >= t_obs).mean())
    return out


def single_pair_p(diff: np.ndarray, mean_block: float = 7.0, n_boot: int = 2000, seed: int = 0) -> float:
    """Naive one-sided p-value for one pre-chosen rule (no search correction)."""
    return spa_test(np.asarray(diff, float)[:, None], mean_block, n_boot, seed)["p_spa_c"]
