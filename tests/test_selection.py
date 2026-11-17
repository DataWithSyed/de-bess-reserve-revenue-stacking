import numpy as np
import pandas as pd
import pytest

from stack.selection import spa_test, stationary_bootstrap_indices, walk_forward_select, wide_daily


def _wide(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-11-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"ref": rng.normal(100, 5, n), "a": rng.normal(100, 5, n), "b": rng.normal(100, 5, n)},
                        index=pd.Index(dates, name="date"))


def test_walk_forward_is_non_anticipative():
    w = _wide()
    base = walk_forward_select(w, 28, default="ref", lag=1)
    w2 = w.copy()
    w2.iloc[80:, :] = w2.iloc[80:, :] * np.array([0.0, 10.0, -5.0])  # change the future only
    alt = walk_forward_select(w2, 28, default="ref", lag=1)
    # picks for days <= 80 depend on rows <= 79 only
    assert (base.rule.iloc[:81] == alt.rule.iloc[:81]).all()
    assert (base.total.iloc[:80] == alt.total.iloc[:80]).all()
    with pytest.raises(ValueError):
        walk_forward_select(w, 28, default="ref", lag=0)


def test_walk_forward_follows_a_persistent_winner_and_uses_default_early():
    w = _wide()
    w["b"] += 50.0
    sel = walk_forward_select(w, 28, default="ref", lag=1)
    assert (sel.rule.iloc[:28] == "ref").all()           # not enough history yet
    assert (sel.rule.iloc[29:] == "b").all()
    exp = walk_forward_select(w, None, default="ref", lag=2)
    assert (exp.rule.iloc[30:] == "b").all()
    assert sel.total.iloc[40] == pytest.approx(w["b"].iloc[40])


def test_spa_rejects_a_clear_winner_and_not_noise():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 10, (365, 6))
    p_noise = spa_test(noise, n_boot=500, seed=1)
    assert p_noise["p_spa_c"] > 0.05
    assert p_noise["p_spa_l"] <= p_noise["p_spa_c"] + 1e-12 <= p_noise["p_spa_u"] + 2e-12
    win = noise.copy(); win[:, 2] += 4.0
    assert spa_test(win, n_boot=500, seed=1)["p_spa_c"] < 0.01
    # adding many bad rules must not make the best one look worse under SPA_c than under RC
    bad = np.hstack([noise[:, :1] + 0.8, noise[:, 1:] - 30.0])
    r = spa_test(bad, n_boot=500, seed=1)
    assert r["p_spa_c"] <= r["p_spa_u"]


def test_stationary_bootstrap_shape_and_wide_daily():
    idx = stationary_bootstrap_indices(50, 7.0, 20, np.random.default_rng(0))
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50
    daily = pd.DataFrame({"date": ["d1", "d2", "d1", "d2"], "rule": ["x", "x", "y", "y"],
                          "total": [1.0, 2.0, 3.0, 4.0], "duration_h": 2.0})
    w = wide_daily(daily, 2.0)
    assert list(w.columns) == ["x", "y"] and w.loc["d2", "y"] == 4.0
    with pytest.raises(ValueError):
        wide_daily(daily.iloc[:3], 2.0)
