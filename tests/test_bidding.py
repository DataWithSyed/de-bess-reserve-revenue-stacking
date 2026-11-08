import numpy as np
import pandas as pd
import pytest

from stack.backtest import backtest
from stack.bidding import (Rule, add_calibration, bid_table, quantile_bid, reliability,
                           run_rule, selection_bias)
from stack.model import Battery


def synthetic(n_days=45, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    da = {}
    for d in days:
        idx = pd.date_range(d, periods=96, freq="15min", tz="Europe/Berlin")
        base = 80 + 40 * np.sin(np.arange(96) / 96 * 2 * np.pi - 1.5)
        da[d.strftime("%Y-%m-%d")] = pd.Series(base + rng.normal(0, 15, 96), index=idx)
    rows = []
    for d in days:
        for b in range(6):
            rows.append(dict(date=d.strftime("%Y-%m-%d"), block=b,
                             fcr_eur_mw_h=max(0.0, rng.normal(15, 4)),
                             afrr_neg_avg=4.0, afrr_pos_avg=6.0,
                             afrr_neg_marg=max(0.0, rng.normal(6, 3)),
                             afrr_pos_marg=max(0.0, rng.normal(9, 4))))
    return da, pd.DataFrame(rows)


def test_quantile_bid_hits_target_on_history():
    h = np.arange(1, 101, dtype=float)
    b, p = quantile_bid(h, 0.9)
    assert p == pytest.approx(0.9, abs=0.011)
    assert (h >= b).mean() == pytest.approx(p)


def test_calibration_is_non_anticipative():
    da, res = synthetic()
    days = sorted(da)[30:]
    bt = add_calibration(bid_table(res, days, Rule("x")), 7)
    # change the outcome of the last day: factors of all earlier days must not move
    bt2 = bt.drop(columns=["calib", "p_used"]).copy()
    bt2.loc[bt2.date == days[-1], "accepted"] = 1.0 - bt2.loc[bt2.date == days[-1], "accepted"]
    bt2 = add_calibration(bt2, 7)
    early = bt.date < days[-1]
    np.testing.assert_allclose(bt.loc[early, "calib"], bt2.loc[early, "calib"])
    assert (bt.calib <= 1.0).all()


def test_reliability_and_selection_bias_tables():
    bt = pd.DataFrame(dict(p_pred=[0.1, 0.15, 0.85, 0.95], accepted=[0, 0, 1, 0]))
    r = reliability(bt)
    assert r.n.sum() == 4
    det = pd.DataFrame(dict(fcr_mw=[1, 0], afrr_pos_mw=[0, 1], afrr_neg_mw=[0, 0],
                            fcr_fc=[10, 5], fcr_real=[8, 9]))
    sb = selection_bias(det).set_index("chose_afrr")
    assert sb.loc[True, "mean_fc_error"] == -4 and sb.loc[False, "mean_fc_error"] == 2


def test_base_rules_reproduce_backtest():
    da, res = synthetic()
    days = sorted(da)[30:40]
    bat = Battery(duration_h=2.0)
    ref = backtest(da, res, bat, days, strategies=("xa_da_fcr", "xa_stack")) \
        .groupby("strategy").total.sum()
    a, _ = run_rule(da, res, bat, days, Rule("da_fcr", afrr="none"))
    b, _ = run_rule(da, res, bat, days, Rule("stack"))
    assert a.total.sum() == pytest.approx(ref["xa_da_fcr"], rel=1e-9)
    assert b.total.sum() == pytest.approx(ref["xa_stack"], rel=1e-9)


def test_hurdle_reduces_afrr_offers():
    da, res = synthetic()
    days = sorted(da)[30:40]
    bat = Battery(duration_h=2.0)
    lo, _ = run_rule(da, res, bat, days, Rule("h1"))
    hi, _ = run_rule(da, res, bat, days, Rule("h3", hurdle=3.0))
    assert hi.afrr_offered_mwh.sum() <= lo.afrr_offered_mwh.sum() + 1e-9
