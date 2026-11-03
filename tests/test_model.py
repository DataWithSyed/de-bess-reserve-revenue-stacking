import numpy as np
import pandas as pd
import pytest

from stack.backtest import day_arrays, naive_da_forecast, run_day
from stack.model import Battery, optimal_bid, settle, solve_day

BLOCK = np.repeat(np.arange(6), 16)
DT = 0.25


def two_price_day(lo=20.0, hi=120.0):
    p = np.full(96, lo)
    p[64:80] = hi        # 16:00-20:00 peak
    return p


def test_flat_price_no_cycling():
    r = solve_day(np.full(96, 50.0), BLOCK, Battery(), DT, markets=("da",))
    assert r.discharge.sum() * DT < 1e-6
    assert abs(r.objective) < 1e-6


def test_two_price_arbitrage_value():
    bat = Battery(duration_h=2.0, deg_cost_eur_mwh=0.0)
    r = solve_day(two_price_day(), BLOCK, bat, DT, markets=("da",))
    # usable 1.8 MWh -> discharge 1.8*eta, charge 1.8/eta; one cycle possible per peak? up to cycle cap
    e_use = 0.9 * bat.energy_mwh
    one_cycle = 120 * e_use * bat.eta - 20 * e_use / bat.eta
    assert r.objective >= one_cycle - 1e-6


def test_energy_balance_and_bounds():
    bat = Battery(duration_h=1.0)
    rng = np.random.default_rng(0)
    p = rng.normal(80, 40, 96)
    r = solve_day(p, BLOCK, bat, DT, fcr_price=np.full(6, 5.0), afrr_pos_price=np.full(6, 3.0),
                  afrr_neg_price=np.full(6, 3.0))
    s_prev = np.r_[bat.soc0 * bat.energy_mwh, r.soc[:-1]]
    np.testing.assert_allclose(r.soc - s_prev, bat.eta * r.charge * DT - r.discharge * DT / bat.eta, atol=1e-7)
    assert abs(r.soc[-1] - bat.soc0 * bat.energy_mwh) < 1e-7
    f, u, w = r.fcr[BLOCK], r.afrr_pos[BLOCK], r.afrr_neg[BLOCK]
    assert np.all(r.discharge + f + u <= bat.power_mw + 1e-7)
    assert np.all(r.charge + f + w <= bat.power_mw + 1e-7)
    E = bat.energy_mwh
    assert np.all(r.soc - bat.e_fcr_h * f - bat.e_afrr_h * u >= bat.soc_min * E - 1e-7)
    assert np.all(r.soc + bat.e_fcr_h * f + bat.e_afrr_h * w <= bat.soc_max * E + 1e-7)


def test_fcr_only_fills_power():
    r = solve_day(np.full(96, 50.0), BLOCK, Battery(duration_h=1.0), DT, fcr_price=np.full(6, 10.0),
                  markets=("fcr",))
    np.testing.assert_allclose(r.fcr, 1.0, atol=1e-7)
    assert r.objective == pytest.approx(10.0 * 24, rel=1e-6)


def test_afrr_energy_headroom_limits_short_battery():
    bat = Battery(duration_h=1.0, e_afrr_h=1.0)
    r = solve_day(np.full(96, 50.0), BLOCK, bat, DT, afrr_pos_price=np.full(6, 10.0), markets=("afrr",))
    # SOC fixed at 0.5 MWh without DA; headroom above E_min = 0.45 MWh -> 0.45 MW for 1 h
    np.testing.assert_allclose(r.afrr_pos, 0.45, atol=1e-7)


def test_stack_dominates_single_markets():
    rng = np.random.default_rng(1)
    p = 60 + 50 * np.sin(np.linspace(0, 4 * np.pi, 96)) + rng.normal(0, 10, 96)
    kw = dict(fcr_price=np.full(6, 4.0), afrr_pos_price=np.full(6, 3.0), afrr_neg_price=np.full(6, 2.0))
    bat = Battery(duration_h=2.0)
    stack = solve_day(p, BLOCK, bat, DT, **kw).objective
    for mk in (("da",), ("fcr",), ("afrr",), ("da", "fcr")):
        assert stack >= solve_day(p, BLOCK, bat, DT, markets=mk, **kw).objective - 1e-6


def test_cycle_limit_respected():
    p = np.tile(np.r_[np.full(8, 0.0), np.full(8, 200.0)], 6)
    bat = Battery(duration_h=1.0, max_cycles_per_day=1.0)
    r = solve_day(p, BLOCK, bat, DT, markets=("da",))
    assert r.discharge.sum() * DT <= bat.energy_mwh + 1e-7


def test_fixed_reserves_are_honoured():
    fixed = (np.full(6, 0.3), np.full(6, 0.2), np.zeros(6))
    r = solve_day(two_price_day(), BLOCK, Battery(), DT, markets=("da",), fixed_reserves=fixed)
    np.testing.assert_allclose(r.fcr, 0.3)
    np.testing.assert_allclose(r.afrr_pos, 0.2)
    assert np.all(r.discharge <= 0.5 + 1e-7)


def test_optimal_bid():
    b, p = optimal_bid(np.array([1.0, 2.0, 3.0, 10.0]))
    assert b == 10.0 and p == 0.25          # 10*0.25 = 2.5 > 3*0.5, 2*0.75, 1*1
    b, p = optimal_bid(np.array([5.0, 5.0, 5.0, 1.0]))
    assert b == 5.0 and p == 0.75
    assert optimal_bid(np.array([np.nan])) == (0.0, 0.0)


def test_naive_forecast_dst_day():
    idx = pd.date_range("2026-03-28", "2026-03-30", freq="15min", tz="Europe/Berlin", inclusive="left")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    prev, day = s[s.index.date == pd.Timestamp("2026-03-28").date()], s[s.index.date == pd.Timestamp("2026-03-29").date()]
    assert len(day) == 92
    _, block, slots = day_arrays(day)
    fc = naive_da_forecast(prev, slots)
    assert len(fc) == 92 and np.isfinite(fc).all()
    assert block.max() == 5


def test_ex_ante_never_beats_perfect_foresight():
    rng = np.random.default_rng(3)
    bat = Battery(duration_h=2.0)
    for _ in range(5):
        p = 70 + 40 * np.sin(np.linspace(0, 2 * np.pi, 96)) + rng.normal(0, 25, 96)
        rr = pd.DataFrame({"fcr_eur_mw_h": rng.uniform(2, 15, 6), "afrr_pos_marg": rng.uniform(1, 20, 6),
                           "afrr_neg_marg": rng.uniform(1, 20, 6)})
        fc = p + rng.normal(0, 30, 96)
        bids = ((rng.uniform(1, 20, 6), np.full(6, 0.5)), (rng.uniform(1, 20, 6), np.full(6, 0.5)))
        pf = run_day(p, BLOCK, bat, rr, "pf_stack")
        for s in ("xa_stack", "xa_stack_pfda"):
            xa = run_day(p, BLOCK, bat, rr, s, fc_price=fc, fcr_fc=rr["fcr_eur_mw_h"].to_numpy() * 0.8, bids=bids)
            tot = lambda d: d["da"] + d["fcr"] + d["afrr_pos"] + d["afrr_neg"] + d["deg"]
            assert tot(xa) <= tot(pf) + 1e-6


def test_settle_matches_objective_under_pf():
    rng = np.random.default_rng(4)
    p = rng.normal(80, 40, 96)
    bat = Battery()
    fp, up, wp = np.full(6, 5.0), np.full(6, 3.0), np.full(6, 2.0)
    r = solve_day(p, BLOCK, bat, DT, fp, up, wp)
    s = settle(p, BLOCK, r, bat, DT, fp, up, wp)
    assert s["da"] + s["fcr"] + s["afrr_pos"] + s["afrr_neg"] + s["deg"] == pytest.approx(r.objective, rel=1e-6)
