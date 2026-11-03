"""Year-long daily backtest: perfect-foresight bounds and an ex-ante (non-anticipative) strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Battery, optimal_bid, settle, solve_day

DT = 0.25


def day_arrays(g: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """price, block index and (hour*4+quarter) slot per step for one local delivery day."""
    idx = g.index
    return g.to_numpy(float), (idx.hour // 4).to_numpy(), (idx.hour * 4 + idx.minute // 15).to_numpy()


def naive_da_forecast(prev: pd.Series, slots: np.ndarray) -> np.ndarray:
    """Previous delivery day's price at the same local quarter-hour (known before the D-1 auctions)."""
    _, _, ps = day_arrays(prev)
    m = pd.Series(prev.to_numpy(float), index=ps).groupby(level=0).mean()
    return m.reindex(range(96)).ffill().bfill().to_numpy()[slots]


def run_day(price, block, bat: Battery, res_row: pd.DataFrame, strategy: str,
            fc_price=None, fcr_fc=None, bids=None) -> dict:
    """Return realised revenue dict for one day and strategy."""
    fcr = res_row["fcr_eur_mw_h"].to_numpy()
    pos_m, neg_m = res_row["afrr_pos_marg"].to_numpy(), res_row["afrr_neg_marg"].to_numpy()
    z = np.zeros(6)
    if strategy == "pf_stack_avg":
        # perfect foresight, but aFRR paid the volume-weighted average awarded price (a typical
        # pay-as-bid bidder rather than one bidding exactly at the marginal price)
        pa, na = res_row["afrr_pos_avg"].to_numpy(), res_row["afrr_neg_avg"].to_numpy()
        r = solve_day(price, block, bat, DT, fcr, pa, na)
        return settle(price, block, r, bat, DT, fcr, pa, na)
    if strategy.startswith("pf_"):
        mk = {"pf_da": ("da",), "pf_fcr": ("fcr",), "pf_da_fcr": ("da", "fcr"),
              "pf_da_afrr": ("da", "afrr"), "pf_stack": ("da", "fcr", "afrr")}[strategy]
        r = solve_day(price, block, bat, DT, fcr, pos_m, neg_m, markets=mk)
        return settle(price, block, r, bat, DT, fcr, pos_m, neg_m)
    if strategy == "xa_da":
        r = solve_day(fc_price, block, bat, DT, markets=("da",))
        return settle(price, block, r, bat, DT, z, z, z)
    if strategy == "xa_da_fcr":
        r1 = solve_day(fc_price, block, bat, DT, fcr_fc, markets=("da", "fcr"))
        r2 = solve_day(fc_price, block, bat, DT, markets=("da",), fixed_reserves=(r1.fcr, z, z))
        out = settle(price, block, r2, bat, DT, fcr, z, z)
        out["fcr_mw_avg"] = float(np.mean(r1.fcr))
        return out
    # ex-ante stack: stage 1 on forecasts / expected bid values, then auctions clear
    (bp, pp), (bn, pn) = bids
    r1 = solve_day(fc_price, block, bat, DT, fcr_fc, bp * pp, bn * pn, markets=("da", "fcr", "afrr"))
    acc_p = (bp <= pos_m + 1e-9) & (r1.afrr_pos > 1e-6)
    acc_n = (bn <= neg_m + 1e-9) & (r1.afrr_neg > 1e-6)
    fixed = (r1.fcr, np.where(acc_p, r1.afrr_pos, 0.0), np.where(acc_n, r1.afrr_neg, 0.0))
    da_input = price if strategy == "xa_stack_pfda" else fc_price
    r2 = solve_day(da_input, block, bat, DT, markets=("da",), fixed_reserves=fixed)
    out = settle(price, block, r2, bat, DT, fcr, np.where(acc_p, bp, 0.0), np.where(acc_n, bn, 0.0))
    out["afrr_pos_offered_mwh"] = float(np.sum(r1.afrr_pos * np.bincount(block, minlength=6) * DT))
    out["afrr_pos_accepted_mwh"] = float(np.sum(fixed[1] * np.bincount(block, minlength=6) * DT))
    out["afrr_neg_offered_mwh"] = float(np.sum(r1.afrr_neg * np.bincount(block, minlength=6) * DT))
    out["afrr_neg_accepted_mwh"] = float(np.sum(fixed[2] * np.bincount(block, minlength=6) * DT))
    out["fcr_mw_avg"] = float(np.mean(r1.fcr))
    out["afrr_pos_mw_avg"] = float(np.mean(fixed[1]))
    out["afrr_neg_mw_avg"] = float(np.mean(fixed[2]))
    return out


STRATEGIES = ("pf_da", "pf_fcr", "pf_da_fcr", "pf_da_afrr", "pf_stack", "pf_stack_avg", "xa_da", "xa_da_fcr", "xa_stack", "xa_stack_pfda")


def backtest(da_days: dict, reserves: pd.DataFrame, bat: Battery, eval_days: list[str],
             strategies=STRATEGIES, bid_window: int = 28) -> pd.DataFrame:
    res = reserves.set_index(["date", "block"]).sort_index()
    all_days = sorted(da_days)
    rows = []
    for day in eval_days:
        g = da_days[day]
        price, block, slots = day_arrays(g)
        prev_day = all_days[all_days.index(day) - 1]
        rr = res.loc[day].reindex(range(6))
        fc = naive_da_forecast(da_days[prev_day], slots)
        fcr_fc = res.loc[prev_day].reindex(range(6))["fcr_eur_mw_h"].to_numpy()
        d0 = pd.Timestamp(day)
        hist_days = [(d0 - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(1, bid_window + 1)]
        hist = res.loc[res.index.get_level_values(0).isin(hist_days)]
        bp = np.array([optimal_bid(hist.xs(b, level=1)["afrr_pos_marg"].to_numpy()) for b in range(6)])
        bn = np.array([optimal_bid(hist.xs(b, level=1)["afrr_neg_marg"].to_numpy()) for b in range(6)])
        bids = ((bp[:, 0], bp[:, 1]), (bn[:, 0], bn[:, 1]))
        for s in strategies:
            o = run_day(price, block, bat, rr, s, fc_price=fc, fcr_fc=fcr_fc, bids=bids)
            o.update(date=day, strategy=s)
            rows.append(o)
    df = pd.DataFrame(rows)
    df["total"] = df[["da", "fcr", "afrr_pos", "afrr_neg", "deg"]].sum(axis=1)
    return df
